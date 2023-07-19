'''
@article{ge2022domain,
  title={Domain adaptation via prompt learning},
  author={Ge, Chunjiang and Huang, Rui and Xie, Mixue and Lai, Zihang and Song, Shiji and Li, Shuang and Huang, Gao},
  journal={arXiv preprint arXiv:2202.06687},
  year={2022}
}

Adapted from https://github.com/LeapLabTHU/DAPrompt
'''

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, count_num_param
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from trainers.baseda import *
from utils.clip_part import *

_tokenizer = _Tokenizer()


class PromptLearner(Base_PromptLearner):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__(cfg, classnames, clip_model)
        # clip
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]   # dim of last fc
        clip_imsize = clip_model.visual.input_resolution    # img size
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # class & domain
        n_cls = len(classnames)     # num of class
        self.n_cls = n_cls
        n_ctx = cfg.TRAINER.DAPL.N_CTX  # len of class-specific context
        self.n_ctx = n_ctx
        n_dm = len(cfg.DATASET.SOURCE_DOMAINS) + len(cfg.DATASET.TARGET_DOMAINS)  # num of domain
        n_dmx = cfg.TRAINER.DAPL.N_DMX  # len of domain-specific context
        n = n_dmx + n_ctx
        self.n_dm = n_dm
        self.n_dmx = n_dmx
        print(f"Number of context words (tokens): {n_ctx}")
        print(f"Number of domain context words (tokens): {n_dmx}")

        # class learnable prompt
        if cfg.TRAINER.DAPL.CSC:
            # each class has a specific context
            print("Initializing class-specific contexts")
            ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
        else:
            # all classes have a share context
            print("Initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)  # class prompt is to be optimized
        print("ctx vectors size: ".format(ctx_vectors.size()))

        # domain learnable prompt
        # each domain has a specific context
        domain_vectors = torch.empty(n_dm, n_dmx, ctx_dim, dtype=dtype)
        nn.init.normal_(domain_vectors, std=0.02)
        self.dmx = nn.Parameter(domain_vectors)
        print("dmx vectors size: ".format(domain_vectors.size()))

        # complete prompt
        prompt_prefix = " ".join(["X"] * n)     # "X X ... X"
        naive_prompt_prefix = "a photo of a".replace(
            "_", " ")  # "a photo of a"
        print(f'Initial context: "{prompt_prefix}"')
        print(f'Initial naive context: "{naive_prompt_prefix}"')

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        self.name_lens = name_lens
        domainnames = cfg.DATASET.SOURCE_DOMAINS + cfg.DATASET.TARGET_DOMAINS
        domainnames = [", a {} image".format(domain) for domain in domainnames]

        prompts = [  # 'X X ... X aeroplane , a synthetic image.'
            prompt_prefix + " " + name + " " + domain + "."
            # （cd, -1）= (n_cls*n_dm，-1)
            for domain in domainnames for name in classnames
        ]
        naive_prompts = [   # 'a photo of a aeroplane.'
            naive_prompt_prefix + " " + name + "." for name in classnames
        ]

        tokenized_prompts = torch.cat(
            [clip.tokenize(p) for p in prompts])  # [cd, 77]
        naive_tokenized_prompts = torch.cat(
            [clip.tokenize(p) for p in naive_prompts])  # [c, 77]

        with torch.no_grad():
            embedding = clip_model.token_embedding(
                tokenized_prompts).type(dtype)   # [cd, 1+32+7+20, 512]
            naive_embedding = clip_model.token_embedding(
                naive_tokenized_prompts).type(dtype)    # [c, 77, 512]

        tokenized_prompts = torch.cat(
            [tokenized_prompts, naive_tokenized_prompts])
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n:, :])  # CLS, EOS

        self.csc = cfg.TRAINER.DAPL.CSC
        self.tokenized_prompts = tokenized_prompts
        self.naive_embedding = naive_embedding.to(torch.device("cuda:{}".format(cfg.GPU)))    

    def forward(self):
        ctx = self.ctx
        ctx_dim = ctx.size(-1)
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_dm, -1, -1)  # [n_dm, 16, 512]
            if not self.csc:
                ctx = ctx.unsqueeze(1).expand(-1, self.n_cls, -1, -1)   # [n_dm, n_cls, 16, 512]
        else:
            ctx = ctx.unsqueeze(0).expand(self.n_dm, -1, -1, -1)  # [n_dm, n_cls, 16, 512]

        dmx = self.dmx  # [n_dm, 16, 512]
        dmx = dmx.unsqueeze(1).expand(-1, self.n_cls, -1, -1)  # [n_dm, n_cls, 16, 512]

        ctxdmx = torch.cat([ctx, dmx], dim=2).reshape(
            self.n_cls * self.n_dm, self.n_ctx + self.n_dmx, ctx_dim)   # [24, 32, 512]

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = torch.cat(
            [
                prefix,  # (cd, 1, dim)
                ctxdmx,  # (cd, 32, dim)
                suffix,  # (cd, *, dim)
            ],
            dim=1,  # [24, 77, 512]
        )
        prompts = torch.cat([prompts, self.naive_embedding], dim=0) # [36, 77, 512]

        return prompts


class CustomCLIP(Base_CustomCLIP):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__(cfg, classnames, clip_model)
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        
        self.text_encoder = TextEncoder(cfg, clip_model, self.prompt_learner)
        self.image_encoder = clip_model.visual
       
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, self.tokenized_prompts)   # [36, 77, 512]
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()   # [b, 512] x [512, 12 * 3]

        return logits


@TRAINER_REGISTRY.register()
class DAPL(BaseDA):
    """Domain Adaptation via Prompt Learning(DAPL).

    Domain Adaptation via Prompt Learning
    https://arxiv.org/abs/2202.06687
    """
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        self.n_cls = len(classnames)
        self.domains = cfg.DOMAINS
        self.save = cfg.SAVE_MODEL

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.DAPL.PREC == "fp32" or cfg.TRAINER.DAPL.PREC == "amp":
            clip_model.float()  # CLIP's default precision is fp16

        print("Building custom CLIP...")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        # plus one for pseudo label
        self.n_dm = self.model.prompt_learner.n_dm + 1
        self.n_cls = self.model.prompt_learner.n_cls

        print("Turning off gradients in both the image and the text encoder...")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)
        print("# params: {:,}".format(count_num_param(self.model.prompt_learner)))

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(
                self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        # transform the epoch to step schedule
        len_train_loader_x = len(self.train_loader_x)
        len_train_loader_u = len(self.train_loader_u)
        if self.cfg.TRAIN.COUNT_ITER == "train_x":
            self.num_batches = len_train_loader_x
        elif self.cfg.TRAIN.COUNT_ITER == "train_u":
            self.num_batches = len_train_loader_u
        elif self.cfg.TRAIN.COUNT_ITER == "smaller_one":
            self.num_batches = min(len_train_loader_x, len_train_loader_u)
        else:
            raise ValueError('Training batch name is wrong!')

        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        self.scaler = GradScaler() if cfg.TRAINER.DAPL.PREC == "amp" else None
    
    def forward_backward(self, batch_x, batch_u):
        # label_u only used for matric
        image_x, label, image_u = self.parse_batch_train(batch_x, batch_u)
        prec = self.cfg.TRAINER.DAPL.PREC
        if prec == "amp":
            with autocast():    
                output_x = self.model(image_x)
                output_u = self.model(image_u)

                # only clip annotation
                pseudo_label = torch.softmax(
                    output_u[:, -self.n_cls:].reshape(-1,
                                                      self.n_cls) / self.cfg.TRAINER.DAPL.T,
                    dim=-1,
                )
                max_probs, label_p = torch.max(pseudo_label, dim=-1)
                mask = max_probs.ge(self.cfg.TRAINER.DAPL.TAU).float()

                loss_x = F.cross_entropy(output_x[:, :self.n_cls], label)
                loss_u = (F.cross_entropy(output_u[:, self.n_cls:2 * self.n_cls], label_p,
                                          reduction="none") * mask).sum() / mask.sum()
                loss = loss_x + self.cfg.TRAINER.DAPL.U * loss_u

            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output_x = self.model(image_x)
            output_u = self.model(image_u)
            # only clip annotation
            pseudo_label = torch.softmax(
                output_u[:, -self.n_cls:].reshape(-1,
                                                  self.n_cls) / self.cfg.TRAINER.DAPL.T,
                dim=-1,
            )
            max_probs, label_p = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(self.cfg.TRAINER.DAPL.TAU).float()

            loss_x = F.cross_entropy(output_x[:, :self.n_cls], label)
            if mask.sum() == 0:
                loss_u = torch.tensor(0.)
            else:
                loss_u = (F.cross_entropy(output_u[:, self.n_cls:2 * self.n_cls], label_p,
                                    reduction="none") * mask).sum() / mask.sum()
            loss = loss_x + self.cfg.TRAINER.DAPL.U * loss_u

            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "loss_x": loss_x.item(),
            "loss_u": loss_u.item(),
            "acc_x": compute_accuracy(output_x[:, :self.n_cls], label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        data_loader = self.test_loader
        print("Do evaluation on test set")

        for batch_idx, batch in enumerate(data_loader):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input).reshape(-1, self.n_dm, self.n_cls)
            # the last second slice is the logits for target domain
            output = output[:, -2, :]
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()
        for k, v in results.items():
            tag = "{}/{}".format(split, k)
            self.write_scalar(tag, v, self.epoch)

        results_all = results["accuracy"]

        return results_all