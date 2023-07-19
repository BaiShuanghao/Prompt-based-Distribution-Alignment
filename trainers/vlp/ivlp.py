'''
@inproceedings{khattak2023maple,
  title={Maple: Multi-modal prompt learning},
  author={Khattak, Muhammad Uzair and Rasheed, Hanoona and Maaz, Muhammad and Khan, Salman and Khan, Fahad Shahbaz},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={19113--19122},
  year={2023}
}

Adapted from https://github.com/muzairkhattak/multimodal-prompt-learning
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
from utils.templates import CUSTOM_TEMPLATES

_tokenizer = _Tokenizer()


class PromptLearner(Base_PromptLearner):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__(cfg, classnames, clip_model)
        # **************** Text Prompt ****************
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.IVLP.N_CTX
        self.num_tokens = cfg.TRAINER.IVLP.NUM_TOKENS    # number of prompted tokens
        self.deep_layer = cfg.TRAINER.IVLP.DEEP_LAYERS
        ctx_init = cfg.TRAINER.IVLP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        self.t_deep = cfg.TRAINER.IVLP.T_DEEP
        self.v_deep = cfg.TRAINER.IVLP.V_DEEP
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        self.ctx = None
        if cfg.TRAINER.IVLP.TP:
            if ctx_init and n_ctx <= 4:
                prompt_prefix = ctx_init
                ctx_init = ctx_init.replace("_", " ")
                prompt = clip.tokenize(ctx_init)
                with torch.no_grad():
                    embedding = clip_model.token_embedding(prompt).type(dtype)
                ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            else:
                prompt_prefix = " ".join(["X"] * n_ctx)
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
                nn.init.normal_(ctx_vectors, std=0.02)
            self.ctx = nn.Parameter(ctx_vectors)
        
        self.deep_ctx = None
        if self.t_deep:  
            if self.deep_layer == None:
                deep_ctx_vectors = torch.empty(cfg.MODEL.NUM_LAYER - 1, self.num_tokens, ctx_dim)
            else:
                deep_ctx_vectors = torch.empty(self.deep_layer[1] - self.deep_layer[0] + 1, self.num_tokens, ctx_dim)
            nn.init.normal_(deep_ctx_vectors, std=0.02)
            self.deep_ctx = nn.Parameter(deep_ctx_vectors)
 
        print(f"Independent V-L design")
        print(f'Initial text context: "{prompt_prefix}"')
        print(f"Number of context words (tokens) for Language prompting: {n_ctx}")
        print(f"Number of context words (tokens) for Vision prompting: {cfg.TRAINER.IVLP.NUM_TOKENS}")
        
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        
        # **************** Visual Prompt ****************
        prompt_dim = cfg.MODEL.HIDDEN_SIZE
        self.prompt_dropout = nn.Dropout(cfg.TRAINER.IVLP.DROPOUT)
        self.location = cfg.TRAINER.IVLP.LOCATION
        
        self.vctx = None
        if cfg.TRAINER.IVLP.VP:
            vctx_vectors = torch.empty(self.num_tokens, prompt_dim)
            nn.init.normal_(vctx_vectors, std=0.02)
            self.vctx = nn.Parameter(vctx_vectors)
            
        self.deep_vctx = None
        if self.v_deep: 
            if self.deep_layer == None:
                deep_vctx_vectors = torch.empty(cfg.MODEL.NUM_LAYER - 1, self.num_tokens, prompt_dim)
            else:
                deep_vctx_vectors = torch.empty(self.deep_layer[1] - self.deep_layer[0] + 1, self.num_tokens, prompt_dim)
            nn.init.normal_(deep_vctx_vectors, std=0.02)
            self.deep_vctx = nn.Parameter(deep_vctx_vectors)
    
    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)
            
        return prompts, self.deep_ctx, self.vctx, self.deep_vctx
        

class CustomCLIP(Base_CustomCLIP):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__(cfg, classnames, clip_model)
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        
        self.text_encoder = TextEncoder(cfg, clip_model, self.prompt_learner)
        
        if cfg.MODEL.BACKBONE.NAME.split('-')[0] == 'ViT':
            self.image_encoder = ImageEncoder_Trans(cfg, clip_model, self.prompt_learner)
        else:  # RN50, RN101
            raise ValueError('For IVLP, backbone must be ViT!')
        
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        
        self.text_encoder_u = Simple_TextEncoder(clip_model)
        prompt_prefix = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts_u = [prompt_prefix.format(c.replace("_", " ")) for c in classnames]
        self.tokenized_prompts_u = clip.tokenize(prompts_u)

    def forward(self, image, prompt=False):
        prompts, deep_ctx, vctx, deep_vctx = self.prompt_learner()
        if prompt:
            text_features = self.text_encoder_u(self.tokenized_prompts_u.to(self.logit_scale.device))
        else:
            text_features = self.text_encoder(prompts, self.tokenized_prompts, deep_ctx) 
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        if prompt:
            image_features = self.image_encoder(image.type(self.dtype), None, None) 
        else:   
            image_features = self.image_encoder(image.type(self.dtype), vctx, deep_vctx)    
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


@TRAINER_REGISTRY.register()
class IVLP(BaseDA):
    '''Independent V-L prompting (IVLP)
    
    Adapt from MaPLe: Multi-modal Prompt Learning
    https://arxiv.org/abs/2210.03117
    '''
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        self.domains = cfg.DOMAINS
        self.save = cfg.SAVE_MODEL

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.IVLP.PREC == "fp32" or cfg.TRAINER.IVLP.PREC == "amp":
            clip_model.float()  # CLIP's default precision is fp16

        print("Building custom CLIP...")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder...")
        for name, param in self.model.named_parameters():
            param.requires_grad_(False)
            if "prompt_learner" in name:
                param.requires_grad_(True)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {sorted(enabled)}")
        print("# params: {:,}".format(count_num_param(self.model.prompt_learner)))

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.IVLP.PREC == "amp" else None
           
    def forward_backward(self, batch_x, batch_u):
        image_x, label, image_u = self.parse_batch_train(batch_x, batch_u)

        prec = self.cfg.TRAINER.IVLP.PREC
        if prec == "amp":
            with autocast():
                output_x = self.model(image_x)
                loss = F.cross_entropy(output_x, label)
            
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output_x = self.model(image_x)   
            loss = F.cross_entropy(output_x, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output_x, label)[0].item(),   
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary
