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
from utils.attention_block import *

_tokenizer = _Tokenizer()


class PromptLearner(Base_PromptLearner):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__(cfg, classnames, clip_model)
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.APT.N_CTX
        ctx_init = cfg.TRAINER.APT.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]   # text encoder hidden size(512)
        self.dim = clip_model.text_projection.shape[1]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        
        self.tp = cfg.TRAINER.APT.TP
        self.vp = cfg.TRAINER.APT.VP
        self.t_deep = cfg.TRAINER.APT.T_DEEP
        self.v_deep = cfg.TRAINER.APT.V_DEEP
        self.deep_share = cfg.TRAINER.APT.DEEP_SHARED
        self.share_layer = cfg.TRAINER.APT.SHARE_LAYER
        self.num_tokens = cfg.TRAINER.APT.NUM_TOKENS    # number of prompted tokens
        self.deep_layer = cfg.TRAINER.APT.DEEP_LAYERS # num of layer has prompt ([1,3]: 1~3 layer has)
        self.location = cfg.TRAINER.APT.LOCATION
        self.prompt_dropout = nn.Dropout(cfg.TRAINER.APT.DROPOUT)
        self.num_layer = cfg.MODEL.NUM_LAYER
        self.hidden_size = cfg.MODEL.HIDDEN_SIZE    # visual encoder hiden size(768)
        
        self.ctx = None
        if self.tp:
            if ctx_init and n_ctx <= 4:   # use given words to initialize context vectors
                ctx_init = ctx_init.replace("_", " ") 
                prompt = clip.tokenize(ctx_init)
                with torch.no_grad():
                    embedding = clip_model.token_embedding(prompt).type(dtype)
                ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
                self.ctx = nn.Parameter(ctx_vectors)
                nn.init.normal_(ctx_vectors, std=0.02)
                prompt_prefix = ctx_init
            else:
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
                nn.init.normal_(ctx_vectors, std=0.02)
                prompt_prefix = " ".join(["X"] * n_ctx)
            self.ctx = nn.Parameter(ctx_vectors)
                
        self.vctx = None
        self.proj = None
        if self.vp:
            if self.share_layer != None:
                if self.share_layer[0] == 0:
                    self.proj = nn.Linear(ctx_dim, self.hidden_size).half()
                else:
                    vctx_vectors = torch.empty(n_ctx, self.hidden_size, dtype=dtype)
                    nn.init.normal_(vctx_vectors, std=0.02)
                    self.vctx = nn.Parameter(vctx_vectors)
            else:
                vctx_vectors = torch.empty(n_ctx, self.hidden_size, dtype=dtype)
                nn.init.normal_(vctx_vectors, std=0.02)
                self.vctx = nn.Parameter(vctx_vectors)
        
        self.deep_ctx = None
        if self.t_deep:
            if self.deep_layer == None:
                deep_ctx_vectors = torch.empty(self.num_layer - 1, self.num_tokens, ctx_dim)
            else:
                deep_ctx_vectors = torch.empty(self.deep_layer[1] - self.deep_layer[0] + 1, self.num_tokens, ctx_dim)
            nn.init.normal_(deep_ctx_vectors, std=0.02)
            self.deep_ctx = nn.Parameter(deep_ctx_vectors)
        
        self.deep_vctx = None
        if self.v_deep and not self.deep_share:    
            if self.deep_layer == None:
                deep_vctx_vectors = torch.empty(self.num_layer - 1, self.num_tokens, self.hidden_size)
            elif self.deep_layer != None:
                deep_vctx_vectors = torch.empty(self.deep_layer[1] - self.deep_layer[0] - 1, self.num_tokens, self.hidden_size)
            nn.init.normal_(deep_vctx_vectors, std=0.02)
            self.deep_vctx = nn.Parameter(deep_vctx_vectors) 
            
        elif self.v_deep and self.deep_share:
            single_layer = nn.Linear(ctx_dim, self.hidden_size)   
            if self.share_layer == None and self.deep_layer == None:  
                deep_vctx_vectors = torch.empty(self.num_layer - 1, self.num_tokens, self.hidden_size)
                self.deep_prompt_proj = get_clones(single_layer, self.num_layer - 1)
            elif self.share_layer != None and self.deep_layer == None:
                deep_vctx_vectors = torch.empty(self.num_layer - self.share_layer[1] - 1, self.num_tokens, self.hidden_size)
                self.deep_prompt_proj = get_clones(single_layer, self.share_layer[1] - self.share_layer[0] + 1)
            elif self.share_layer != None and self.deep_layer != None:
                deep_vctx_vectors = torch.empty(self.deep_layer[1] - self.share_layer[1], self.num_tokens, self.hidden_size)
                self.deep_prompt_proj = get_clones(single_layer, self.share_layer[1] - self.share_layer[0] + 1)
            else:
                raise ValueError('deep layer and share layer are not compatible!')
            nn.init.normal_(deep_vctx_vectors, std=0.02)
            self.deep_vctx = nn.Parameter(deep_vctx_vectors)
                
        print('APT design: Attention-based Prompt Tuning for UDA')
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of APT context words (tokens): {n_ctx}")
        
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
        self.tokenized_prompts = tokenized_prompts  
        self.name_lens = name_lens

        self.attn_block = APT_ATTN_Block(clip_model, beta_s=0.1, beta_t=0.1)
        self.K = 5
        self.dim = clip_model.text_projection.shape[1]
        source_feat_bank = torch.zeros((self.n_cls * self.K, self.dim)).half()
        target_feat_bank = torch.zeros((self.n_cls * self.K, self.dim)).half()
        self.source_feat_bank = nn.Parameter(source_feat_bank)
        self.target_feat_bank = nn.Parameter(target_feat_bank)        

    def forward(self):
        if self.proj != None:
            vctx = self.proj(self.ctx)
        else:
            vctx = self.vctx

        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)   # [65, 16, 512]

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)
        
        if self.deep_share:
            deep_vctx = []
            for index, layer in enumerate(self.deep_prompt_proj):
                deep_vctx.append(layer(self.deep_ctx[index]))
            deep_vctx = torch.stack(deep_vctx)
            deep_vctx = torch.cat((deep_vctx, self.deep_vctx), dim=0)
        else:
            deep_vctx = self.deep_vctx
            
        return prompts, self.deep_ctx, vctx, deep_vctx


class CustomCLIP(Base_CustomCLIP):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__(cfg, classnames, clip_model)
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.K = self.prompt_learner.K
        self.dim = clip_model.text_projection.shape[1]
        
        self.text_encoder = TextEncoder(cfg, clip_model, self.prompt_learner)
        
        self.n_cls = len(classnames)
        if cfg.MODEL.BACKBONE.NAME.split('-')[0] == 'ViT':
            self.image_encoder = ImageEncoder_Trans(cfg, clip_model)
        else:  # RN50, RN101
            self.image_encoder = ImageEncoder_Conv(cfg, clip_model)
        
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        
        self.text_encoder_u = Simple_TextEncoder(clip_model)
        prompt_prefix = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts_u = [prompt_prefix.format(c.replace("_", " ")) for c in classnames]
        self.tokenized_prompts_u = clip.tokenize(prompts_u)
        
        self.source_key_dict =  {i: i for i in range(self.n_cls * self.K)}
        self.target_key_dict =  {i: i for i in range(self.n_cls * self.K)}
        self.source_max_probs_list = [0.0 for i in range(self.n_cls * self.K)]
        self.target_max_probs_list = [0.0 for i in range(self.n_cls * self.K)]
        self.all_s, self.right_s, self.all_fs, self.right_fs = 0, 0, 0, 0
        self.all_t, self.right_t, self.all_ft, self.right_ft = 0, 0, 0, 0
        
        self.confi = cfg.CONFI
        self.epoch = cfg.EPOCH
        self.warm_up = cfg.WARM_UP

    def forward(self, image, label=None, epoch=None, train=False, construct=False, source=True):
        prompts, deep_ctx, vctx, deep_vctx = self.prompt_learner()
                
        text_features = self.text_encoder(prompts, self.tokenized_prompts, deep_ctx)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        image_features = self.image_encoder(image.type(self.dtype), vctx, deep_vctx)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True) 
        
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        
        text_features_u = self.text_encoder_u(self.tokenized_prompts_u.to(self.logit_scale.device))
        text_features_u = text_features_u / text_features_u.norm(dim=-1, keepdim=True)
        
        F_u = self.image_encoder(image.type(self.dtype), None, None)   
        F_u = F_u / F_u.norm(dim=-1, keepdim=True)

        if construct:
            logits_u = logit_scale * F_u @ text_features_u.t()
            pseudo_label = torch.softmax(logits_u, dim=-1)
            max_probs, label_p = torch.max(pseudo_label, dim=-1)
            
            if source:
                for i, l in enumerate(label):
                    if l == label_p[i]:
                        index = l.item() * self.K
                        l_list = self.source_max_probs_list[index: index + self.K]
                        if max_probs[i] > min(l_list):
                            min_index = l_list.index(min(l_list))
                            self.source_max_probs_list[index+min_index] = max_probs[i]
                            self.prompt_learner.source_feat_bank[index+min_index] = F_u[i]  
                            self.source_key_dict[index+min_index] = label_p[i]
            else:
                for i, l in enumerate(label_p):
                    index = l.item() * self.K
                    l_list = self.target_max_probs_list[index: index + self.K]
                    if max_probs[i] > min(l_list):
                        min_index = l_list.index(min(l_list))
                        self.target_max_probs_list[index+min_index] = max_probs[i]
                        self.prompt_learner.target_feat_bank[index+min_index] = F_u[i]    
                        self.target_key_dict[index+min_index] = label_p[i]
                        
            return      
            
        elif train:
            source_bank = torch.mean(self.prompt_learner.source_feat_bank.reshape(self.n_cls, self.K, self.dim), dim=1)
            target_bank = torch.mean(self.prompt_learner.target_feat_bank.reshape(self.n_cls, self.K, self.dim), dim=1)
            logits_c = self.prompt_learner.attn_block(text_features, image_features, source_bank, target_bank)
            
            if source:
                loss_c = F.cross_entropy(logits_c, label) 
                loss_x = F.cross_entropy(logits, label) 
                
                return logits, loss_x+loss_c
            
            else:
                if epoch == None or epoch <= self.epoch:
                    logits_u = logit_scale * F_u @ text_features_u.t()
                    pseudo_label = torch.softmax(logits_u, dim=-1)
                else:
                    pseudo_label = torch.softmax(logits, dim=-1)
                max_probs, label_p = torch.max(pseudo_label, dim=-1)

                mask = max_probs.ge(self.confi).float()
                if mask.sum() == 0 or self.warm_up > epoch:
                    loss_c = torch.tensor(0.)
                    loss_x = torch.tensor(0.)
                else:
                    loss_c = (F.cross_entropy(logits_c, label_p, reduction="none") * mask).sum() / mask.sum()
                    loss_x = (F.cross_entropy(logits, label_p, reduction="none") * mask).sum() / mask.sum()
                
                return loss_x+loss_c
        
        else:      
            source_bank = torch.mean(self.prompt_learner.source_feat_bank.reshape(self.n_cls, self.K, self.dim), dim=1)
            target_bank = torch.mean(self.prompt_learner.target_feat_bank.reshape(self.n_cls, self.K, self.dim), dim=1)
            logits_c = self.prompt_learner.attn_block(text_features, image_features, source_bank, target_bank)

            return logits + 0.5 * logits_c


@TRAINER_REGISTRY.register()
class APT(BaseDA):
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        self.domains = cfg.DOMAINS
        self.save = cfg.SAVE_MODEL

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.APT.PREC == "fp32" or cfg.TRAINER.APT.PREC == "amp":
            clip_model.float()  # CLIP's default precision is fp16

        print("Building custom CLIP...")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder...")
        for name, param in self.model.named_parameters():
            param.requires_grad_(False)
            if "prompt_learner" in name:
                param.requires_grad_(True)
            if "bank" in name:
                param.requires_grad_(False)
                
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
        self.scaler = GradScaler() if cfg.TRAINER.APT.PREC == "amp" else None

        self.construct_bank()
        
    def forward_backward(self, batch_x, batch_u):
        prec = self.cfg.TRAINER.APT.PREC
        image_x, label, image_u = self.parse_batch_train(batch_x, batch_u)

        if prec == "amp":
            with autocast():  
                output_x, loss_x = self.model(image_x, label, epoch=self.epoch, train=True)
                loss_u = self.model(image_u, epoch=self.epoch, train=True, source=False)   
                loss = loss_x + loss_u

            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output_x, loss_x = self.model(image_x, label, epoch=self.epoch, train=True)
            loss_u = self.model(image_u, epoch=self.epoch, train=True, source=False)   
            loss = loss_x + loss_u
            self.model_backward_and_update(loss)  
            
        loss_summary = {
            "loss": loss.item(),
            "loss_x": loss_x.item(),
            "loss_u": loss_u.item(),
            "acc_x": compute_accuracy(output_x, label)[0].item(),
        }
        
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary
    
    def parse_batch_train(self, batch_x, batch_u):
        input = batch_x["img"]
        label = batch_x["label"]
        input_u = batch_u["img"]
        
        input = input.to(self.device)
        label = label.to(self.device)
        input_u = input_u.to(self.device)
        return input, label, input_u
    
    @torch.no_grad()
    def construct_bank(self):
        self.set_model_mode("eval")
        
        print("Constructing source feature bank...")
        data_loader_x = self.train_loader_x
        for batch_idx, batch in enumerate(data_loader_x):
            input, label = self.parse_batch_test(batch)
            self.model(input, label=label, construct=True)
            if min(self.model.source_max_probs_list) > 0.99:  
                break
            
        print("Constructing target feature bank...")
        data_loader_u = self.train_loader_u
        for batch_idx, batch in enumerate(data_loader_u):
            input, label = self.parse_batch_test(batch)
            self.model(input, label=label, construct=True, source=False)
            if min(self.model.target_max_probs_list) > 0.99:
                break
        
        print('Feature banks are completed!')    

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
            output = self.model_inference(input)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()
        for k, v in results.items():
            tag = "{}/{}".format(split, k)
            self.write_scalar(tag, v, self.epoch)

        results_all = results["accuracy"]

        return results_all
             
            