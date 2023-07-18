import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
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
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.MAPLE.N_CTX
        ctx_init = cfg.TRAINER.MAPLE.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]   # text encoder hidden size(512)
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        
        self.deep = cfg.TRAINER.MAPLE.V_DEEP
        self.num_tokens = cfg.TRAINER.MAPLE.NUM_TOKENS    # number of prompted tokens
        self.deep_layer = cfg.TRAINER.MAPLE.DEEP_LAYERS # num of layer has prompt ([1,3]: 1~3 layer has)
        self.share_layer = cfg.TRAINER.MAPLE.SHARE_LAYER        # in maple, deep_layer = share_layer
        self.location = cfg.TRAINER.MAPLE.LOCATION  
        self.prompt_dropout = nn.Dropout(cfg.TRAINER.MAPLE.DROPOUT)
        self.hidden_size = cfg.MODEL.HIDDEN_SIZE    # visual encoder hiden size(768)
        
        if ctx_init and n_ctx <= 4:
            ctx_init = ctx_init.replace("_", " ")
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        self.ctx = nn.Parameter(ctx_vectors)
        self.proj = nn.Linear(ctx_dim, self.hidden_size) # map first text dim to vis dim
        self.proj.half()
        
        self.deep_vctx = None
        if self.deep:  
            single_layer = nn.Linear(ctx_dim, self.hidden_size)
            if self.deep_layer == None:
                deep_ctx_vectors = torch.empty(cfg.MODEL.NUM_LAYER - 1, self.num_tokens, ctx_dim)
                self.deep_prompt_proj = get_clones(single_layer, cfg.MODEL.NUM_LAYER - 1)
            else:
                deep_ctx_vectors = torch.empty(self.deep_layer[1] - self.deep_layer[0] + 1, self.num_tokens, ctx_dim)
                self.deep_prompt_proj = get_clones(single_layer, self.deep_layer[1] - self.deep_layer[0] + 1)
            nn.init.normal_(deep_ctx_vectors, std=0.02)
            self.deep_ctx = nn.Parameter(deep_ctx_vectors)
                
        print('MaPLe design: Multi-modal Prompt Learning')
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of MaPLe context words (tokens): {n_ctx}")
        
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

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)
        
        if self.deep:
            deep_vctx = []
            for index, layer in enumerate(self.deep_prompt_proj):
                deep_vctx.append(layer(self.deep_ctx[index]))
            deep_vctx = torch.stack(deep_vctx)
            
            return prompts, self.deep_ctx, self.proj(self.ctx), deep_vctx
        
        return prompts, None, self.proj(self.ctx), None


class CustomCLIP(Base_CustomCLIP):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__(cfg, classnames, clip_model)
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        
        self.text_encoder = TextEncoder(cfg, clip_model, self.prompt_learner)
        
        if cfg.MODEL.BACKBONE.NAME.split('-')[0] == 'ViT':
            self.image_encoder = ImageEncoder_Trans(cfg, clip_model, self.prompt_learner)
        else:  # RN50, RN101
            raise ValueError('For MaPLe, backbone must be ViT!')
        
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
class MaPLe(BaseDA):
    '''Multi-modal Prompt Learning (MaPLe)
    
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

        if cfg.TRAINER.MAPLE.PREC == "fp32" or cfg.TRAINER.MAPLE.PREC == "amp":
            clip_model.float()  # CLIP's default precision is fp16

        print("Building custom CLIP")
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
        self.scaler = GradScaler() if cfg.TRAINER.MAPLE.PREC == "amp" else None

    def forward_backward(self, batch_x, batch_u):
        image_x, label, image_u = self.parse_batch_train(batch_x, batch_u)
        prec = self.cfg.TRAINER.MAPLE.PREC
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
   