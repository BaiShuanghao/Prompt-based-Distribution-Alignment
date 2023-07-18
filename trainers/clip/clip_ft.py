from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, count_num_param
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip

from trainers.baseda import *
from utils.templates import CUSTOM_TEMPLATES
from utils.clip_part import *


class CustomCLIP(Base_CustomCLIP):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.text_encoder = Simple_TextEncoder(clip_model)
        self.image_encoder = clip_model.visual
        
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        
        prompt_prefix = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [prompt_prefix.format(c.replace("_", " ")) for c in classnames]
        self.tokenized_prompts = clip.tokenize(prompts)


@TRAINER_REGISTRY.register()
class CLIP_FT(BaseDA):
    '''
    CLIP full fine-tuning
    '''
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        self.domains = cfg.DOMAINS
        self.save = cfg.SAVE_MODEL
        
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.CLIP.PREC == "fp32" or cfg.TRAINER.CLIP.PREC == "amp":
            clip_model.float()  # CLIP's default precision is fp16

        print("Building custom CLIP...")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning on gradients in both the image and the text encoder...")
        for name, param in self.model.named_parameters():
            param.requires_grad_(True)
        print("# params: {:,}".format(count_num_param(self.model)))
        
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

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

        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("full_model", self.model, self.optim, self.sched)
        self.scaler = GradScaler() if cfg.TRAINER.CLIP.PREC == "amp" else None

    def forward_backward(self, batch_x, batch_u):
        image_x, label, image_u = self.parse_batch_train(batch_x, batch_u)
        prec = self.cfg.TRAINER.CLIP.PREC
        
        if prec == "amp":
            with autocast():    # 混合精度训练
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
            "acc_x": compute_accuracy(output_x, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary
