import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, count_num_param
from dassl.optim import build_optimizer, build_lr_scheduler

from trainers.baseda import *
from utils.clip_part import *


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__(cfg, classnames, clip_model)
        self.image_encoder = clip_model.visual
        self.dtype = clip_model.dtype
        self.classifier = nn.Linear(clip_model.text_projection.shape[1], len(classnames))
        
    def forward(self, image):    
        image_features = self.image_encoder(image.type(self.dtype))
        logits = self.classifier(image_features)

        return logits


@TRAINER_REGISTRY.register()
class CLIP_FC(BaseDA):
    """
    Fc: Linear fc Classifier
    """
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
        self.model = CustomCLIP(classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder...")
        for name, param in self.model.named_parameters():
            if "classifier" not in name:
                param.requires_grad_(False)
                
        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        print("# params: {:,}".format(count_num_param(self.model.classifier)))

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.classifier, cfg.MODEL.INIT_WEIGHTS)

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
            raise ValueError

        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.classifier, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        '''
        register model could be updated. When new module needs to be updated
        register the module before use
        '''
        self.register_model("clip_fc", self.model.classifier, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.CLIP.PREC == "amp" else None

    def forward_backward(self, batch_x, batch_u):
        image_x, label, image_u = self.parse_batch_train(batch_x, batch_u)
        prec = self.cfg.TRAINER.CLIP.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image_x)
                loss = F.cross_entropy(output, label)
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
    