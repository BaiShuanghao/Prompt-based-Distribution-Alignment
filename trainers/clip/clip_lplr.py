from sklearn.linear_model import LogisticRegression

import torch
from torch.cuda.amp import GradScaler

from dassl.engine import TRAINER_REGISTRY
from dassl.optim import build_optimizer, build_lr_scheduler

from trainers.baseda import *
from utils.clip_part import *


class CustomCLIP(Base_CustomCLIP):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__(cfg, classnames, clip_model)
        self.image_encoder = clip_model.visual
        self.dtype = clip_model.dtype
        
    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        return image_features


@TRAINER_REGISTRY.register()
class CLIP_LR(BaseDA):
    """
    LP: Logistic Regression Classifier
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

        print("Building custom CLIP and Classifier...")
        self.model = CustomCLIP(cfg, classnames, clip_model)
        self.classifier = LogisticRegression(solver="lbfgs", penalty="l2", random_state=0, C=0.316, max_iter=1000, verbose=1)
        
        print("Turning off gradients in both the image and the text encoder...")
        for name, param in self.model.named_parameters():
            param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

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
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("CLIP_model", self.model, self.optim, self.sched)
        self.scaler = GradScaler() if cfg.TRAINER.CLIP.PREC == "amp" else None

    def train(self):
        self.before_train()
        self.after_train()
        
    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        train_data_loader = self.train_loader_x
        data_loader = self.test_loader
        print("Do evaluation on test set")

        train_features = []
        train_labels = []
        for batch_idx, batch in enumerate(train_data_loader):
            input, label = self.parse_batch_test(batch)
            features = self.model_inference(input)
            
            train_features.append(features)
            train_labels.append(label)
            
        train_features = torch.cat(train_features).cpu().numpy()
        train_labels = torch.cat(train_labels).cpu().numpy()
        
        self.classifier.fit(train_features, train_labels)

        for batch_idx, batch in enumerate(data_loader):
            input, label = self.parse_batch_test(batch)
            features = self.model_inference(input)
            features = features.cpu().numpy()
            predictions = self.classifier.predict(features)
            predictions = torch.from_numpy(predictions).cuda(label.device)

            '''
            ***** Here we modify the evaluator.process in dassl library *****
            def process(self, mo, gt, distri=True):
                # mo (torch.Tensor): model output [batch, num_classes]
                # gt (torch.LongTensor): ground truth [batch]
                if distri:
                    pred = mo.max(1)[1]
                else:
                    pred = mo
                matches = pred.eq(gt).float()
                self._correct += int(matches.sum().item())
                self._total += gt.shape[0]

                self._y_true.extend(gt.data.cpu().numpy().tolist())
                self._y_pred.extend(pred.data.cpu().numpy().tolist())

                if self._per_class_res is not None:
                    for i, label in enumerate(gt):
                        label = label.item()
                        matches_i = int(matches[i].item())
                        self._per_class_res[label].append(matches_i)
            '''
            
            self.evaluator.process(predictions, label, False)    

        results = self.evaluator.evaluate()
        for k, v in results.items():
            tag = "{}/{}".format(split, k)
            self.write_scalar(tag, v, self.epoch)

        results_all = results["accuracy"]

        return results_all
