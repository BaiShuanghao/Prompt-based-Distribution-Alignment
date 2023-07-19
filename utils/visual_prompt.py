"""
@article{bahng2022exploring,
  title={Exploring visual prompts for adapting large-scale models},
  author={Bahng, Hyojin and Jahanian, Ali and Sankaranarayanan, Swami and Isola, Phillip},
  journal={arXiv preprint arXiv:2203.17274},
  year={2022}
}

Adapted from https://github.com/hjbahng/visual_prompting
"""

import torch
import torch.nn as nn
import numpy as np


class PadPrompter(nn.Module):
    def __init__(self, cfg):
        super(PadPrompter, self).__init__()
        self.gpu = cfg.GPU
        pad_size = cfg.TRAINER.VPT.NUM_TOKENS
        image_size = cfg.INPUT.SIZE[0]

        self.base_size = image_size - pad_size*2
        self.pad_up = nn.Parameter(torch.randn([1, 3, pad_size, image_size]))
        self.pad_down = nn.Parameter(torch.randn([1, 3, pad_size, image_size]))
        self.pad_left = nn.Parameter(torch.randn([1, 3, image_size - pad_size*2, pad_size]))
        self.pad_right = nn.Parameter(torch.randn([1, 3, image_size - pad_size*2, pad_size]))

    def forward(self, x):
        base = torch.zeros(1, 3, self.base_size, self.base_size).to(torch.device("cuda:{}".format(self.gpu)))
        prompt = torch.cat([self.pad_left, base, self.pad_right], dim=3)
        prompt = torch.cat([self.pad_up, prompt, self.pad_down], dim=2)
        prompt = torch.cat(x.size(0) * [prompt])

        return x + prompt


class FixedPatchPrompter(nn.Module):
    def __init__(self, cfg):
        super(FixedPatchPrompter, self).__init__()
        self.gpu = cfg.GPU
        self.isize = cfg.INPUT.SIZE[0]
        self.psize = cfg.TRAINER.VPT.NUM_TOKENS
        self.patch = nn.Parameter(torch.randn([1, 3, self.psize, self.psize]))

    def forward(self, x):
        prompt = torch.zeros([1, 3, self.isize, self.isize]).to(torch.device("cuda:{}".format(self.gpu)))
        prompt[:, :, :self.psize, :self.psize] = self.patch

        return x + prompt


class RandomPatchPrompter(nn.Module):
    def __init__(self, cfg):
        super(RandomPatchPrompter, self).__init__()
        self.gpu = cfg.GPU
        self.isize = cfg.INPUT.SIZE[0]
        self.psize = cfg.TRAINER.VPT.NUM_TOKENS
        self.patch = nn.Parameter(torch.randn([1, 3, self.psize, self.psize]))

    def forward(self, x):
        x_ = np.random.choice(self.isize - self.psize)
        y_ = np.random.choice(self.isize - self.psize)

        prompt = torch.zeros([1, 3, self.isize, self.isize]).to(torch.device("cuda:{}".format(self.gpu)))
        prompt[:, :, x_:x_ + self.psize, y_:y_ + self.psize] = self.patch

        return x + prompt


def padding(cfg):
    return PadPrompter(cfg)


def fixed_patch(cfg):
    return FixedPatchPrompter(cfg)


def random_patch(cfg):
    return RandomPatchPrompter(cfg)