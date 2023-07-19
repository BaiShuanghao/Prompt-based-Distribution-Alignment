# Attention-based-Prompt-Tuning for Unsupervised Domain Adaptation



> [**Attention-based-Prompt-Tuning for Unsupervised Domain Adaptation**](arXiv网址)<br>
> [作者名](作者网页),


Official implementation of the paper "[Attention-based-Prompt-Tuning for Unsupervised Domain Adaptation](arXiv网站)".
<hr />

## Highlights

![main figure](model.png)
> **<p align="justify"> Abstract:** *摘要* </p>

## Main Contributions

1) ****


## Supported Methods

[comment]: <> (| Language Prompting            | MaPLe |  [link]&#40;configs/trainers/IVLP/vit_b16_c2_ep5_batch4_4ctx_language_only.yaml&#41;      |      |)

| Method                    | Paper                                         |                             Code                            |  
|---------------------------|:----------------------------------------------|:---------------------------------------------------------------:|
 
| CoOp                      | [IJCV 2022](https://arxiv.org/abs/2109.01134) |                  [link](configs/trainers/CoOp)                  |
| CoCoOp                    | [CVPR 2022](https://arxiv.org/abs/2203.05557) |                 [link](configs/trainers/CoCoOp)                 |
| IVLP                      | [CVPR 2023](https://arxiv.org/abs/2210.03117) | [link](configs/trainers/MaPLe/vit_b16_c2_ep5_batch4_2ctx.yaml)  |
| MaPLe                     | [CVPR 2023](https://arxiv.org/abs/2210.03117) | [link](configs/trainers/MaPLe/vit_b16_c2_ep5_batch4_2ctx.yaml)  |

<hr />

## Results
### APT in comparison with existing methods
Results reported below show accuracy for base and novel classes for across 11 recognition datasets averaged over 3 seeds.

| Name                                                      | Base Acc. | Novel Acc. |    HM     | Epochs | 
|-----------------------------------------------------------|:---------:|:----------:|:---------:|:------:|
| [CLIP](https://arxiv.org/abs/2103.00020)                  |   69.34   |   74.22    |   71.70   |   -    |  
| [CoOp](https://arxiv.org/abs/2109.01134)                  | **82.69** |   63.22    |   71.66   |  200   | 
| [CoCoOp](https://arxiv.org/abs/2203.05557) |   80.47   |   71.69    |   75.83   |   10   | 
| [MaPLe](https://arxiv.org/abs/2210.03117)  |   82.28   | **75.14**  | **78.55** |   5    |  
| [VPT](网址)  |   82.28   | **75.14**  | **78.55** |   5    |  

## Installation 
For installation and other package requirements, please follow the instructions as follows. 
This codebase is tested on Ubuntu 20.04.2 LTS with python 3.8. Follow the below steps to create environment and install dependencies.

* Setup conda environment.
```bash
# Create a conda environment
conda create -y -n apt python=3.7

# Activate the environment
conda activate apt

# Install torch (requires version >= 1.8.1) and torchvision
# Please refer to https://pytorch.org/get-started/previous-versions/ if your cuda version is different
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
```

* Install dassl library.
```bash
# Instructions borrowed from https://github.com/KaiyangZhou/Dassl.pytorch#installation

# Clone this repo
git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
cd Dassl.pytorch/

# Install dependencies
pip install -r requirements.txt

# Install this library (no need to re-build if the source code is modified)
python setup.py develop
cd ..
```

* Clone MaPLe code repository and install requirements
```bash
# Clone MaPLe code base
git clone https://github.com/muzairkhattak/multimodal-prompt-learning.git

cd multimodal-prompt-learning/
# Install requirements

pip install -r requirements.txt

## Data preparation
Please follow the instructions at [DATASETS.md](docs/DATASETS.md) to prepare all datasets.
Datasets list:
- [Office-Home](网站)
- [Office-31](网站)
- [VisDA-2017](网站)

## Model Zoo

### Vision-Language prompting methods
| Name  (configs)                                                                                | Base Acc. | Novel Acc. |    HM     | Epochs |                                         Model / Logs                                         |
|------------------------------------------------------------------------------------------------|:---------:|:----------:|:---------:|:------:|:--------------------------------------------------------------------------------------------:|
| [Deep Vision Prompting](configs/trainers/VPT/vit_b16_c2_ep5_batch4_4.yaml)                     |   80.24   |   73.43    |   76.68   |   5    |  [link](https://drive.google.com/drive/folders/1zJnaod8UVvo1HuxNzymLhBBS_OHq6cYp?usp=sharing)                                                                                      | 
| [Deep Language Prompting](configs/trainers/IVLP/vit_b16_c2_ep5_batch4_4ctx_language_only.yaml) |   81.72   |   73.81    |   77.56   |   5    | [link](https://drive.google.com/drive/folders/1PPLtvQIGprRUyxPiTwOSEh_oQ46zQfCN?usp=sharing) |
| [Independent V-L Prompting](configs/trainers/IVLP/vit_b16_c2_ep5_batch4_2+2ctx.yaml)           |   82.15   |   74.07    |   77.90   |   5    | [link](https://drive.google.com/drive/folders/14NxzrRirK2GfyfWajsEGDiWa2suJoTBW?usp=sharing) |
| [MaPLe](configs/trainers/MaPLe/vit_b16_c2_ep5_batch4_2ctx.yaml)                                | **82.28** | **75.14**  | **78.55** |   5    | [link](https://drive.google.com/drive/folders/1EvuvgR8566bL0T7ucvAL3LFVwuUPMRas?usp=sharing) |


## Training and Evaluation
Please refer to the [RUN.md](docs/RUN.md) for detailed instructions on training, evaluating and reproducing the results using our pre-trained models.


<hr />

## Citation
If you use our work, please consider citing:
```bibtex
bibtex
```


## Acknowledgements

Our code is based on [CoOp and CoCoOp](https://github.com/KaiyangZhou/CoOp), [DAPL](https://github.com/LeapLabTHU/DAPrompt/tree/main) and [MaPLe](https://github.com/muzairkhattak/multimodal-prompt-learning) repository. We thank the authors for releasing their code. If you use our model and code, please consider citing these works as well.

