# Multimodal Self-Distillation for Enhanced Generalized Category Discovery


📄 Paper: This repository contains the official implementation of the paper "Multimodal Self-Distillation for Enhanced Generalized Category Discovery", submitted to The Visual Computer (Springer).
![image](./assets/progcd_pipeline.png)


Generalized Category Discovery (GCD) presents a challenging open-world learning problem by clustering novel samples while classifying known but unlabeled samples with limited prior knowledge. Existing methods primarily focus on single-modal approaches, overlooking the complementary advantages of textual information. Recent multimodal methods have begun exploring text generation from visual features, yet rely solely on visual encoders during inference, discarding textual cues precisely when they are most needed for distinguishing novel categories. This paper introduces ProGCD, a multimodal self-distillation framework that preserves both visual and textual modalities throughout inference to enhance category discovery. Our approach employs a lightweight Prompt Generation Network to synthesize instance-specific pseudo-prompts through conditional prompt learning with supervised alignment and self-supervised cross-modal consistency, then fuses visual and textual features within a self-distillation framework. To address fusion-induced prediction bias toward known classes, we incorporate Sinkhorn-Knopp regularization, ensuring balanced category discovery while preserving discriminative signals. Extensive experiments on standard datasets (CIFAR100, ImageNet-100) and fine-grained datasets (CUB, Stanford Cars, FGVC-Aircraft) demonstrate that ProGCD significantly outperforms state-of-the-art single-modal methods and achieves competitive performance among multimodal approaches, with improvements of up to 7.6\% on CIFAR100 and 19.4\% on CUB.

## Running
### Dependencies
```shell
pip install -r requirments.txt
```
### Config
Set paths to datasets and desired log directories in `config.py`
### Training
#### 1. Prompt Generation Stage
1.1. Prepare prompts:
```shell
cd prompt_generation
python get_img_and_generate.py
python txt_to_pkl.py
python get_pth.py
```
1.2. Pre-training PGN:
```shell
python prompt_generation.py
```
#### 2. Discovery Learning Stage
```pgthon
bash ./scripts/run_${DATASET_NAME}.sh
```

## Result
![image](./assets/radar.png)

Our results:
|    Dataset    |  All |  Old |  New |
|:-------------:|:----:|:----:|:----:|
| CIFAR100      | 87.7 | 86.2 | 90.7 |
| ImageNet100   | 91.9 | 92.0 | 91.7 |
| CUB           | 79.9 | 82.9 | 78.1 |
| Stanford Cars | 79.0 | 85.6 | 75.8 |
| FGVC-Aircraft | 59.6 | 58.2 | 60.4 |


## Acknowledgements
The codebase is heavily built on [Cocoop](https://github.com/KaiyangZhou/CoOp?tab=readme-ov-file) and [SimGCD](https://github.com/CVMI-Lab/SimGCD). Thanks for their contribution!


