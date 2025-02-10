# ProGCD: Boosting Generalized Category Discovery with Conditional Prompt Learning

This is the PyTorch implementation of ProGCD.

Generalized Category Discovery (GCD) addresses a challenging open-world learning problem by leveraging limited prior knowledge in an unsupervised environment. The key objective is to effectively cluster novel samples while simultaneously classifying known but unlabeled samples. Existing approaches primarily focus on enhancing representation learning through single-modal contrastive methods, overlooking naturally available textual information. In image recognition tasks, visual and textual modalities provide complementary advantages. Visual features capture essential perceptual details like shapes, textures, and spatial relationships, whereas textual information helps distinguish between visually similar but semantically distinct categories. However, acquiring descriptions for unlabeled data through image-text pairs remains challenging due to the absence of class name information. To address this limitation, we introduce conditional prompt learning and enhance the Contrastive Language-Image Pre-training (CLIP) model to generate appropriate pseudo prompts as textual information. Our approach introduces three key innovations through a novel multimodal fusion method for enhanced sample representation, a self-distillation mechanism for generating high-quality pseudo labels with fusion features, and a Sinkhorn-Knopp regularizer to mitigate model bias from modalities fusion. Extensive experiments on standard datasets (CIFAR100, ImageNet-100) and fine-grained datasets (CUB, Stanford Cars, FGVC-Airplane) demonstrate that our Prompt-based GCD (ProGCD) method significantly outperforms state-of-the-art single-modal GCD methods while achieving leading performance among multimodal approaches.

## Table of Contents

- [Introduction](#running)
- [Training](#training)

## Running
### Dependencies
```shell
pip install -r requirments.txt
```
## Training

### 1. Prompt Generation Stage

### 2. Discovery Learning Stage

## Result

## Acknowledgements

