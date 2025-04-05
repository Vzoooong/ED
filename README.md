# Emphasizing Differences
Official Emphasizing Differences repository

## Introduction

In this work, we propose a novel anomaly detection framework called Emphasize Differences (ED), which addresses the "learning shortcut" problem in reconstruction-based methods by simultaneously modeling feature discrepancies within individual samples and variations across different samples.  The ED framework begins by extracting multi-scale feature maps from input images using a CNN backbone pre-trained on ImageNet. These reference features, combined with the training data, are then processed by a Transformer network. The 2D input images are first transformed into a high-dimensional feature space via an Embedding2D module. Next, the features are passed through a deep network consisting of Difference-Aware Attention and D2Fusion blocks, which progressively learn both intra-sample relationships and inter-sample feature correlations.
