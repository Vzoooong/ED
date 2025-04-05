# Emphasizing Differences
Official Emphasizing Differences repository

## Introduction

<p align="center">
    <img src="1.png" width= "600">
</p>

we propose a novel anomaly detection framework called Emphasize Differences (ED), which performs feature reconstruction by jointly learning intra-image and inter-image differences. Inspired by human visual discrepancy detection, the ED framework explicitly amplifies these two types of differential signals during feature reconstruction. It achieves this through dual-path encoding: intra-image differences are captured via self-supervised constraints to identify local anomalies that disrupt structural consistency, while inter-image differences are modeled by a contrastive memory module that aligns reconstructed features with normal prototypes, thereby isolating anomalous responses in latent space.

<p align="center">
    <img src="2.png" width= "600">
</p>

In this work, we propose a novel anomaly detection framework called Emphasize Differences (ED), which addresses the "learning shortcut" problem in reconstruction-based methods by simultaneously modeling feature discrepancies within individual samples and variations across different samples.  The ED framework begins by extracting multi-scale feature maps from input images using a CNN backbone pre-trained on ImageNet. These reference features, combined with the training data, are then processed by a Transformer network. The 2D input images are first transformed into a high-dimensional feature space via an Embedding2D module. Next, the features are passed through a deep network consisting of Difference-Aware Attention and D2Fusion blocks, which progressively learn both intra-sample relationships and inter-sample feature correlations.
