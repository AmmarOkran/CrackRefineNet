<div align="center">
<h1>CrackRefineNet</h1>
<h3>CrackRefineNet: A Context- and Refinement-Driven Architecture for Robust Crack Segmentation under Real-World and Zero-Shot Conditions</h3>
</div>

This repository provides the official implementation of **CrackRefineNet**, a context- and refinement-driven convolutional architecture for **robust crack segmentation** under various conditions.  
CrackRefineNet is implemented using **[MMSegmentation v1.1.1](https://github.com/open-mmlab/mmsegmentation/tree/v1.1.1)** as the core framework.

## Network Architecture

<p align="center">
  <img src="./resources/arch.png" alt="CrackRefineNet Architecture" width="80%">
</p>

## Installation
We use **[MMSegmentation v1.1.1](https://github.com/open-mmlab/mmsegmentation/tree/v1.1.1)** as the codebase.

For install and data preparation, please find the guidelines in **[MMSegmentation v1.1.1](https://github.com/open-mmlab/mmsegmentation/tree/v1.1.1)** for the installation and data preparation.


## Datasets
### 1. RCFD dataset
### 2. CFD dataset
### 3. Crack500 dataset
The [Crack500](https://ieeexplore.ieee.org/document/8694955) dataset comprises 500 high-resolution pavement images (approximately 2000 × 1500 pixels) captured using mobile phones on the main campus of Temple University.
Each image was divided into 16 non-overlapping regions, and only those patches containing more than 1000 crack pixels were retained.
Following this preprocessing, the dataset includes 1,896 training images, 348 validation images, and 1,124 test images.
The dataset can be downloaded from [this](https://github.com/fyangneil/pavement-crack-detection) link.
```
|-- Crack500
    |-- train
        |-- images
        |   |-- 20160222_081011_1_361.jpg
            ......
        |-- masks
        |   |-- 20160222_081011_1_361.png
            ......
    |-- test
        |-- images
        |   |-- 20160222_080933_361_1.jpg
            ......
        |-- masks
        |   |-- 20160222_080933_361_1.png
            ......
    |-- val
        |-- images
        |   |-- 20160222_080850_1_361.jpg
            ......
        |-- masks
        |   |-- 20160222_080850_1_361.png
            ......
```
### 4. Sylvie dataset

## Pretrained Weights
| Dataset  | Checkpoint File               | Download Link                                                         |
| -------- | ----------------------------- | --------------------------------------------------------------------- |
| RCFD     | `crackrefinenet_rcfd.pth`     | [Google Drive](https://drive.google.com/file/d/1p2KSbGwrH6LdBhFh92eUEbPgshkP-CXt/view?usp=drive_link) |
| CFD      | `crackrefinenet_cfd.pth`      | [Google Drive](https://drive.google.com/file/d/1gjXX9tE6y2I8udP8SKZmjd419NXPx9IK/view?usp=drive_link) |
| Crack500 | `crackrefinenet_crack500.pth` | [Google Drive](https://drive.google.com/file/d/1hhGWZ-6jt9Drpjacs01ZAK9SqJyT3wFF/view?usp=drive_link) |
| Sylvie   | `Same file of CFD`            | [Google Drive](https://drive.google.com/file/d/1gjXX9tE6y2I8udP8SKZmjd419NXPx9IK/view?usp=drive_link) |


## Acknowledgements

This project is built upon **[MMSegmentation v1.1.1](https://github.com/open-mmlab/mmsegmentation/tree/v1.1.1)**. We thank the open-source community for enabling high-quality research and development in semantic segmentation.