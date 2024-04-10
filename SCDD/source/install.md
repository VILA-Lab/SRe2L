# Installation Guide

### Prerequisites 
Please ensure you have the following environments
- Python 3.9
- CUDA 12.2
- PyTorch 2.0.1

### Installation Steps 
1. Set up a conda environments
```
$ conda create -n scdd python=3.9
$ conda activate scdd
```

2. Install PyTorch and TorchVision
```
$ pip install torch==2.0.1 torchvision==0.15.2
```

3. Additional dependence
```
$ pip install pillow 
$ pip install timm
```