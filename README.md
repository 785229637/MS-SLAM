<!-- PROJECT LOGO -->

<p align="center">

  <h1 align="center">MS-SLAM: Memory-Augmented 3D Gaussian Splatting SLAM for Efficient Robotic Applications</h1>

</p>



## Installation

##### (Recommended)
MS-SLAM has been benchmarked with Python 3.10, Torch 1.12.1 & CUDA=11.6. However, Torch 1.12 is not a hard requirement and the code has also been tested with other versions of Torch and CUDA such as Torch 2.3.0 & CUDA 12.1.

The simplest way to install all dependences is to use [anaconda](https://www.anaconda.com/) and [pip](https://pypi.org/project/pip/) in the following steps: 

```bash
conda create -n MS-SLAM python=3.10
conda activate MS-SLAM
conda install -c "nvidia/label/cuda-11.6.0" cuda-toolkit
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
pip install -r requirements.txt
```


### Replica

To run MS-SLAM on the `room0` scene, run the following command:

```bash
python scripts/MS-SLAM.py configs/replica/replic.py
```

For other scenes, please look at the `configs` file.

Please take care to adjust the parameters to suit different scenes.


## Acknowledgement

We thank the authors of the following repositories for their open-source code:

- 3D Gaussians
  - [Dynamic 3D Gaussians](https://github.com/JonathonLuiten/Dynamic3DGaussians)
  - [3D Gaussian Splating](https://github.com/graphdeco-inria/gaussian-splatting)
- Dataloaders
  - [GradSLAM & ConceptFusion](https://github.com/gradslam/gradslam/tree/conceptfusion)
- Baselines
  - [Nice-SLAM](https://github.com/cvg/nice-slam)
  - [Point-SLAM](https://github.com/eriksandstroem/Point-SLAM)
  - [SplaTAM](https://github.com/spla-tam/SplaTAM/blob/main/README.md)

## Citation

If you find our paper and code useful, please cite us:

```
Ben Wang, Yueri Cai and Jianqiao Xu. "MS-SLAM: Memory-Augmented 3D Gaussian Splatting SLAM for Efficient Robotic Applications." *GitHub*, Version 1.4.0, 2025, .
```


# MS-SLAM
