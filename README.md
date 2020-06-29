# Attention Mechanism Exploits Temporal Contexts: Real-time 3D Human Pose Reconstruction (CVPR 2020 Oral)
More  extensive  evaluation  andcode can be found at our lab website: (https://sites.google.com/a/udayton.edu/jshen1/cvpr2020)
![network](Figures/structure.jpg)
<p align="left">
  <img width="512" height="512" src=Figures/GIF.gif>
  &nbsp
  &nbsp
  &nbsp
  <img width="312" height="510" src=Figures/results.png>
</p>

PyTorch code of the paper "Attention Mechanism Exploits Temporal Contexts: Real-time 3D Human Pose Reconstruction". [pdf](http://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_Attention_Mechanism_Exploits_Temporal_Contexts_Real-Time_3D_Human_Pose_Reconstruction_CVPR_2020_paper.pdf)

### [Bibtex](https://scholar.googleusercontent.com/scholar.bib?q=info:sVZlnopW0ZQJ:scholar.google.com/&output=citation&scisdr=CgUvGH_mEIi98y29oOM:AAGBfm0AAAAAXu-4uOOunCSIKKuamAWN5VjFJ_OC0cHs&scisig=AAGBfm0AAAAAXu-4uBa5vr92Yk6AXlKVO0mVXEXZorOx&scisf=4&ct=citation&cd=-1&hl=en)

If you found this code useful, please cite the following paper:
    
    @inproceedings{liu2020attention,
      title={Attention Mechanism Exploits Temporal Contexts: Real-Time 3D Human Pose Reconstruction},
      author={Liu, Ruixu and Shen, Ju and Wang, He and Chen, Chen and Cheung, Sen-ching and Asari, Vijayan},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
      pages={5064--5073},
      year={2020}
    }
    
### Environment

The code is developed and tested on the following environment

* Python 3.6
* PyTorch 1.1 or higher
* CUDA 10

### Dataset

The source code is for training/evaluating on the [Human3.6M](http://vision.imar.ro/human3.6m) dataset. Our code is compatible with the dataset setup introduced by [Martinez et al.](https://github.com/una-dinosauria/3d-pose-baseline) and [Pavllo et al.](https://github.com/facebookresearch/VideoPose3D). Please refer to [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) to set up the Human3.6M dataset (`./data` directory).

### Training new models

To train a model from scratch, run:

```bash
python run.py -da -tta
```

`-da` controls the data augments during training and `-tta` is the testing data augmentation. 

For example, to train our 243-frame ground truth model or causal model in our paper, please run:

```bash
python run.py -k gt
```

or

```bash
python run.py -k cpn_ft_h36m_dbb --causal
```

It should require 24 hours to train on one TITAN RTX GPU.

### Evaluating pre-trained models

We provide the pre-trained cpn model [here](https://drive.google.com/file/d/1jiZWqAOJmXoTL8dxhPX8QgK0QeECeoAM/view?usp=sharing) and ground truth model [here](https://drive.google.com/file/d/1EAS9PUddznBPqNaEHV6-tCfqsQOHZ1Of/view?usp=sharing). To evaluate them, put them into the `./checkpoint` directory and run:

For cpn model:
```bash
python run.py -tta --evaluate cpn.bin
```

For ground truth model:
```bash
python run.py -k gt -tta --evaluate gt.bin
```

### Visualization and other functions

We keep our code consistent with [VideoPose3D](https://github.com/facebookresearch/VideoPose3D). Please refer to their project page for further information. 


