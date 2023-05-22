# Semantic Segmentation for Indoor Scene Analysis

The code in this repository is mostly taken from 2021 state of the art ESA-Net.
Paper: [IEEE Xplore](https://ieeexplore.ieee.org/document/9561675),  [arXiv](https://arxiv.org/pdf/2011.06961.pdf).
Repository: [Github](https://github.com/TUI-NICR/ESANet.git)

This repository contains the code of 
Training on RGB only, Depth only and RGBD for both Real and Synthetic Datasets.
Validation of Real on Real, Real on Synthetic, Synthetic on Synthetic, Synthetic on Real.

## Setup

1. Clone repository:
    ```bash
    git clone git@github.com:hussainhadi673/Semantic-Segmentation.git
   
    cd /path/to/this/repository
    ```

2. Set up anaconda environment including all dependencies:
    ```bash
    # create conda environment from YAML file
    conda env create -f rgbd_segmentation.yaml
    # activate environment
    conda activate rgbd_segmentation
    ```

3. Data preparation (training / evaluation / dataset inference):  
    We trained our networks on 
    [NYUv2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html), 
    [SceneNet RGB-D](https://robotvault.bitbucket.io/scenenet-rgbd.html). 

    The folder [`src/datasets`](src/datasets) contains the code to prepare
    NYUv2, SunRGB-D, Cityscapes, SceneNet RGB-D for training and evaluation. 
    Please follow the instructions given for the respective dataset and store 
    the created datasets in `./datasets`.

4. Pretrained models (evaluation):  
   We provide the weights for our selected ESANet on NYUv2 and Scenenet RGBD:
   
   | Dataset                       | Modality         |Classes| mIoU  | URL  |
   |-------------------------------|------------------|-------|-------|------|
   | NYUv2 (test)                  | RGBD             |  40   | 47.92 | [Download](https://drive.google.com/file/d/1-ZpOQEbuqeEuBpoOCQ8StP2iUxPQ2wKo/view?usp=share_link) |
   |    	         	   | RGB Only	      |	 40   | 40.42 | [Download](https://drive.google.com/file/d/1-CQlRMhTbTgdIGEFQb8sKwcbmz2urnfi/view?usp=share_link) |
   | 		                   | Depth Only       |  40   | 37.21 | [Download](https://drive.google.com/file/d/1-QymKrTfohhK-jaWN_i80SKY3OkUZZfZ/view?usp=share_link) |
   | 		                   | RGBD             |  13   | 63.18 | [Download](https://drive.google.com/file/d/10DFvJmJ6B9Cq1UdOGAZYG-kEkbVWWxSh/view?usp=share_link) |                                       |
   | SceneNet-RGBD (test)          | RGBD	      |  13   | 46.7  | [Download](https://drive.google.com/file/d/1-vk5KxMv3lhguqxTZktGlV3L9PGWgg1d/view?usp=share_link) |


   Download and extract the models to `./trained_models`.

5. Cross Validation Results:

   | Dataset                       | Modality         |Classes| mIoU  | 
   |-------------------------------|------------------|-------|-------|
   | NYUv2 - ScenetNet RGBD        | RGBD             |  13   | 16.82 | 
   |    	         	   | RGBD	      |	 40   | 20.77 | 
   | NYUv2 - SUN-RGBD		   | RGBD             |  40   | 24.83 | 
   | SceneNet-RGBD - NYUv2         | RGBD             |  13   | 19.92 |  

## Content
There are subsection for different things to do:
- [Evaluation](#evaluation): Calculate Miou.
- [Sample Inference](#sample-inference): Infer model on Image.
- [Training](#training): Train new ESANet model.

## Evaluation
To reproduce the mIoUs reported in our paper, use `eval.py`.

> To Evaluate the Model, Provide the path of model along with number of classes and Path of dataset.  

Examples: 
- To evaluate NYUv2 trained on Nyuv2, run:
    ```bash
    python eval.py \
        --dataset nyuv2 \
        --dataset_dir ./datasets/nyuv2 \
        --ckpt_path ./path/to/Nyuv2_40class_trained_model \
        --modality rgbd
        --no_of_class 40
    ```
    Similarly change modality to depth and rgb and no_of_class to 13 to re-produce results.

- To evaluate Sun-RGBD trained on NYUv2, run:
    ```bash
    python eval.py \
        --dataset sunrgbd \
        --dataset_dir ./datasets/sunrgbd \
        --ckpt_path ./path/to/Nyuv2_40class_trained_model \
        --modality rgbd
        --no_of_class 40
    ```
- To evaluate Scene-Net trained on NYUv2, run:
    ```bash
    python eval.py \
        --dataset nyuv2 \
        --dataset_dir ./datasets/scenenetrgbd \
        --ckpt_path ./path/to/Nyuv2_13class_trained_model \
        --modality rgbd
        --no_of_class 13
    ```

- To evaluate Scenenetrgbd trained on Scenenetrgbd, run:
    ```bash
    python eval.py \
        --dataset scenenetrgbd \
        --dataset_dir ./datasets/scenenetrgbd \
        --ckpt_path ./path/to/scenetnet_13class_trained_model \
        --modality rgbd
        --no_of_class 13
    ```

- To evaluate Nyuv2 trained on Scenenetrgbd, run:
    ```bash
    python eval.py \
        --dataset nyuv2 \
        --dataset_dir ./datasets/nyuv2 \
        --ckpt_path ./path/to/scenetnet_13class_trained_model \
        --modality rgbd
        --no_of_class 13
    ```

### Sample Inference
Use `inference_samples.py` to apply a trained model to the samples given in 
`./samples`.


Examples: 
- To Make Inference on a Model, run:
    ```bash
    python inference_samples.py \
     --ckpt_path ./path/to/Model \
     --depth_scale 0.1 \
     --raw_depth \
     --modality rgbd \
     --no_of_class 13
    ```
Inference Results:
 Trained on Nyuv2 Infered on Nyuv2 (40 Class)
![img](samples/result_nyuv2-nyuv2(40).jpg)

 Trained on Nyuv2 Inferred on Nyuv2 (40 Class)
![img](samples/result_nyuv2-nyuv2(13).jpg)

### Training
Use `train.py` to train ESANet on NYUv2 orSceneNet RGB-D
(or implement your own dataset by following the implementation of the provided 
datasets). See below Examples

Examples: 
- Train ESANet on NYUv2 (except for the dataset arguments, also 
valid for SUNRGB-D):
    ```bash
    # either specify all arguments yourself
    python train.py \
        --dataset nyuv2 \
        --dataset_dir ./datasets/nyuv2 \
        --pretrained_dir ./trained_models/imagenet \
        --results_dir ./results \
        --height 480 \
        --width 640 \
        --batch_size 8 \
        --batch_size_valid 24 \
        --lr 0.01 \
        --optimizer SGD \
        --class_weighting median_frequency \
        --encoder resnet34 \
        --encoder_block NonBottleneck1D \
        --nr_decoder_blocks 3 \
        --modality rgbd \
        --encoder_decoder_fusion add \
        --context_module ppm \
        --decoder_channels_mode decreasing \
        --fuse_depth_in_rgb_encoder SE-add \
        --upsampling learned-3x3-zeropad
    
    # or use the default arguments
    python train.py \
        --dataset nyuv2 \
        --dataset_dir ./datasets/nyuv2 \
        --pretrained_dir ./trained_models/imagenet \
        --results_dir ./results
    ```

For further information, use `python train.py --help` or take a look at 
`src/args.py`.


