# Decoupling and Recoupling Spatiotemporal Representation for RGB-D-based Motion Recognition, [arxiv](https://arxiv.org/***)

This is a PyTorch implementation of our paper. 
## 1. Requirements

torch>=1.7.0; torchvision>=0.8.0; Visdom(optional)

data prepare: Database with the following folder structure:

```
│NTURGBD/
├──dataset_splits/
│  ├── @CS
│  │   ├── train.txt
                video name               total frames    label
│  │   │    ├──S001C001P001R001A001_rgb      103          0 
│  │   │    ├──S001C001P001R001A004_rgb      99           3 
│  │   │    ├──...... 
│  │   ├── valid.txt
│  ├── @CV
│  │   ├── train.txt
│  │   ├── valid.txt
├──Images/
│  │   ├── S001C002P001R001A002_rgb
│  │   │   ├──000000.jpg
│  │   │   ├──000001.jpg
│  │   │   ├──......
├──nturgb+d_depth_masked/
│  │   ├── S001C002P001R001A002
│  │   │   ├──MDepth-00000000.png
│  │   │   ├──MDepth-00000001.png
│  │   │   ├──......
```
## 2. Methodology
<p align="center">
  <img width="400" height="200" src="demo/pipline.jpg">
  <img width="800" height="200" src="demo/decouple_recouple.jpg">
</p>
 We propose to decouple and recouple spatiotemporal representation for RGB-D-based motion recognition. Figure left illustrates the proposed multi-modal spatiotemporal representation learning framework. The RGB-D-based motion recognition can be described as spatiotemporal information decoupling modeling, compact representation recoupling learning, and cross-modal representation interactive learning. 
 Figure right shows the process of decoupling and recoupling saptiotemporal representation of a unimodal data.

## 3. Train and Evaluate
All of our models are pre-trained on the [20BN Jester V1 dataset](https://www.kaggle.com/toxicmender/20bn-jester) and the pretrained model can be download [here](https://drive.google.com/drive/folders/1eBXED3uXlzBZzix7TvtDlJrZ3SlDCSF6?usp=sharing). Before cross-modal representation interactive learning, we first separately perform unimodal representation learning on RGB and depth data modalities. 
### Unimodal Training
Take training an RGB model with 8 GPUs on the NTU-RGBD dataset as an example,
some basic configuration:
```bash
common:
  dataset: NTU 
  batch_size: 6
  test_batch_size: 6
  num_workers: 6
  learning_rate: 0.01
  learning_rate_min: 0.00001
  momentum: 0.9
  weight_decay: 0.0003
  init_epochs: 0
  epochs: 100
  optim: SGD
  scheduler:
    name: cosin                     # Represent decayed learning rate with the cosine schedule
    warm_up_epochs: 3 
  loss:
    name: CE                        # cross entropy loss function
    labelsmooth: True
  MultiLoss: True                   # Enable multi-loss training strategy.
  loss_lamdb: [ 1, 0.5, 0.5, 0.5 ]  # The loss weight coefficient assigned for each sub-branch.
  distill: 1.                       # The loss weight coefficient assigned for distillation task.

model:
  Network: I3DWTrans                # I3DWTrans represent unimodal training, set FusionNet for multi-modal fusion training.
  sample_duration: 64               # Sampled frames in a video.
  sample_size: 224                  # The image is croped into 224x224.
  grad_clip: 5.
  SYNC_BN: 1                        # Utilize SyncBatchNorm.
  w: 10                             # Sliding window size.
  temper: 0.5                       # Distillation temperature setting.
  recoupling: True                  # Enable recoupling strategy during training.
  knn_attention: 0.7                # Hyperparameter used in k-NN attention: selecting Top-70% tokens.
  sharpness: True                   # Enable sharpness for each sub-branch's output.
  temp: [ 0.04, 0.07 ]              # Temperature parameter follows a cosine schedule from 0.04 to 0.07 during the training.
  frp: True                         # Enable FRP module.
  SEHeads: 1                        # Number of heads used in RCM module.
  N: 6                              # Number of Transformer blochs configured for each sub-branch.

dataset:
  type: M                           # M: RGB modality, K: Depth modality.
  flip: 0.5                         # Horizontal flip.
  rotated: 0.5                      # Horizontal rotation
  angle: (-10, 10)                  # Rotation angle
  Blur: False                       # Enable random blur operation for each video frame.
  resize: (320, 240)                # The input is spatially resized to 320x240 for NTU dataset.
  crop_size: 224                
  low_frames: 16                    # Number of frames sampled for small Transformer.       
  media_frames: 32                  # Number of frames sampled for medium Transformer.  
  high_frames: 48                   # Number of frames sampled for large Transformer.
```

```bash
bash run.sh tools/train.py config/NTU.yml 0,1,2,3,4,5,6,7 8
```
or
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 train.py --config config/NTU.yml --nprocs 8  
```

### Cross-modal Representation Interactive Learning
Take training a fusion model with 8 GPUs on the NTU-RGBD dataset as an example.
```bash
bash run.sh tools/fusion.py config/NTU.yml 0,1,2,3,4,5,6,7 8
```
or
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 fusion.py --config config/NTU.yml --nprocs 8  
```

### Evaluation
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 train.py --config config/NTU.yml --nprocs 1 --eval_only --resume /path/to/model_best.pth.tar 
```

## 4. Models Download
| Dataset    |  Modality | Acc| Download | 
| :---     |   :---:    |  :---: |  :---:  |  :---:  |
| NvGesture  |    RGB     |  89.58  | [here](https://drive.google.com/drive/folders/1jLAoMFkJ8-z3lyF0-mTvCuY8QchhiNOA?usp=sharing) |
| NvGesture  |    Depth     |  90.62  | [here](https://drive.google.com/drive/folders/1jLAoMFkJ8-z3lyF0-mTvCuY8QchhiNOA?usp=sharing) |
| NvGesture  |    RGB-D     |  91.70  | [here](https://drive.google.com/drive/folders/1jLAoMFkJ8-z3lyF0-mTvCuY8QchhiNOA?usp=sharing) |
| THU-READ(CS2)  |    RGB     |  81.67  | [here](https://drive.google.com/drive/folders/1_oihwEN-AhhTvkmoTb5If6d1Hdf9Z5JC?usp=sharing) |
| THU-READ(CS2)  |    Depth     |  81.03  | [here](https://drive.google.com/drive/folders/1_oihwEN-AhhTvkmoTb5If6d1Hdf9Z5JC?usp=sharing) |
| THU-READ(CS2)  |    RGB-D     |  90.00  | [here](https://drive.google.com/drive/folders/1_oihwEN-AhhTvkmoTb5If6d1Hdf9Z5JC?usp=sharing) |
| NTU-RGBD(CS)  |    RGB     |  90.3  | [here](https://drive.google.com/drive/folders/1iMFsZA7X-8rIkafTIZT5Z0aEFvhT4pq6?usp=sharing) |
| NTU-RGBD(CS)  |    Depth     |  92.7  | [here](https://drive.google.com/drive/folders/1iMFsZA7X-8rIkafTIZT5Z0aEFvhT4pq6?usp=sharing) |
| NTU-RGBD(CS)  |    RGB-D     |  94.2  | [here](https://drive.google.com/drive/folders/1iMFsZA7X-8rIkafTIZT5Z0aEFvhT4pq6?usp=sharing) |
| NTU-RGBD(CV)  |    RGB     |  95.4  | [here](https://drive.google.com/drive/folders/1iMFsZA7X-8rIkafTIZT5Z0aEFvhT4pq6?usp=sharing) |
| NTU-RGBD(CV)  |    Depth     |  96.2  | [here](https://drive.google.com/drive/folders/1iMFsZA7X-8rIkafTIZT5Z0aEFvhT4pq6?usp=sharing) |
| NTU-RGBD(CV)  |    RGB-D     |  97.3  | [here](https://drive.google.com/drive/folders/1iMFsZA7X-8rIkafTIZT5Z0aEFvhT4pq6?usp=sharing) |
| IsoGD  |    RGB     |  60.87  | [here](https://drive.google.com/drive/folders/1oBSzhslRy34jqA-VsPFIYYCJfN8GdqED?usp=sharing) |
| IsoGD  |    Depth     |  60.17  | [here](https://drive.google.com/drive/folders/1oBSzhslRy34jqA-VsPFIYYCJfN8GdqED?usp=sharing) |
| IsoGD  |    RGB-D     |  66.78  | [here](https://drive.google.com/drive/folders/1oBSzhslRy34jqA-VsPFIYYCJfN8GdqED?usp=sharing) |

# Citation
```
@inproceedings{zhou2021DRSR,
      title={Decoupling and Recoupling Spatiotemporal Representation for RGB-D-based Motion Recognition}, 
      author={Benjia Zhou and Pichao Wang and Jun Wan and Yanyan Liang and Fan Wang and Du Zhang and Zhen Lei and Hao Li and Rong Jin},
      year={2021},
}
```
# LICENSE
The code is released under the MIT license.
# Copyright
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
