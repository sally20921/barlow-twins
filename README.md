
  
# Barlow Twins: Self-Supervised Learning via Redundancy Reduction

## Introduction
- a successful approach to SSL is to learn embeddings which are invariant to distortions of the input sample
- a recurring issue with this approach is the existence of trivial constant solutions (trivial constant embeddings) 
- propose an objective function that naturally avoids collapse by measuring the cross-correlation matrix between the outpus of two identical networks fed with distorted versions of a sample, and making it as close to the identity matrix as possible
- being robust to the training batch size
- does not require any asymmetric mechanisms like prediction networks, momentum encoders, non-differentiable operators, or stop-graidents
- strongly benefit from the use of very high-dimensional embeddings


![image](https://user-images.githubusercontent.com/38284936/128422997-d8d29703-7c4c-4755-a477-97e012e1e0fc.png)


```
pip install -r requirements.txt
```

## Training 
```
python train.py --gpus 8 --batch_size 256
```

## Linear Evaluation
```
python evaluate.py --gpus=8 --batch_size 256
```
```
python evaluate.py --gpus=8 --batch_size 256 --model_path /path/to/lightning_logs/model.ckpt
```

## Implementation Details
### Image augmentations
* Each input image is transformed twice to produce the two distorted views.
* random cropping, resizing to 224x224 (always applied)
* horizontal flipping, color jittering, converting to grayscale, Gaussian blurring and solarization (applied randomly, with some probability)
* same augmentation parameters as BYOL

### Architecture
* encoder: ResNet-50 (without the final classification layer, 2048 output units) followed by a projector network
* projector network: three linear layers, each with 8192 output units
* first two layers of the projector are followed by a batch normalization layer and rectified linear units
* encoder $\rightarrow$ representation $\rightarrow$ projector $\rightarrow$ embeddings

### Optimization
* follow the optimization protocol described in BYOL
* LARS optimizer
* train for 100 epochs with a batch size of 2048
* learning rate 0.2
* bias 0.0048
* learning rate warm-up period of 10 epochs 
* cosine decay schedule
* $\lambda = 5 \cdot 10^{-3}$
* distributed across 32 V100 GPUs and takes about 124 hours 
* BYOL with batch size 4096 takes 113 hours