
  
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
