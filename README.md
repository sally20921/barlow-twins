#Barlow Twins

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
