# Pytorch Implementation of [Video Transformers for Autonomous Driving]

## project report
Please find the report in the current repository

## Data Preparation
We leverage the recently released large-scale Waymo Open Dataset. We used only the front images of 13 training tars (32.5GB) and 3 validation tars (7.5GB) to analyze the potential of our model.

## Training and Testing

```bash
python3 train.py --cuda 3 --batch_size 20 --epochs 2 --lr 0.00007 --gamma 0.7 --seed  42 --num_frames  10 --num_dims  20 --num_layers  2 --num_heads  2 --dim_head  10 --mlp_dim  10 --drop_prob  0.4 --emb_drop_prob  0.4 --cls_dim  10
```

## Reference

```
[1]
```


