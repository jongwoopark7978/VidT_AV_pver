# Pytorch Implementation of [Video Transformers for Autonomous Driving]

## project report
[Video Transformers for Autonomous Driving](https://github.com/jongwoopark7978/VidT_AV_pver/blob/main/Video%20Transformers%20for%20Autonomous%20Driving_Jongwoo%20Park%2C%20Sounak%20Mondal.pdf)

## Data Preparation
We leveraged the recently released large-scale Waymo Open Dataset. We used only the front images of 13 training tars (32.5GB) and 3 validation tars (7.5GB) to analyze the potential of our model.

## Training and Testing

```bash

#testing
python3 train.py --cuda 3 --batch_size 20 --epochs 2 --lr 0.00007 --gamma 0.7 --seed  42 --num_frames  10 --num_dims  20 --num_layers  2 --num_heads  2 --dim_head  10 --mlp_dim  10 --drop_prob  0.4 --emb_drop_prob  0.4 --cls_dim  10

#training
python3 train.py --cuda 3 --batch_size 64 --epochs 100 --lr 0.00007 --gamma 0.7 --seed  42 --num_frames  10 --num_dims  128 --num_layers  6 --num_heads  8 --dim_head  128 --mlp_dim  128 --drop_prob  0.4 --emb_drop_prob  0.4 --cls_dim  64

```

## Reference

```
[1] Mariusz  Bojarski,  Davide  D  Testa,  Daniel  Dworakowski,Bernhard Firner,  Beat Flepp,  Prasoon Goyal,  Lawrence D,Jackel,  Mathew  Monfort,  Urs  Muller,  Jiakai  Zhang,  et  al.End  to  end  learning  for  self-driving  cars.arXiv  preprintarXiv:1604.07316, 2016.

[2] Jacob Devlin,  Ming-Wei Chang,  Kenton Lee,  and KristinaToutanova.Bert:Pre-training   of   deep   bidirectionaltransformers  for  language  understanding.arXiv  preprintarXiv:1810.04805, 2018.

[3] Alexey  Dosovitskiy,  Lucas  Beyer,  Alexander  Kolesnikov,Dirk   Weissenborn,   Xiaohua   Zhai,   Thomas   Unterthiner,Mostafa Dehghani, Matthias Minderer, Georg Heigold, Syl-vain Gelly, et al.   An image is worth 16x16 words:  Trans-formers  for  image  recognition  at  scale.arXiv  preprintarXiv:2010.11929, 2020.

[4] Zhicheng  Gu,  Zhihao  Li,  Xuan  Di,  and  Rongye  Shi.   Anlstm-based autonomous driving model using a waymo opendataset.Applied Sciences, 10(6):2046, 2020.

[5] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.Deep residual learning for image recognition.   InProceed-ings of the IEEE conference on computer vision and patternrecognition, pages 770–778, 2016.

[6] Pengcheng  He,  Xiaodong  Liu,  Jianfeng  Gao,  and  WeizhuChen.  Deberta:  Decoding-enhanced bert with disentangledattention.arXiv preprint arXiv:2006.03654, 2020.

[7] Diederik P Kingma and Jimmy Ba.   Adam:  A method forstochastic  optimization.arXiv  preprint  arXiv:1412.6980,2014.

[8] Yang Liu and Mirella Lapata.  Text summarization with pre-trained encoders.arXiv preprint arXiv:1908.08345, 2019

[9] Pei  Sun,  Henrik  Kretzschmar,  Xerxes  Dotiwalla,  AurelienChouard, Vijaysai Patnaik, Paul Tsui, James Guo, Yin Zhou,Yuning Chai, Benjamin Caine, et al. Scalability in perceptionfor autonomous driving:  Waymo open dataset.  InProceed-ings of the IEEE/CVF Conference on Computer Vision andPattern Recognition, pages 2446–2454, 2020.

[10] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszko-reit,  Llion  Jones,  Aidan  N  Gomez,  Lukasz  Kaiser,  and  Il-lia  Polosukhin.   Attention  is  all  you  need.arXiv  preprintarXiv:1706.03762, 2017
```


