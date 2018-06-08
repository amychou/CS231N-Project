# CS231N: Adversarial attack on YOLO v2
Addapted DAG attacks:
```
python ./attack.py model_data/yolo.h5 -lr 5e-3 -iter 101 --clip -eps 0.01
```

Adapted T-FGSM attacks (unsigned):
```
python ./attack.py model_data/yolo.h5 -lr 1e-3 -iter 101 --clip -eps 0.01 --bounding
```

Code adapted from https://github.com/allanzelener/YAD2K
