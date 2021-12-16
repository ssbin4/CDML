# Classification-based deep metric learning (CDML)

Final Project Code for 2021 Fall AIGS539 Computer Vision.

The code is based on https://github.com/neka-nat/pytorch-hdml and 
referenced some codes in https://github.com/wzzheng/HDML and https://github.com/clovaai/embedding-expansion.

### Prepare data

```
$ cd data
$ python cars196_downloader.py
$ python cars196_converter.py
```
For CUB200_2011, you should download manually from web, unzip the file, and convert it using 
cub200_2011_converter.py

### Training

When the number of candidates is 2, α=0.25, and β=1.0, type as follows.
```
$ python train.py -p -nc 2 -a 0.25 -b 1.0
```
You can set hyperparameters by changing nc, a, and b.

### Testing

When you test the model saved after 10000 iterations, type as follows.
```
$ python test.py -p -nc 2 -a 0.25 -b 1.0 -e 10000
```

### Loss Visualization

```
$ tensorboard --logdir=./runs
```
