# hair-seg

The architecture was proposed by [Alex L. Cheng C, etc. 'Real-time deep hair matting on mobile devices'](https://arxiv.org/pdf/1712.07168.pdf). This repository is based on https://github.com/aobo-y/hair-dye.

Tested with pytorch 1.0 
```
install it with conda install pytorch==1.0.0 torchvision==0.2.1 cuda100 -c pytorch)
```

## Download dataset

Download dataset from https://github.com/switchablenorms/CelebAMask-HQ and split it in train/val/test folds using https://github.com/switchablenorms/CelebAMask-HQ/tree/master/face_parsing#preprocessing. You should get 23608 in train and 2931 in validation.


**Data structure training**
```
├── data/dataset_celeba
│   ├── train
│   │   ├──images
│   │   │   ├── 1.jpg
│   │   │   ├── 2.jpg
│   │   │   ├── 3.jpg
...
│   │   ├──masks
│   │   │   ├── 1.png
│   │   │   ├── 2.png
│   │   │   ├── 3.png
...
│   ├── val
...
```

## Train

```
$ CUDA_VISIBLE_DEVICES=0 python -u main.py --mode=train --model_name default --print_freq 5 --optimizer adam --lr 3e-4 --wup 2000 --ep 16 --lr_schedule cosine
```

The checkpoint and sample images are saved in `checkpoint/default/` by default.
Trained model gives around 0.89 IoU.

## Run

Plot a groundtruth image, the predicted segmentation and the hue adjusted result from the datasets or any specified image

```
$ python main.py --mode=run --set=test --num=4 --checkpoint train_16 --model_name default
$ python main.py --mode=run --checkpoint train_16 --model_name default --image=./path/to/the/image.png
```

`set` can be one `train` and `test`, default is `test`

`num` is the random number of images from the set, default is `4`

