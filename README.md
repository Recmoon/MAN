# MAN
Source code of our Neurocomputing'21 paper [Multi-level Alignment Network for Domain Adaptive Cross-modal Retrieval](https://www.researchgate.net/publication/349383195_Multi-level_Alignment_Network_for_Domain_Adaptive_Cross-modal_Retrieval).

## Requirements

#### Environments
* **Ubuntu** 16.04
* **CUDA** 9.0
* **Python** 2.7
* **PyTorch** 0.3.1

We used virtualenv to setup a deep learning workspace that supports PyTorch.
Run the following script to install the required packages.
```shell
virtualenv --system-site-packages -p python2.7 ~/ws_dual
source ~/ws_dual/bin/activate
git clone https://github.com/Recmoon/MAN.git
cd ~/MAN
pip install -r requirements.txt
deactivate
```

#### Required Data(todo)
Run `do_get_dataset.sh` or the following script to download and extract [MSR-VTT(1.9G)](http://lixirong.net/data/cvpr2019/msrvtt10k-text-and-resnet-152-img1k.tar.gz) dataset and a pre-trained [word2vec(3.0G)](http://lixirong.net/data/w2vv-tmm2018/word2vec.tar.gz).
The extracted data is placed in `$HOME/VisualSearch/`.
```shell
ROOTPATH=$HOME/VisualSearch
mkdir -p $ROOTPATH && cd $ROOTPATH

# download and extract dataset

# download and extract pre-trained word2vec
wget http://lixirong.net/data/w2vv-tmm2018/word2vec.tar.gz
tar zxf word2vec.tar.gz
```
The data can also be downloaded from [Google Drive](https://drive.google.com/drive/folders/1GoomucXoAmhd3Jhngdnea7t0GOnJoGth?usp=sharing) and [Baidu Pan]().
Note: Code of video feature extraction is available [here](https://github.com/xuchaoxi/video-cnn-feat).

## Getting started

#### Single-source training
Run the following script to train on VATEX as a source dataset and MSR-VTT as a target dataset and evaluate `MAN` network on MSR-VTT.
```shell
source ~/ws_dual/bin/activate
./do_all.sh 
deactive
```
Running the script will do the following things:
1. Generate a vocabulary on the training set.
2. Train `MAN` network and select a checkpoint that performs best on the validation set as the final model. Notice that we only save the best-performing checkpoint on the validation set to save disk space.
3. Evaluate the final model on the test set.

#### Multi-source training
Run the following script to train on TGIF and VATEX as source datasets and MSR-VTT as a target dataset and evaluate `MAN` network on MSR-VTT.
```shell
source ~/ws_dual/bin/activate
./do_all_multi.sh 
deactive
```
Running the script will do the following things:
1. Generate a vocabulary on the training set.
2. Train `MAN` network and select a checkpoint that performs best on the validation set as the final model. Notice that we only save the best-performing checkpoint on the validation set to save disk space.
3. Evaluate the final model on the test set.

## How to run MAN on another datasets?

Store the training, validation and test subset into three folders in the following structure respectively.
```shell
${subset_name}
├── FeatureData
│   └── ${feature_name}
│       ├── feature.bin
│       ├── shape.txt
│       └── id.txt
├── ImageSets
│   └── ${subset_name}.txt
└── TextData
    └── ${subset_name}.caption.txt

```

* `FeatureData`: video frame features. Using [txt2bin.py](https://github.com/danieljf24/simpleknn/blob/master/txt2bin.py) to convert video frame feature in the required binary format.
* `${subset_name}.txt`: all video IDs in the specific subset, one video ID per line.
* `${dsubset_name}.caption.txt`: caption data. The file structure is as follows, in which the video and sent in the same line are relevant.
```
video_id_1#1 sentence_1
video_id_1#2 sentence_2
...
video_id_n#1 sentence_k
...
```

You can run the following script to check whether the data is ready:
```shell
./do_format_check.sh ${train_set} ${val_set} ${test_set} ${rootpath} ${feature_name}
```
where `train_set`, `val_set` and `test_set` indicate the name of training, validation and test set, respectively, ${rootpath} denotes the path where datasets are saved and `feature_name` is the video frame feature name.


If you pass the format check, first set the sub-set name in `do_all.sh` for single-source training and `do_all_multi.sh` for multi-source training and use the following script to train and evaluate MAN on your own dataset:
```shell
source ~/ws_dual/bin/activate
./do_all.sh
deactive
```
or

```shell
source ~/ws_dual/bin/activate
./do_all_multi.sh
deactive
```

where `caption_num` denotes the number of captions for each video. For the MSRVTT dataset, the value of `caption_num` is 20. 

## References
If you find the package useful, please consider citing our Neurocomputing'21 paper:
```
@article{dong2021multi,
  title={Multi-level Alignment Network for Domain Adaptive Cross-modal Retrieval},
  author={Dong, Jianfeng and Long, Zhongzi and Mao, Xiaofeng and Lin, Changting and He, Yuan and Ji, Shouling},
  journal={Neurocomputing},
  volume={440},
  pages={207--219},
  year={2021},
  publisher={Elsevier}
}
```

