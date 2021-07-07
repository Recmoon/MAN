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
virtualenv --system-site-packages -p python2.7 ~/ws_man
source ~/ws_man/bin/activate
git clone https://github.com/Recmoon/MAN.git
cd ~/MAN
pip install -r requirements.txt
deactivate
```

#### Required Data(todo)
Run the following script to download and extract [MSR-VTT(1.9G)]() dataset, [TGIF(1.9G)]() dataset, [VATEX(1.9G)]() dataset and a pre-trained [word2vec(3.0G)](http://lixirong.net/data/w2vv-tmm2018/word2vec.tar.gz). Note that the train, val, test set of MSR-VTT dataset share the same FeatureData, and TextData can be downloaded from here(). 
The extracted data is placed in `$HOME/VisualSearch/`.
```shell
ROOTPATH=$HOME/VisualSearch
mkdir -p $ROOTPATH && cd $ROOTPATH

# download and extract dataset
wget 
wget
wget
wget
unzip tgif.zip
unzip vatex.zip
unzip mstvtt10ktrain.zip
unzip TextData.zip 
cp -r msrvtt10ktrain msrvtt10ktest
cp -r msrvtt10ktrain msrvtt10kval
mv TextData/msrvtt10kval.caption.txt msrvtt10kval/TextData/
mv TextData/msrvtt10ktest.caption.txt msrvtt10ktest/TextData/

# download and extract pre-trained word2vec
wget http://lixirong.net/data/w2vv-tmm2018/word2vec.tar.gz
tar zxf word2vec.tar.gz
```
The data can also be downloaded from [Google Drive]() and [Baidu Pan]().
Note: Code of video feature extraction is available [here](https://github.com/xuchaoxi/video-cnn-feat).

## Getting started

#### Single-source training
Run the following script to train on VATEX as a source dataset and MSR-VTT as a target dataset and evaluate `MAN` network on MSR-VTT.
```shell
source ~/ws_man/bin/activate
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
source ~/ws_man/bin/activate
./do_all_multi.sh 
deactive
```
Running the script will do the following things:
1. Generate a vocabulary on the training set.
2. Train `MAN` network and select a checkpoint that performs best on the validation set as the final model. Notice that we only save the best-performing checkpoint on the validation set to save disk space.
3. Evaluate the final model on the test set.

## Expected Performance

The expected performance of single-source training on VATEX is as follows. Notice that due to random factors in SGD based training, the numbers differ slightly from those reported in the paper.

|  | R@1 | R@5 | R@10 | Med r | mAP |
| ------------- | ------------- | ------------- | ------------- |  ------------- | ------------- |
| Text-to-Video | 5.9  | 16.1 | 22.9 | 74 | 0.117 |
| Video-to-Text | 10.8 | 23.9 | 32.3 | 32 | 0.05 |

The expected performance of multi-source training on VATEX and TGIF is as follows. Notice that due to random factors in SGD based training, the numbers differ slightly from those reported in the paper.

|  | R@1 | R@5 | R@10 | Med r | mAP |
| ------------- | ------------- | ------------- | ------------- |  ------------- | ------------- |
| Text-to-Video | 9.7  | 21.4 | 30.8 | 45 | 0.152 |
| Video-to-Text | 14.3 | 31.7 | 41.6 | 16 | 0.08 |

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
source ~/ws_man/bin/activate
./do_all.sh
deactive
```
or

```shell
source ~/ws_man/bin/activate
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

