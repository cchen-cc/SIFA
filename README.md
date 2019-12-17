## Synergistic Image and Feature Adaptation:<br/> Towards Cross-Modality Domain Adaptation for Medical Image Segmentation

Tensorflow implementation of our unsupervised cross-modality domain adaptation framework. <br/>
Please refer to the branch SIFA-v1 for the exact version of our AAAI paper. <br/>
This is a updated version with more effective and stable training. 

## Paper
[Synergistic Image and Feature Adaptation: Towards Cross-Modality Domain Adaptation for Medical Image Segmentation](https://arxiv.org/abs/1901.08211)
<br/>
AAAI Conference on Artificial Intelligence, 2019 (oral)
<br/>
<br/>
![](figure/framework.png)

## Installation
* Install TensorFlow 1.0 and CUDA 9.0
* Clone this repo
```
git clone https://github.com/cchen-cc/SIFA
cd SIFA
```

## Data Preparation
* Raw data needs to be written into `tfrecord` format to be decoded by `./data_loader.py`.
* Put `tfrecord` data of two domains into corresponding folders under `./data` accordingly.
* Run `./create_datalist.py` to generate the datalists containing the path of each data.

## Train
* Modify paramter values in `./config_param.json`
* Run `./main.py` to start the training process

## Citation
If you find the code useful for your research, please cite our paper.
```
@inproceedings{chen2019synergistic,
  author    = {Chen, Cheng and Dou, Qi and Chen, Hao and Qin, Jing and Heng, Pheng-Ann},
  title     = {Synergistic Image and Feature Adaptation: Towards Cross-Modality Domain Adaptation for Medical Image Segmentation},
  booktitle = {Proceedings of The Thirty-Third Conference on Artificial Intelligence (AAAI)},
  pages     = {865--872},
  year      = {2019},
}
```

## Acknowledgement
Part of the code is revised from the [Tensorflow implementation of CycleGAN](https://github.com/leehomyc/cyclegan-1).

## Note
* The repository is being updated
* Contact: Cheng Chen (chencheng236@gmail.com)
