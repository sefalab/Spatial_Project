# DeepLabV3+

[DeepLabV3+](https://arxiv.org/pdf/1802.02611.pdf) implementation for Semantic Segmentation using TensorFlow 2

#### Train
* Run `python train.py` for final training

#### Test
* Run `python test.py`

#### Dataset structure (similar to CamVid dataset)
    ├── Dataset folder 
        ├── train
            ├── 1111.png
            ├── 2222.png
        ├── train_labels
            ├── 1111_L.png
            ├── 2222_L.png
        ├── class_dict.csv
 
#### Note 
* default feature extractor is [EfficientNetV2-S](https://arxiv.org/pdf/2104.00298.pdf) 
* you can change backbone network in `nets\nn.py`
* changing configuration of training, change parameters in `utils/config.py`