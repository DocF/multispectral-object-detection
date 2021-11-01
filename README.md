# Multispectral-Object-Detection

## Intro
Multispectral Object Detection with Transformer and Yolov5



## Installation 
Python>=3.6.0 is required with all requirements.txt installed including PyTorch>=1.7 (The same as yolov5 https://github.com/ultralytics/yolov5 ).

### Clone the repo
    git clone https://github.com/DocF/multispectral-object-detection
  
### Install requirements
 ```bash
$ cd  multispectral-object-detection
$ pip install -r requirements.txt
```

## Dataset
-[FLIR]  [download](http://shorturl.at/ahAY4) A new aligned version.

-[LLVIP]  [download](https://github.com/bupt-ai-cz/LLVIP)

-[VEDAI]  [download](https://downloads.greyc.fr/vedai/)


## Download the pretrained weights
yolov5 weights:

CFT weights:

## Change the data cfg
some example in data/multispectral/


## Results

|Dataset|CFT|mAP50|mAP75|mAP|
|:---------: |------------|:-----:|:-----------------:|:-------------:|
|FLIR||73.0|32.0|37.4|
|FLIR| ✔️ |**77.7 (Δ4.7)**|**34.8 (Δ2.8)**|**40.0 (Δ2.6)**|
|LLVIP||95.8|71.4|62.3|
|LLVIP| ✔️ |**97.5 (Δ1.7)**|**72.9 (Δ1.5)**|**63.6 (Δ1.3)**|
|VEDAI||79.7 | 47.7  | 46.8
|VEDAI| ✔️ |**85.3 (Δ5.6)**|**65.9(Δ18.2)**|**56.0 (Δ9.2)**|


  
