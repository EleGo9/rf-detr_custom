# RF-DETR: SOTA Real-Time Detection and Segmentation Model

[![version](https://badge.fury.io/py/rfdetr.svg)](https://badge.fury.io/py/rfdetr)
[![downloads](https://img.shields.io/pypi/dm/rfdetr)](https://pypistats.org/packages/rfdetr)
[![arXiv](https://img.shields.io/badge/arXiv-2511.09554-b31b1b.svg)](https://arxiv.org/abs/2511.09554)
[![python-version](https://img.shields.io/pypi/pyversions/rfdetr)](https://badge.fury.io/py/rfdetr)
[![license](https://img.shields.io/badge/license-Apache%202.0-blue)](https://github.com/roboflow/rfdetr/blob/main/LICENSE)

[![hf space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/SkalskiP/RF-DETR)
[![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-finetune-rf-detr-on-detection-dataset.ipynb)
[![roboflow](https://raw.githubusercontent.com/roboflow-ai/notebooks/main/assets/badges/roboflow-blogpost.svg)](https://blog.roboflow.com/rf-detr)
[![discord](https://img.shields.io/discord/1159501506232451173?logo=discord&label=discord&labelColor=fff&color=5865f2&link=https%3A%2F%2Fdiscord.gg%2FGbfgXGJ8Bk)](https://discord.gg/GbfgXGJ8Bk)

RF-DETR is a real-time, transformer-based object detection and instance segmentation model architecture developed by Roboflow and released under the Apache 2.0 license.

## Installation

To install RF-DETR, install the `rfdetr` package in a [**Python>=3.9**](https://www.python.org/) environment with `pip`:

```bash
pip install rfdetr
```

To install the entire environment:

```
conda create --file environment.yml
```
To install locally rfdetr:
```
export PYTHONPATH=/home/elenagovi/repos/rf-detr_custom:$PYTHONPATH
python -c "import rfdetr; print(rfdetr.__file__)"
```

## Onnx exportation

 ```
 python export_onnx_custom_v2.py   --weights path/to/checkpoint_best.pth   --num-classes 7   --output output/model.onnx
 ```

## Debug comparison between onnx and pth model

1) Uncomment lines 327-330 into /home/elena/repos/rf-detr_custom/rfdetr/detr.py
2) Write your custom input in debug.py (line 7, 10 and 29 into session, image and model)
3) ```python debug.py```


## Convert an existing dataset from yolo to coco format
1) Insert the correct paths
2) ```python yolo2coco_format.py```

## Test on a video 
1) Choose the weights you want (default coco weights)
2) ```python test_from_video.py <video.mp4> [output.mp4]```

## Test on a folder of images and compare two rfdetr models
1) Change the config in rfdetr/test.py: 
num_classes = 7
weights_path = "/path/to/weights.pth"
images_files = "path/to/images/*"
output = "path/to/predictions"
model1 = RFDETRNano() #COCO weights (default)
model2 = RFDETRNano(num_classes=num_classes, pretrain_weights= weights_path)

2) Based on the weights you want to use, change this import: ```from util.coco_classes import COCO_CLASSES```
You must have a file like COCO_CLASSES with your model's classes.

3) ```python test.py```

## Training

1) Change the config in train.py
dataset = "/media/elena/T7/BDD100K/coco"
epochs = 100
num_classes = 7
early_stopping = True
early_stopping_patience = 20
batch_size = 8
grad_accum_steps = 2 # tot BATCH SIZE = batch_size * grad_accum_steps
output_dir = "/path/to/output"

2) ```python train.py```
