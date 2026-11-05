"""
RF-DETR training launcher for tkTrain.

Usage: python3 tktrain/train.py <train_config.json>

The JSON is written by tkTrain (trainpipe/cvat/nets/RFDETR.py) in the experiment folder:
    {
        "model_class": "RFDETRNano",        # rfdetr model class
        "model_args": {...},                # ModelConfig fields (model constructor)
        "train_args": {...}                 # TrainConfig fields, dataset_dir and output_dir included
    }
The number of classes is detected from the dataset COCO categories.
Base weights are cached in $RF_HOME (set by tkTrain to data/weights/rfdetr).
"""

import argparse
import json

import rfdetr


def main():
    parser = argparse.ArgumentParser(description="RF-DETR training launcher")
    parser.add_argument("config", type=str, help="Path to the train config JSON")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    model_class = getattr(rfdetr, cfg.get("model_class", "RFDETRNano"))
    model = model_class(**cfg.get("model_args", {}))
    model.train(**cfg["train_args"])


# Required when the DataLoader workers use "spawn"/"forkserver": each worker re-imports this
# file as __main__, and model.train() must not run again inside them.
if __name__ == "__main__":
    main()
