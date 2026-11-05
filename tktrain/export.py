"""
RF-DETR export launcher for tkTrain.

Usage: python3 tktrain/export.py <export_config.json>

The JSON is written by tkTrain (trainpipe/cvat/nets/RFDETR.py) next to the weights:
    {
        "weights": "trainings/<exp>/checkpoint_best_total.pth",
        "export_args": {...}                # RFDETR.export() arguments
    }
The model class and the number of classes are read from the checkpoint.
"""

import argparse
import json

from rfdetr import RFDETR


def main():
    parser = argparse.ArgumentParser(description="RF-DETR export launcher")
    parser.add_argument("config", type=str, help="Path to the export config JSON")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    export_args = cfg.get("export_args", {})
    if export_args.get("shape") is not None:
        export_args["shape"] = tuple(export_args["shape"])

    model = RFDETR.from_checkpoint(cfg["weights"])
    path = model.export(**export_args)
    print(f"Exported model: {path}")


if __name__ == "__main__":
    main()
