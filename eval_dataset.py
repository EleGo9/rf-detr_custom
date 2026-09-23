"""Valuta un checkpoint RF-DETR su uno split di un dataset in formato COCO.

Il dataset deve avere il layout Roboflow standard (cartelle train/valid/test, ognuna
con un _annotations.coco.json) — lo stesso prodotto da yolo2coco_format.py.

Esempio:
    python eval_dataset.py \
        --weights data/20260922_lacisa_coco_format/checkpoint_best_total.pth \
        --dataset-dir /media/elena/T9/HAura/mcap/lacisa_coco_format \
        --split val --per-class
"""

import argparse
import json

from rfdetr import RFDETR


def main():
    parser = argparse.ArgumentParser(description="Valuta RF-DETR su un dataset COCO")
    parser.add_argument("--weights", required=True, help="checkpoint .pth da valutare")
    parser.add_argument("--dataset-dir", required=True, help="cartella del dataset (contiene valid/ e/o test/)")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--per-class", action="store_true", help="stampa anche AP/AR per classe")
    args = parser.parse_args()

    model = RFDETR.from_checkpoint(args.weights)
    metrics = model.evaluate(
        split=args.split,
        dataset_dir=args.dataset_dir,
        batch_size=args.batch_size,
        log_per_class_metrics=args.per_class,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
