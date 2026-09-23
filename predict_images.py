"""Esegue l'inferenza RF-DETR su una cartella di immagini e salva le copie annotate.

Esempio:
    python predict_images.py \
        --weights data/20260922_lacisa_coco_format/checkpoint_best_total.pth \
        --images /media/elena/T9/HAura/mcap/lacisa_coco_format/valid \
        --output data/20260922_lacisa_coco_format/val_predictions \
        --threshold 0.35
"""

import argparse
import glob
import os

import supervision as sv
from PIL import Image

from rfdetr import RFDETR

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def main():
    parser = argparse.ArgumentParser(description="Inferenza RF-DETR su una cartella di immagini")
    parser.add_argument("--weights", required=True, help="checkpoint .pth (es. checkpoint_best_total.pth)")
    parser.add_argument("--images", required=True, help="cartella di immagini, oppure un pattern glob")
    parser.add_argument("--output", required=True, help="cartella dove salvare le immagini annotate")
    parser.add_argument("--threshold", type=float, default=0.5, help="soglia di confidenza (default: 0.5)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    model = RFDETR.from_checkpoint(args.weights)

    if os.path.isdir(args.images):
        paths = sorted(p for p in glob.glob(os.path.join(args.images, "*")) if p.lower().endswith(IMAGE_EXTS))
    else:
        paths = sorted(glob.glob(args.images))

    if not paths:
        raise SystemExit(f"Nessuna immagine trovata in: {args.images}")

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    for path in paths:
        image = Image.open(path).convert("RGB")
        detections = model.predict(image, threshold=args.threshold)
        class_names = detections.data.get("class_name", [])
        labels = [f"{name} {conf:.2f}" for name, conf in zip(class_names, detections.confidence)]

        annotated = box_annotator.annotate(image.copy(), detections)
        annotated = label_annotator.annotate(annotated, detections, labels)
        annotated.save(os.path.join(args.output, os.path.basename(path)))
        print(f"{os.path.basename(path)}: {len(detections)} detection")

    print(f"\n{len(paths)} immagini salvate in: {args.output}")


if __name__ == "__main__":
    main()
