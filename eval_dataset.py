"""Valuta un checkpoint RF-DETR su uno split di un dataset in formato COCO.

Il dataset deve avere il layout Roboflow standard (cartelle train/valid/test, ognuna
con un _annotations.coco.json) — lo stesso prodotto da yolo2coco_format.py.

Esempio (modello fine-tuned, dataset con le sue classi "proprie"):
    python eval_dataset.py \
        --weights data/20260922_lacisa_coco_format/checkpoint_best_total.pth \
        --dataset-dir /media/elena/T9/HAura/mcap/lacisa_coco_format \
        --split val --per-class

Per valutare un checkpoint COCO *pre-addestrato, non fine-tuned* (es. num_classes=90)
su un dataset che usa i veri category_id sparsi di COCO (1-90, con buchi) per le sue
classi, passa --dataset-file coco: con il default "roboflow" le label vengono rimappate
per POSIZIONE nella lista categories del json (category_id viene ignorato ai fini della
label), il che non corrisponde a come un modello COCO a 90 classi indicizza il suo
output (lì l'indice coincide col category_id sparso). --dataset-file coco usa invece
category_id direttamente come label, senza alcuna rimappatura posizionale — ma il
loader "coco" di rfdetr si aspetta il layout rigido di COCO2017 (train2017/, val2017/,
annotations/instances_*2017.json), non le cartelle train/valid di Roboflow: questo
script crea in automatico (con symlink, nessuna copia) una vista temporanea in quel
layout a partire dal tuo --dataset-dir, quando passi --dataset-file coco.
"""

import argparse
import json
import os
import tempfile

from rfdetr import RFDETR


def _stage_coco2017_layout(dataset_dir, split):
    """Crea una vista temporanea nel layout COCO2017 a partire da un dataset Roboflow.

    Il loader dataset_file="coco" di rfdetr vuole <root>/<split>2017/ per le immagini e
    <root>/annotations/instances_<split>2017.json per le annotazioni. Qui si creano solo
    symlink (niente copie) a partire dalla cartella Roboflow train/valid/test esistente.
    """
    split_dir_name = {"val": "valid", "test": "test"}[split]
    src_img_dir = os.path.join(dataset_dir, split_dir_name)
    src_ann = os.path.join(src_img_dir, "_annotations.coco.json")
    if not os.path.isdir(src_img_dir) or not os.path.isfile(src_ann):
        raise FileNotFoundError(
            f"Attesi {src_img_dir}/ e {src_ann} (layout Roboflow) per costruire la vista COCO2017."
        )

    staging = tempfile.mkdtemp(prefix="rfdetr_coco2017_")
    os.symlink(os.path.abspath(src_img_dir), os.path.join(staging, f"{split}2017"))
    ann_dir = os.path.join(staging, "annotations")
    os.makedirs(ann_dir)
    os.symlink(os.path.abspath(src_ann), os.path.join(ann_dir, f"instances_{split}2017.json"))
    return staging


def main():
    parser = argparse.ArgumentParser(description="Valuta RF-DETR su un dataset COCO")
    parser.add_argument("--weights", required=True, help="checkpoint .pth da valutare")
    parser.add_argument("--dataset-dir", required=True, help="cartella del dataset (contiene valid/ e/o test/)")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--per-class", action="store_true", help="stampa anche AP/AR per classe")
    parser.add_argument(
        "--dataset-file",
        choices=["roboflow", "coco", "o365", "yolo"],
        default="roboflow",
        help=(
            "Default 'roboflow': category_id rimappato per posizione nella lista categories "
            "(per modelli fine-tuned sulle proprie classi). 'coco': category_id usato "
            "direttamente come label, senza rimappatura — per valutare un checkpoint COCO "
            "a 90 classi non fine-tuned con un dataset che usa i veri category_id sparsi di COCO."
        ),
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    if args.dataset_file == "coco":
        dataset_dir = _stage_coco2017_layout(args.dataset_dir, args.split)
        print(f"Vista COCO2017 temporanea (symlink): {dataset_dir}")

    model = RFDETR.from_checkpoint(args.weights)
    metrics = model.evaluate(
        split=args.split,
        dataset_dir=dataset_dir,
        dataset_file=args.dataset_file,
        batch_size=args.batch_size,
        log_per_class_metrics=args.per_class,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
