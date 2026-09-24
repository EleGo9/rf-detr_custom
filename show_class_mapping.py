"""Mostra a quale id/slot corrisponde ogni classe per un checkpoint RF-DETR.

Tre "id" diversi coesistono e spesso si confondono:
  - category_id  : l'id nel _annotations.coco.json del dataset (spesso parte da 1)
  - label index  : posizione 0-based della classe (è ciò che il modello impara e ciò che
                   predict() restituisce in `class_id`; `class_names[label_index]` = nome)
  - slot logit   : colonna di class_embed; le prime N coincidono col label index, l'ultima
                   (indice N) è lo slot extra di "nessun oggetto", mai una classe vera

Esempi:
    python show_class_mapping.py --weights data/20260922_lacisa_coco_format/checkpoint_best_total.pth
    python show_class_mapping.py --weights ...pth --dataset-dir /media/elena/T9/HAura/mcap/lacisa_coco_format
"""

import argparse
from pathlib import Path

import torch

from rfdetr import RFDETR
from rfdetr.datasets.coco import _train_split_cat2label


def main():
    parser = argparse.ArgumentParser(description="Mappa classe <-> id per un checkpoint RF-DETR")
    parser.add_argument("--weights", required=True, help="checkpoint .pth")
    parser.add_argument("--dataset-dir", help="dataset COCO (con train/): aggiunge la colonna category_id")
    args = parser.parse_args()

    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    class_embed_rows = ckpt["model"]["class_embed.weight"].shape[0]
    model = RFDETR.from_checkpoint(args.weights)
    class_names = model.class_names

    label_to_cat = {}
    if args.dataset_dir:
        cat2label = _train_split_cat2label(Path(args.dataset_dir))
        if cat2label is None:
            raise SystemExit(f"Impossibile leggere {args.dataset_dir}/train/_annotations.coco.json")
        label_to_cat = {label: cat for cat, label in cat2label.items()}

    print(f"\nCheckpoint : {args.weights}")
    print(f"Classi     : {len(class_names)}  |  righe di class_embed: {class_embed_rows}\n")
    header = f"{'label index (= class_id)':<26}{'nome':<16}"
    if label_to_cat:
        header += f"{'category_id nel dataset':<26}"
    print(header)
    print("-" * len(header))
    for i, name in enumerate(class_names):
        row = f"{i:<26}{name:<16}"
        if label_to_cat:
            row += f"{label_to_cat.get(i, '?'):<26}"
        print(row)
    if class_embed_rows == len(class_names) + 1:
        print(f"{class_embed_rows - 1:<26}{'(background / nessun oggetto)':<16}")
    elif class_embed_rows == len(class_names):
        print("\nNota: class_embed non ha lo slot di background extra (checkpoint del vecchio "
              "codice pre-1.10: le righe = classi reali, vedi README).")


if __name__ == "__main__":
    main()
