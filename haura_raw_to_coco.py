"""Converte i dataset raw HAura (cam_r/cam_l con *_gt.json + *.png) in formato COCO Roboflow.

Formato di origine: per ogni timestamp, un file <ts>.png (immagine) e un file <ts>_gt.json
con gli oggetti annotati in pixel assoluti, angolo in alto a sinistra:
    {"width":, "height":, "objects": [{"class": <id raw>, "x":, "y":, "w":, "h":, ...}, ...]}

Le classi dei dataset raw HAura usano una numerazione propria (vedi RAW_CLASS_TO_NAME),
diversa da quella dei dataset COCO già pronti per il training (es. lacisa). Per allenare
su entrambi con lo stesso num_classes/ordine, qui si rimappano esplicitamente sugli stessi
category_id usati per lacisa (truck=1, car=2, person=3, forklift=4). "car" resta nella
lista delle categorie anche se in questi dataset non ha nessuna istanza, per restare
compatibile con un checkpoint/training basato su num_classes=4.
"""

import json
import os
import random
import shutil

INPUT_DATASETS = [
    "/home/elena/repos/rf-detr_custom/data/complete_dataset_haura1",
    "/home/elena/repos/rf-detr_custom/data/complete_dataset_haura2",
    "/home/elena/repos/rf-detr_custom/data/complete_dataset_haura3",
]
CAMERAS = ["cam_r", "cam_l"]
OUTPUT_DIR = "/home/elena/repos/rf-detr_custom/data/haura123_coco_format"
VAL_FRACTION = 0.15
SEED = 42

# id di classe nei _gt.json raw -> nome
RAW_CLASS_TO_NAME = {
    25: "person",
    26: "truck",
    27: "forklift",
}

# stessi nomi/ordine/category_id usati nel training su lacisa
CATEGORIES = [
    {"id": 1, "name": "truck", "supercategory": "common-objects"},
    {"id": 2, "name": "car", "supercategory": "common-objects"},
    {"id": 3, "name": "person", "supercategory": "common-objects"},
    {"id": 4, "name": "forklift", "supercategory": "common-objects"},
]
NAME_TO_CAT_ID = {c["name"]: c["id"] for c in CATEGORIES}


def collect_pairs():
    """Ritorna la lista di (dataset_dir, cam, timestamp) per ogni immagine con annotazioni."""
    pairs = []
    for ds in INPUT_DATASETS:
        for cam in CAMERAS:
            cam_dir = os.path.join(ds, cam)
            if not os.path.isdir(cam_dir):
                continue
            for fname in os.listdir(cam_dir):
                if fname.endswith("_gt.json"):
                    ts = fname[: -len("_gt.json")]
                    if os.path.isfile(os.path.join(cam_dir, f"{ts}.png")):
                        pairs.append((ds, cam, ts))
    return pairs


def build_split(pairs, output_split_dir):
    os.makedirs(output_split_dir, exist_ok=True)
    images = []
    annotations = []
    image_id = 1
    ann_id = 1
    skipped_unknown_class = 0

    for ds, cam, ts in pairs:
        cam_dir = os.path.join(ds, cam)
        ds_name = os.path.basename(ds)
        out_fname = f"{ds_name}_{cam}_{ts}.png"
        shutil.copy(os.path.join(cam_dir, f"{ts}.png"), os.path.join(output_split_dir, out_fname))

        gt = json.load(open(os.path.join(cam_dir, f"{ts}_gt.json")))
        images.append({
            "id": image_id,
            "file_name": out_fname,
            "width": gt["width"],
            "height": gt["height"],
        })

        for obj in gt.get("objects", []):
            name = RAW_CLASS_TO_NAME.get(obj["class"])
            if name is None:
                skipped_unknown_class += 1
                continue
            x, y, w, h = obj["x"], obj["y"], obj["w"], obj["h"]
            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": NAME_TO_CAT_ID[name],
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0,
            })
            ann_id += 1
        image_id += 1

    coco = {"info": {}, "licenses": [], "categories": CATEGORIES, "images": images, "annotations": annotations}
    with open(os.path.join(output_split_dir, "_annotations.coco.json"), "w") as f:
        json.dump(coco, f)

    return len(images), len(annotations), skipped_unknown_class


def main():
    random.seed(SEED)
    pairs = collect_pairs()
    print(f"Coppie immagine+annotazioni trovate: {len(pairs)}")
    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * VAL_FRACTION))
    val_pairs, train_pairs = pairs[:n_val], pairs[n_val:]

    for split_name, split_pairs in [("train", train_pairs), ("valid", val_pairs)]:
        n_img, n_ann, n_skip = build_split(split_pairs, os.path.join(OUTPUT_DIR, split_name))
        msg = f"{split_name}: {n_img} immagini, {n_ann} annotazioni"
        if n_skip:
            msg += f", {n_skip} oggetti con classe raw ignota saltati"
        print(msg)

    print(f"\nDataset salvato in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
