import supervision as sv
import numpy as np
import os
import yaml


def write_filtered_yaml(source_yaml_path, keep_classes, output_dir):
    with open(source_yaml_path) as f:
        data = yaml.safe_load(f)
    data['names'] = keep_classes
    data['nc'] = len(keep_classes)
    out_path = os.path.join(output_dir, "data.yaml")
    os.makedirs(output_dir, exist_ok=True)
    with open(out_path, 'w') as f:
        yaml.dump(data, f, allow_unicode=True)
    print(f"  data.yaml aggiornato salvato in {out_path}")


def filter_classes(dataset, keep_classes, drop_empty_images=True):
    """
    Filtra un DetectionDataset tenendo solo le classi in keep_classes.
    Gli ID vengono rimappati a 0, 1, 2, ... nell'ordine di keep_classes.
    Se drop_empty_images=True, le immagini senza annotazioni vengono escluse.
    """
    old_ids = [dataset.classes.index(c) for c in keep_classes]
    id_map = {old: new for new, old in enumerate(old_ids)}

    new_annotations = {}
    for path, detections in dataset.annotations.items():
        mask = np.isin(detections.class_id, old_ids)
        filtered = detections[mask]
        filtered.class_id = np.array([id_map[c] for c in filtered.class_id])
        if drop_empty_images and len(filtered) == 0:
            continue
        new_annotations[path] = filtered

    new_image_paths = [p for p in dataset.image_paths if p in new_annotations]
    print(f"  Immagini mantenute: {len(new_image_paths)} / {len(dataset.image_paths)}")
    return sv.DetectionDataset(classes=keep_classes, images=new_image_paths, annotations=new_annotations)

# KEEP_CLASSES = None # (all)
KEEP_CLASSES = ["truck", "vtlm", "dump", "excavator", "tractor", "bulldozer", "flatbed"]

OUTPUT_DIR = "where/you/want/to/save/your/dataset/"

dataset_train = sv.DetectionDataset.from_yolo(
    images_directory_path="/path/to/images/train",
    annotations_directory_path="/path/to/labels/train/",
    data_yaml_path="/path/to/data.yaml"
)
print(f"Train images: {len(dataset_train)}")
if KEEP_CLASSES:
    dataset_train = filter_classes(dataset_train, KEEP_CLASSES)

dataset_val = sv.DetectionDataset.from_yolo(
    images_directory_path="/path/to/images/valid",
    annotations_directory_path="/path/to/labels/val",
    data_yaml_path="/path/to/data.yaml"
)
print(f"Valid images: {len(dataset_val)}")
if KEEP_CLASSES:
    dataset_val = filter_classes(dataset_val, KEEP_CLASSES)
    write_filtered_yaml(
        "/path/to/data.yaml",
        KEEP_CLASSES,
        OUTPUT_DIR
    )

# Crea directory output
os.makedirs(f"{OUTPUT_DIR}/train", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/valid", exist_ok=True)

# Salva TRAIN in formato COCO
dataset_train.as_coco(
    images_directory_path=f"{OUTPUT_DIR}/train",
    annotations_path=f"{OUTPUT_DIR}/train/_annotations.coco.json"
)
print(f"Train saved to {OUTPUT_DIR}/train/")

# Salva VALID in formato COCO
dataset_val.as_coco(
    images_directory_path=f"{OUTPUT_DIR}/valid",
    annotations_path=f"{OUTPUT_DIR}/valid/_annotations.coco.json"
)
print(f"Valid saved to {OUTPUT_DIR}/valid/")