import json
import random
from PIL import Image, ImageDraw, ImageFont

ANN_JSON = "/media/elena/T7/BDD100K/coco/train/_annotations.coco.json"
OUTPUT_DIR = "/home/elena/repos/rf-detr/bbox_viz/"
N_SAMPLES = 10

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(ANN_JSON) as f:
    data = json.load(f)

cats = {c["id"]: c["name"] for c in data["categories"]}
anns_by_image = {}
for ann in data["annotations"]:
    anns_by_image.setdefault(ann["image_id"], []).append(ann)

images_with_anns = [img for img in data["images"] if img["id"] in anns_by_image]
samples = random.sample(images_with_anns, min(N_SAMPLES, len(images_with_anns)))

COLORS = ["#e6194b","#3cb44b","#ffe119","#4363d8","#f58231","#911eb4","#42d4f4","#f032e6","#bfef45","#fabed4"]

for img_info in samples:
    img = Image.open(img_info["file_name"]).convert("RGB")
    draw = ImageDraw.Draw(img)

    for ann in anns_by_image[img_info["id"]]:
        x, y, w, h = ann["bbox"]
        cid = ann["category_id"]
        color = COLORS[cid % len(COLORS)]
        draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
        draw.text((x + 4, y + 2), cats[cid], fill=color)

    out_name = os.path.basename(img_info["file_name"])
    img.save(os.path.join(OUTPUT_DIR, out_name))
    print(f"Saved: {out_name} ({len(anns_by_image[img_info['id']])} bbox)")

print(f"\nImages save in {OUTPUT_DIR}")
