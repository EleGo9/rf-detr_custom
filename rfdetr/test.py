import io
import requests
import supervision as sv
from PIL import Image
from rfdetr import RFDETRNano
from util.coco_classes import COCO_CLASSES
import glob
import os
import random

state_dict = None
# Compile with your model data
# Config:
num_classes = 90
weights_path = "/path/to/weights.pth"
images_files = "path/to/images/*"
output = "path/to/predictions"
model1 = RFDETRNano() #COCO weights (default)
model2 = RFDETRNano(num_classes=num_classes, pretrain_weights= weights_path)



images = glob.glob(images_files)
for n, image_filename in enumerate(images):
    image = Image.open(image_filename).convert("RGB")
    detections1 = model1.predict(image, threshold=0.3)
    detections2 = model2.predict(image, threshold=0.3)
    

    labels = [
        f"{COCO_CLASSES[class_id]} {confidence:.2f}"
        for class_id, confidence
        in zip(detections2.class_id, detections2.confidence)
    ]

    annotated_image = image.copy()
    annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections2)
    annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections2, labels)
    # sv.plot_image(annotated_image)
    img_name = os.path.basename(image_filename)
    # annotated_image.save(f"{output}rfdetr_fine-tuned_{img_name}")

    labels = [
        f"{COCO_CLASSES[class_id]} {confidence:.2f}"
        for class_id, confidence
        in zip(detections1.class_id, detections1.confidence)
    ]
    # print(labels)
    # print(detections1)

    annotated_image2 = image.copy()
    annotated_image2 = sv.BoxAnnotator().annotate(annotated_image2, detections1)
    annotated_image2 = sv.LabelAnnotator().annotate(annotated_image2, detections1, labels)
    # sv.plot_image(annotated_image2)
    img_name = os.path.basename(image_filename)
    comparison = Image.new('RGB', (annotated_image.width * 2, annotated_image.height))
    comparison.paste(annotated_image, (0, 0))
    comparison.paste(annotated_image2, (annotated_image.width, 0))
    comparison.save(f"{output}rfdetr_comparison_{img_name}")
    # annotated_image.save(f"{output}rfdetr_original_{img_name}")
    if n>=100:
        break
    # sv.save_image(annotated_image, f"output/rfdetr_preds/{os.path.basename(image.filename)}")