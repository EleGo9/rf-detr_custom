import io
import requests
import supervision as sv
from PIL import Image, ImageDraw, ImageFont
from rfdetr import RFDETRNano
from util.coco_classes import COCO_CLASSES
from util.berkley_classes import BERKLEY_CLASSES #import a file with the classes of your dataset during training. 
import glob
import os
import random

state_dict = None
# Compile with your model data
# Config:
num_classes = 90
weights_path = "/home/elena/repos/rf-detr/rf-detr-nano_berkley_cv-resize/checkpoint_best_total.pth"
# images_files = "/media/elena/T7/rfdetr_ferrari/20260424_4porte_giro_strada_dritta_nonantola/images/*"
images_files = "/home/elena/repos/HAura_new/tkHAura/build/sim_data/*"
output = "/home/elena/repos/HAura_new/tkHAura/build/sim_data/rfdetr_preds/"
model1 = RFDETRNano() #COCO weights (default)
model2 = RFDETRNano(num_classes=num_classes, pretrain_weights= weights_path)



IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
images = [
    f for f in glob.glob(images_files)
    if os.path.isfile(f) and f.lower().endswith(IMG_EXTS)
]
for n, image_filename in enumerate(images):
    image = Image.open(image_filename).convert("RGB")
    detections1 = model1.predict(image, threshold=0.4)
    detections2 = model2.predict(image, threshold=0.4)
    

    labels = [
        f"{BERKLEY_CLASSES[class_id]} {confidence:.2f}"
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

    banner_h = 28
    left_label = f"berkley weights: {os.path.basename(weights_path)}"
    right_label = "coco weights (RFDETRNano default)"
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    comparison = Image.new('RGB', (annotated_image.width * 2, annotated_image.height + banner_h), (0, 0, 0))
    comparison.paste(annotated_image, (0, banner_h))
    comparison.paste(annotated_image2, (annotated_image.width, banner_h))

    draw = ImageDraw.Draw(comparison)
    draw.text((5, 5), left_label, fill=(255, 255, 255), font=font)
    draw.text((annotated_image.width + 5, 5), right_label, fill=(255, 255, 255), font=font)

    comparison.save(f"{output}rfdetr_comparison_{img_name}")


    # annotated_image.save(f"{output}rfdetr_{img_name}")
    # if n>=100:
    #     break
    # sv.save_image(annotated_image, f"output/rfdetr_preds/{os.path.basename(image.filename)}")