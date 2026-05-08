import sys
import os
import cv2
import numpy as np
from PIL import Image
import supervision as sv
from rfdetr import RFDETRNano, RFDETRSmall
from util.coco_classes import COCO_CLASSES

THRESHOLD = 0.3
TARGET_FPS = 5  # frame al secondo da processare (None = tutti)

if len(sys.argv) < 2:
    print("Uso: python test_from_video.py <video.mp4> [output.mp4]")
    sys.exit(1)

input_path = sys.argv[1]
output_path = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(input_path)[0] + "_annotated.mp4"

# Choose the model you prefer:
# model = RFDETRNano()
# model = RFDETRSmall()
model = RFDETRNano(shape=864)

cap = cv2.VideoCapture(input_path)
src_fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

out_fps = TARGET_FPS if TARGET_FPS else src_fps
# ogni quanti frame originali prenderne uno
step = max(1, round(src_fps / out_fps)) if TARGET_FPS else 1

print(f"Video: {src_fps:.1f} fps, {total} frame totali")
print(f"Processo 1 frame ogni {step} -> output a {src_fps/step:.1f} fps effettivi")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(output_path, fourcc, src_fps / step, (width, height))

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

n = 0
written = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if n % step == 0:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        detections = model.predict(image, threshold=THRESHOLD)

        labels = [
            f"{COCO_CLASSES[cid]} {conf:.2f}"
            for cid, conf in zip(detections.class_id, detections.confidence)
        ]

        annotated = box_annotator.annotate(image.copy(), detections)
        annotated = label_annotator.annotate(annotated, detections, labels)

        out_frame = cv2.cvtColor(np.array(annotated), cv2.COLOR_RGB2BGR)
        writer.write(out_frame)
        written += 1

        if written % 50 == 0:
            print(f"  Frame processati: {written} ({n}/{total} letti)")

    n += 1

cap.release()
writer.release()
print(f"Salvato: {output_path} ({written} frame scritti)")
