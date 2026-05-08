import onnxruntime as ort
import numpy as np
from PIL import Image
from rfdetr import RFDETRNano

# Load the ONNX model
session = ort.InferenceSession("output/model_ferrari.onnx")

# Prepare input image
image = Image.open("path/to/image").convert("RGB")
image = image.resize((384, 384))  # Resize to model's input resolution
image_array = np.array(image).astype(np.float32) / 255.0

# Normalize
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
image_array = ((image_array - mean) / std).astype(np.float32)

# Convert to NCHW format
image_array = np.transpose(image_array, (2, 0, 1))
image_array = np.expand_dims(image_array, axis=0)

# Run inference
outputs = session.run(None, {"images": image_array})
boxes, labels = outputs



model = RFDETRNano(num_classes=7, pretrain_weights= "/home/elena/repos/rf-detr/ferrari_on_ernesto/berkley/checkpoint_best_ema.pth")
detections = model.predict(image, threshold=0.3)
print('onnx boxes: ', boxes.shape)
print(boxes[:50])
print('onnx labels: ', labels.shape)
print(labels[:10])

# NB: to complete the debug, uncomment print functions into /home/elena/repos/rf-detr_custom/rfdetr/detr.py