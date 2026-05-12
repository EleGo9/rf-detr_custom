import onnxruntime as ort
import numpy as np
from PIL import Image
from rfdetr import RFDETRNano
import torch
import cv2

# Load the ONNX model
session = ort.InferenceSession("weights/model_ferrari.onnx")

# Prepare input image
# image = Image.open("images/red.png").convert("RGB")
# image = Image.open("images/resized_image.png").convert("RGB")
# image = Image.open("/home/sseveri/rf-detr_custom/images/20260424_4porte_giro_strada_dritta_nonantola_cam_f_1777039829690638875.png").convert("RGB")
# image = image.resize((384, 384))  # Resize to model's input resolution
# output_path = "images/resized_image.png"
# image.save(output_path)

image = cv2.imread("/home/sseveri/rf-detr_custom/images/20260424_4porte_giro_strada_dritta_nonantola_cam_f_1777039829690638875.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# float conversion
image = image / 255.0
# resize
image_array = cv2.resize(image, (384, 384), interpolation=cv2.INTER_LINEAR)

# normalization
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
image_array = ((image_array - mean) / std).astype(np.float32)

image_array = np.transpose(image_array, (2, 0, 1))
image_array = np.expand_dims(image_array, axis=0)

image_array_contiguous = np.ascontiguousarray(image_array, dtype=np.float32)
image_array_contiguous.tofile("onnx_input_tensor_dump.bin")

# Run inference
outputs = session.run(None, {"images": image_array})
onnx_boxes, onnx_labels = outputs


model = RFDETRNano(num_classes=7, pretrain_weights= "weights/checkpoint_best_ema.pth")
detections = model.predict(image, threshold=0.3)
logits, boxes = model.get_last_raw_results()
torch_input_tensor = model.get_last_input_tensor()

torch_logits_np = logits.cpu().numpy()
torch_boxes_np = boxes.cpu().numpy()

print('onnx boxes: ', onnx_boxes.shape)
print(onnx_boxes[:50])
print('onnx labels: ', onnx_labels.shape)
print(onnx_labels[:10])

print("\n" + "="*40)
print("=== CONFRONTO SHAPE ===")
print("="*40)
# Assumiamo che onnx_labels sia in realtà il tensore dei logits
print(f"ONNX Logits shape: {onnx_labels.shape} | PyTorch Logits shape: {torch_logits_np.shape}")
print(f"ONNX Boxes shape:  {onnx_boxes.shape} | PyTorch Boxes shape:  {torch_boxes_np.shape}")

# Assicuriamoci che le dimensioni combacino per evitare errori di broadcasting
if onnx_labels.shape == torch_logits_np.shape and onnx_boxes.shape == torch_boxes_np.shape:
    
    # 2. Calcola le differenze assolute
    diff_logits = np.abs(onnx_labels - torch_logits_np)
    diff_boxes = np.abs(onnx_boxes - torch_boxes_np)

    print("\n" + "="*40)
    print("=== STATISTICHE DIFFERENZE LOGITS ===")
    print("="*40)
    print(f"Differenza Max (Max Error):    {np.max(diff_logits):.8f}")
    print(f"Differenza Media (MAE):        {np.mean(diff_logits):.8f}")
    print(f"Differenza Mediana:            {np.median(diff_logits):.8f}")
    print(f"Deviazione standard errore:    {np.std(diff_logits):.8f}")

    print("\n" + "="*40)
    print("=== STATISTICHE DIFFERENZE BOXES ===")
    print("="*40)
    print(f"Differenza Max (Max Error):    {np.max(diff_boxes):.8f}")
    print(f"Differenza Media (MAE):        {np.mean(diff_boxes):.8f}")
    print(f"Differenza Mediana:            {np.median(diff_boxes):.8f}")
    print(f"Deviazione standard errore:    {np.std(diff_boxes):.8f}")

    # 3. Verifica formale con tolleranza (solitamente si usa 1e-4 o 1e-5 per float32)
    tolerance = 1e-2
    logits_match = np.allclose(onnx_labels, torch_logits_np, atol=tolerance)
    boxes_match = np.allclose(onnx_boxes, torch_boxes_np, atol=tolerance)

    print("\n" + "="*40)
    print("=== ESITO ===")
    print("="*40)
    print(f"Logits identici (tolleranza {tolerance}): {logits_match}")
    print(f"Boxes identici (tolleranza {tolerance}):  {boxes_match}")

else:
    print("\n ERRORE: Le dimensioni dei tensori non combaciano! Impossibile calcolare le statistiche di differenza.")