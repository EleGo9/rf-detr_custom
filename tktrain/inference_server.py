import argparse
import os
from contextlib import asynccontextmanager
from typing import List

import cv2
import numpy as np
import supervision as sv
import torch
import uvicorn
import yaml
from easydict import EasyDict
from fastapi import FastAPI, File, UploadFile
from PIL import Image

# Import specifici per RF-DETR
from rfdetr import RFDETR
from rfdetr.assets.model_weights import download_pretrain_weights, get_model_cache_dir

parser = argparse.ArgumentParser(description="RF-DETR Inference Server")
parser.add_argument(
    "--cfg", type=str, required=True, help="Path to the config YAML file"
)
args = parser.parse_args()

MODEL = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_config(path):
    with open(path, "r") as file:
        data = yaml.safe_load(file)
    return EasyDict(data)


def load_rfdetr_model(weights: str) -> RFDETR:
    """
    Carica un modello RF-DETR gestendo sia checkpoint custom locali
    sia il download automatico dei pesi base ufficiali.
    """
    # 1. Se è un nome file puro (senza slash) e non esiste nella cartella corrente,
    # trattalo come un modello base da scaricare nella cache.
    if not os.path.dirname(weights) and not os.path.exists(weights):
        cache_dir = get_model_cache_dir()
        os.makedirs(cache_dir, exist_ok=True)
        cached_weights_path = os.path.join(cache_dir, weights)

        # Scarica i pesi se non sono già nella cache
        if not os.path.exists(cached_weights_path):
            print(f"⬇️ Download modello base {weights} in cache...")
            download_pretrain_weights(cached_weights_path)

        weights = cached_weights_path

    # 2. Carica il modello. RFDETR.from_checkpoint deduce l'architettura in automatico.
    return RFDETR.from_checkpoint(weights)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL
    cfg = load_config(args.cfg)

    print(f"🚀 Caricamento rete RF-DETR in VRAM ({DEVICE})...")
    model_path = cfg.model
    print(f"   WEIGHTS/MODEL: {model_path}")

    # Utilizziamo la nuova funzione wrapper per gestire sia cache che percorsi locali
    MODEL = load_rfdetr_model(model_path)

    print("✅ Modello RF-DETR caricato con successo!")
    yield
    print("🛑 Spegnimento server RF-DETR...")


app = FastAPI(lifespan=lifespan)


@app.get("/get_labels")
async def get_labels():
    if MODEL is None:
        return {"error": "Modello non ancora caricato"}, 500

    try:
        class_names = MODEL.class_names
        if isinstance(class_names, list):
            stringified_labels = {str(i): name for i, name in enumerate(class_names)}
        else:
            stringified_labels = {str(k): v for k, v in class_names.items()}
    except AttributeError:
        stringified_labels = {"0": "sconosciuto - verifica attributi modello"}

    return {"names": stringified_labels}


@app.post("/predict_batch")
async def predict_batch(thr: float = 0.5, files: List[UploadFile] = File(...)):
    images = []

    # 1. Decodifica immagini
    for file in files:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        images.append(pil_img)

    all_annotations = []

    # 2. Inferenza con RF-DETR
    with torch.no_grad():
        for pil_img in images:
            detections = MODEL.predict(pil_img, threshold=thr)

            frame_annotations = []

            if len(detections) == 0:
                all_annotations.append(frame_annotations)
                continue

            class_names_list = detections.data.get("class_name", [])

            for i in range(len(detections)):
                box = detections.xyxy[i]
                score = detections.confidence[i]

                if class_names_list is not None and len(class_names_list) > i:
                    class_name = class_names_list[i]
                else:
                    class_id = detections.class_id[i]
                    class_name = str(class_id)

                x1, y1, x2, y2 = (
                    float(box[0]),
                    float(box[1]),
                    float(box[2]),
                    float(box[3]),
                )

                frame_annotations.append(
                    {
                        "class_name": class_name,
                        "score": float(score),
                        "box": [x1, y1, x2, y2],
                    }
                )

            all_annotations.append(frame_annotations)

    return {"annotations": all_annotations}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
