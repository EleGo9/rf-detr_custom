# rf-detr_custom

Fork locale di [RF-DETR](https://github.com/roboflow/rf-detr) (attualmente allineato alla 1.10.1), con alcune personalizzazioni:

- resize `cv2.INTER_LINEAR` in training/inferenza, per parità con una pipeline C++/OpenCV esterna
- `predict(..., use_cv2_resize=True)` e hook di debug (`get_last_raw_results()`, `get_last_input_tensor()`) per confrontare bit-a-bit output PyTorch vs ONNX/C++
- `export_onnx.py` con supporto nativo al batch dinamico

Il pacchetto vive in `rfdetr/` (layout flat, non `src/`).

## Setup

Ambiente conda: `rfdetr`. Installazione **editable** — ogni modifica ai file in `rfdetr/` è effettiva subito, senza reinstallare:

```bash
conda activate rfdetr
cd rf-detr_custom
pip install -e . --no-deps        # solo il pacchetto, le dipendenze sono già installate
# oppure, dipendenze incluse:
pip install -e ".[train,onnx]"
```

Verifica che punti a questa repo (non a un altro checkout di rf-detr installato in precedenza):

```bash
python -c "import rfdetr; print(rfdetr.__file__)"
```

I dataset usati sono in formato **COCO, layout Roboflow**: una cartella con sottocartelle `train/`, `valid/` (e opzionalmente `test/`), ognuna con le immagini e un file `_annotations.coco.json`.

---

## 1) Convertire un dataset da YOLO a COCO

Script: [`yolo2coco_format.py`](yolo2coco_format.py)

Modifica le costanti in cima al file, poi esegui:

```python
IMAGES_DIR_PATH = "/path/to/yolo/images/"       # contiene train/ e valid/
ANNOTATIONS_DIR_PATH = "/path/to/yolo/labels/"   # contiene train/ e valid/
DATA_YAML_PATH = "/path/to/data.yaml"
OUTPUT_DIR = "/path/to/output_coco_format"
KEEP_CLASSES = ["truck", "car", "person", "forklift"]  # None per tenerle tutte
```

```bash
python yolo2coco_format.py
```

Genera `OUTPUT_DIR/train/_annotations.coco.json` e `OUTPUT_DIR/valid/_annotations.coco.json` (+ immagini), pronti per il training.

Lo split di validazione può chiamarsi `valid/`, `val/` o `validation/` in ingresso (rilevato automaticamente); in uscita viene sempre scritto come `valid/`, il nome che il resto della pipeline (`train.py`, `eval_dataset.py`) si aspetta. Se esiste anche uno split `test/` (o `testing/`) viene convertito automaticamente; se non c'è, viene semplicemente saltato.

---

## 2) Avviare un training

Script: [`train.py`](train.py)

Modifica le costanti in cima al file:

```python
DATASET = "/path/to/dataset_coco_format"   # cartella con train/ e valid/
epochs = 100
num_classes = 4
freeze_encoder = False
early_stopping = True
early_stopping_patience = 20
batch_size = 8
grad_accum_steps = 2   # batch effettivo = batch_size * grad_accum_steps
output_dir = "/path/to/output"
```

```bash
python train.py
```

`num_classes` e `freeze_encoder` vanno passati al **costruttore** del modello (`RFDETRNano(num_classes=..., freeze_encoder=...)`), non a `.train(...)`: dalla 1.8 in poi `TrainConfig` rifiuta kwarg non suoi.

Output in `output_dir/`:
- `checkpoint_best_regular.pth`, `checkpoint_best_ema.pth`, `checkpoint_best_total.pth` (il migliore tra i due — di solito questo è il checkpoint da usare), `last.pth`/`last_ema.pth`
- `metrics.csv` — curve di train/val per epoca (loss, mAP, precision, recall, F1)
- `training_config.json` — configurazione effettiva usata

Il training è basato su PyTorch Lightning: se non migliora per `early_stopping_patience` epoche di fila si ferma da solo.

---

## 3) Inferenza su una cartella di immagini e salvataggio

Script: [`predict_images.py`](predict_images.py)

```bash
python predict_images.py \
  --weights output_dir/checkpoint_best_total.pth \
  --images /path/to/images \
  --output /path/to/predictions \
  --threshold 0.35
```

Disegna box + classe + confidenza su ogni immagine (via `supervision`) e le salva in `--output`, mantenendo il nome file originale. `--images` accetta sia una cartella sia un pattern glob (es. `"/path/*.jpg"`).

---

## 4) Inferenza + metriche su un dataset (formato COCO)

Script: [`eval_dataset.py`](eval_dataset.py)

```bash
python eval_dataset.py \
  --weights output_dir/checkpoint_best_total.pth \
  --dataset-dir /path/to/dataset_coco_format \
  --split val \
  --per-class
```

Stampa e restituisce mAP@50:95, mAP@50, mAP@75, mAR, F1, precision, recall (globali e, con `--per-class`, per ogni classe). `--split test` valuta `test/` invece di `valid/`.

---

## 5) Esportare in ONNX

Script: [`export_onnx.py`](export_onnx.py)

```bash
python export_onnx.py \
  --weights output_dir/checkpoint_best_total.pth \
  --num-classes 4 \
  --output output/model.onnx \
  --test
```

Batch dinamico **attivo di default** (asse `batch` sull'input `input` e sugli output `dets`/`labels`); usa `--static-batch` per un batch fisso pari a `--batch-size`. `--simplify` passa il grafo per `onnxsim`; `--test` verifica l'inferenza con `onnxruntime` dopo l'export, a più batch size se dinamico.

---

## Capire a quale id corrisponde ogni classe

Script: [`show_class_mapping.py`](show_class_mapping.py)

```bash
python show_class_mapping.py \
  --weights output_dir/checkpoint_best_total.pth \
  --dataset-dir /path/to/dataset_coco_format   # opzionale: aggiunge la colonna category_id
```

Esempio (lacisa):

| label index (= `class_id` di `predict()`) | nome | `category_id` nel dataset |
|---|---|---|
| 0 | truck | 1 |
| 1 | car | 2 |
| 2 | person | 3 |
| 3 | forklift | 4 |
| 4 | *(background, mai una classe vera)* | — |

Regole generali:
- `predict()` restituisce `class_id` = **label index 0-based**, e `class_names[class_id]` è il nome (in `detections.data["class_name"]` c'è già risolto).
- Il label index è la **posizione della categoria nella lista `categories`** del `_annotations.coco.json` di `train/`, non `category_id - 1`: coincidono solo se gli id del dataset sono contigui da 1. Se il dataset ha id tipo 0,1,2 o 5,7,9 la mappa cambia — usa lo script invece di dedurla.
- Le classi sono salvate nel checkpoint (`args["class_names"]`) e in `training_config.json`.
- `class_embed` ha `N+1` righe: le prime `N` sono le classi, l'ultima è lo slot di "nessun oggetto". Per questo `num_classes` deve essere **uguale** al numero reale di categorie del dataset.

---

## Altri script nella repo

- [`debug.py`](debug.py) — confronto manuale output PyTorch vs ONNX su un singolo input, usando gli hook `get_last_raw_results()` / `get_last_input_tensor()`.
- [`rfdetr/test.py`](rfdetr/test.py), [`rfdetr/test_from_video.py`](rfdetr/test_from_video.py) — script personali di prova/confronto, non parte dell'API pubblica.
