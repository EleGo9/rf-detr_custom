"""
Script custom per esportare RF-DETR in ONNX.
Supporta tutti i modelli: Nano, Small, Medium, Base, Large.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
import numpy as np
from pathlib import Path
from copy import deepcopy

# Patch per versioni recenti di onnx che non hanno float32_to_bfloat16
import onnx.helper
from onnx import TensorProto

if not hasattr(onnx.helper, 'float32_to_bfloat16'):
    def float32_to_bfloat16(tensor):
        return onnx.helper.make_tensor(
            name=tensor.name,
            data_type=TensorProto.BFLOAT16,
            dims=tensor.dims,
            vals=tensor.raw_data,
            raw=True
        )
    onnx.helper.float32_to_bfloat16 = float32_to_bfloat16


def replace_unsupported_ops(model):
    """
    Sostituisce operazioni non supportate da ONNX nel modello.
    In particolare, sostituisce bicubic con antialiasing con bilinear.
    """
    # Patch F.interpolate per usare bilinear invece di bicubic con antialias
    original_interpolate = F.interpolate

    def patched_interpolate(input, size=None, scale_factor=None, mode='nearest',
                           align_corners=None, recompute_scale_factor=None, antialias=False):
        # Se è bicubic con antialias, usa bilinear senza antialias
        if mode == 'bicubic' and antialias:
            mode = 'bilinear'
            antialias = False
        # Se è bilinear con antialias, rimuovi antialias
        if antialias:
            antialias = False
        return original_interpolate(
            input, size=size, scale_factor=scale_factor, mode=mode,
            align_corners=align_corners, recompute_scale_factor=recompute_scale_factor,
            antialias=antialias
        )

    F.interpolate = patched_interpolate
    return original_interpolate


def restore_ops(original_interpolate):
    F.interpolate = original_interpolate


def export_rfdetr_to_onnx(
    model_class: str = "RFDETRNano",
    weights_path: str = None,
    num_classes: int = None,
    output_path: str = "model.onnx",
    resolution: int = None,  # Se None, usa la risoluzione di default del modello
    batch_size: int = 1,
    opset_version: int = 18,  # Usa opset 18 per compatibilità con PyTorch recenti
    simplify: bool = False,
    device: str = "cpu",
):
    """
    Esporta un modello RF-DETR in formato ONNX.

    Args:
        model_class: Classe del modello ("RFDETRNano", "RFDETRSmall", "RFDETRMedium", "RFDETRBase", "RFDETRLarge")
        weights_path: Percorso ai pesi del modello (opzionale, usa pesi pretrained se None)
        output_path: Percorso di output per il file ONNX
        resolution: Risoluzione dell'immagine di input (None = usa default del modello)
        batch_size: Batch size per l'export (default: 1)
        opset_version: Versione ONNX opset (default: 18)
        simplify: Se True, semplifica il modello ONNX (richiede onnxsim)
        device: Device per l'export ("cpu" o "cuda")
    """
    from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRBase, RFDETRLarge

    # Mappa delle classi
    model_classes = {
        "RFDETRNano": RFDETRNano,
        "RFDETRSmall": RFDETRSmall,
        "RFDETRMedium": RFDETRMedium,
        "RFDETRBase": RFDETRBase,
        "RFDETRLarge": RFDETRLarge,
    }

    
    kwargs = {}
    if weights_path:
        kwargs["pretrain_weights"] = weights_path
    if num_classes:
        kwargs["num_classes"] = num_classes
    model = model_classes[model_class](**kwargs)

    # Usa la risoluzione del modello se non specificata
    if resolution is None:
        resolution = model.model.resolution
        print(f"Usando risoluzione di default del modello: {resolution}")

    # Ottieni il modello interno PyTorch
    pytorch_model = deepcopy(model.model.model)
    pytorch_model.eval()

    # Prepara per l'export
    if hasattr(pytorch_model, 'export'):
        pytorch_model.export()

    # Sposta su CPU per l'export (ONNX export funziona meglio su CPU)
    pytorch_model = pytorch_model.to("cpu")
    pytorch_model = pytorch_model.float()  # Assicura float32

    # Crea input dummy
    dummy_input = torch.randn(batch_size, 3, resolution, resolution, dtype=torch.float32, device="cpu")

    # Applica patch per operazioni non supportate da ONNX
    print("Applicazione patch per compatibilità ONNX...")
    original_interpolate = replace_unsupported_ops(pytorch_model)

    # Test forward pass
    print("Test forward pass...")
    with torch.no_grad():
        outputs = pytorch_model(dummy_input)
        if isinstance(outputs, dict):
            print(f"  - pred_boxes shape: {outputs['pred_boxes'].shape}")
            print(f"  - pred_logits shape: {outputs['pred_logits'].shape}")
        else:
            print(f"  - Output shape: {outputs[0].shape if isinstance(outputs, tuple) else outputs.shape}")

    # Crea directory di output se non esiste
    output_dir = Path(output_path).parent
    if output_dir and str(output_dir) != '.':
        output_dir.mkdir(parents=True, exist_ok=True)

    # Export ONNX
    print(f"Esportazione ONNX in: {output_path}")

    input_names = ["images"]
    output_names = ["pred_boxes", "pred_logits"]

    # Assi dinamici per supportare batch size variabile
    dynamic_axes = {
        "images": {0: "batch_size"},
        "pred_boxes": {0: "batch_size"},
        "pred_logits": {0: "batch_size"},
    }

    # Usa l'exporter legacy per compatibilità con modelli complessi
    # Il nuovo exporter dynamo non supporta bene operazioni custom
    try:
        torch.onnx.export(
            pytorch_model,
            dummy_input,
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
            export_params=True,
            verbose=False,
            dynamo=False,  # Forza l'uso dell'exporter legacy
        )
    finally:
        # Ripristina le operazioni originali
        restore_ops(original_interpolate)

    print(f"ONNX model salvato: {output_path}")

    # Verifica il modello ONNX
    print("Verifica modello ONNX...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("Modello ONNX valido!")

    # Stampa info sul modello
    print("\nInfo modello ONNX:")
    print(f"  - Input: {[i.name for i in onnx_model.graph.input]}")
    print(f"  - Output: {[o.name for o in onnx_model.graph.output]}")
    print(f"  - Opset version: {onnx_model.opset_import[0].version}")

    # Semplificazione opzionale
    if simplify:
        try:
            import onnxsim
            print("\nSemplificazione modello ONNX...")
            simplified_path = output_path.replace(".onnx", "_simplified.onnx")
            onnx_model_simplified, check = onnxsim.simplify(onnx_model)
            if check:
                onnx.save(onnx_model_simplified, simplified_path)
                print(f"Modello semplificato salvato: {simplified_path}")
            else:
                print("Semplificazione fallita, usando modello originale")
        except ImportError:
            print("onnxsim non installato. Installa con: pip install onnxsim")

    return output_path


def test_onnx_inference(onnx_path: str, resolution: int = 640):
    """
    Testa l'inferenza con il modello ONNX usando ONNX Runtime.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime non installato. Installa con: pip install onnxruntime")
        return

    print(f"\nTest inferenza ONNX: {onnx_path}")

    # Crea sessione
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

    # Input dummy
    dummy_input = np.random.randn(1, 3, resolution, resolution).astype(np.float32)

    # Inferenza
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: dummy_input})

    print("Risultati inferenza ONNX:")
    for i, output in enumerate(outputs):
        output_name = session.get_outputs()[i].name
        print(f"  - {output_name}: shape={output.shape}, dtype={output.dtype}")

    print("Test inferenza completato!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Esporta RF-DETR in ONNX")
    parser.add_argument(
        "--model",
        type=str,
        default="RFDETRNano",
        choices=["RFDETRNano", "RFDETRSmall", "RFDETRMedium", "RFDETRBase", "RFDETRLarge"],
        help="Classe del modello da esportare",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Percorso ai pesi custom (opzionale)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Numero di classi del modello (obbligatorio se i pesi non sono COCO)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/model.onnx",
        help="Percorso di output per il file ONNX",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Risoluzione dell'immagine di input (default: usa quella del modello)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size per l'export",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=18,
        help="Versione ONNX opset (default: 18)",
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="Semplifica il modello ONNX",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Testa l'inferenza dopo l'export",
    )

    args = parser.parse_args()

    # Esporta
    output_path = export_rfdetr_to_onnx(
        model_class=args.model,
        weights_path=args.weights,
        num_classes=args.num_classes,
        output_path=args.output,
        resolution=args.resolution,
        batch_size=args.batch_size,
        opset_version=args.opset,
        simplify=args.simplify,
    )

    # Test opzionale
    if args.test:
        test_onnx_inference(output_path, args.resolution or 384)
