"""
Script per esportare RF-DETR in ONNX.
Supporta tutti i modelli: Nano, Small, Medium, Base, Large (+ varianti Seg).

Dalla 1.10.1 in poi rfdetr espone un exporter nativo (RFDETR.export()) che
gestisce già l'asse di batch dinamico, l'opset e le patch di compatibilità
che prima applicavamo a mano qui (bicubic->bilinear, onnx.helper.float32_to_bfloat16):
questo script è ora solo un sottile wrapper CLI attorno a quello.
"""

import os
from pathlib import Path

import numpy as np
import onnx


def export_rfdetr_to_onnx(
    model_class: str = "RFDETRNano",
    weights_path: str = None,
    num_classes: int = None,
    output_path: str = "output/model.onnx",
    resolution: int = None,  # Se None, usa la risoluzione di default del modello
    batch_size: int = 1,
    dynamic_batch: bool = True,
    opset_version: int = 17,
    simplify: bool = False,
    device: str = "cpu",
):
    """
    Esporta un modello RF-DETR in formato ONNX usando l'exporter nativo di rfdetr.

    Args:
        model_class: Classe del modello ("RFDETRNano", "RFDETRSmall", "RFDETRMedium",
            "RFDETRBase", "RFDETRLarge", "RFDETRSegNano", "RFDETRSegSmall")
        weights_path: Percorso ai pesi del modello (opzionale, usa pesi pretrained se None)
        num_classes: Numero di classi del modello (obbligatorio se i pesi non sono COCO)
        output_path: Percorso di output per il file ONNX
        resolution: Risoluzione dell'immagine di input (None = usa default del modello)
        batch_size: Batch size statico da usare per il test/tracciamento dell'export
        dynamic_batch: Se True (default), l'asse batch del grafo ONNX è dinamico
        opset_version: Versione ONNX opset (default: 17)
        simplify: Se True, semplifica ulteriormente il modello ONNX con onnxsim
        device: Device per l'export ("cpu" o "cuda")
    """
    from rfdetr import (
        RFDETRBase,
        RFDETRLarge,
        RFDETRMedium,
        RFDETRNano,
        RFDETRSegNano,
        RFDETRSegSmall,
        RFDETRSmall,
    )

    model_classes = {
        "RFDETRNano": RFDETRNano,
        "RFDETRSmall": RFDETRSmall,
        "RFDETRMedium": RFDETRMedium,
        "RFDETRBase": RFDETRBase,
        "RFDETRLarge": RFDETRLarge,
        "RFDETRSegNano": RFDETRSegNano,
        "RFDETRSegSmall": RFDETRSegSmall,
    }

    kwargs = {"device": device}
    if weights_path:
        kwargs["pretrain_weights"] = weights_path
    if num_classes:
        kwargs["num_classes"] = num_classes
    model = model_classes[model_class](**kwargs)

    shape = (resolution, resolution) if resolution is not None else None

    output_dir = Path(output_path).parent
    if output_dir and str(output_dir) != ".":
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Esportazione ONNX (dynamic_batch={dynamic_batch}) in: {output_dir}")
    exported_path = model.export(
        output_dir=str(output_dir),
        shape=shape,
        batch_size=batch_size,
        dynamic_batch=dynamic_batch,
        opset_version=opset_version,
        format="onnx",
    )
    exported_path = Path(exported_path)
    if str(exported_path) != output_path:
        exported_path = exported_path.rename(output_path)
    print(f"ONNX model salvato: {exported_path}")

    print("Verifica modello ONNX...")
    onnx_model = onnx.load(str(exported_path))
    onnx.checker.check_model(onnx_model)
    print("Modello ONNX valido!")

    print("\nInfo modello ONNX:")
    print(f"  - Input: {[(i.name, [d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim]) for i in onnx_model.graph.input]}")
    print(f"  - Output: {[o.name for o in onnx_model.graph.output]}")
    print(f"  - Opset version: {onnx_model.opset_import[0].version}")

    if simplify:
        try:
            import onnxsim

            print("\nSemplificazione modello ONNX...")
            simplified_path = str(exported_path).replace(".onnx", "_simplified.onnx")
            onnx_model_simplified, check = onnxsim.simplify(onnx_model)
            if check:
                onnx.save(onnx_model_simplified, simplified_path)
                print(f"Modello semplificato salvato: {simplified_path}")
            else:
                print("Semplificazione fallita, usando modello originale")
        except ImportError:
            print("onnxsim non installato. Installa con: pip install onnxsim")

    return str(exported_path)


def test_onnx_inference(onnx_path: str, resolution: int = 384, batch_sizes=(1,)):
    """
    Testa l'inferenza con il modello ONNX usando ONNX Runtime, a una o più batch size
    (utile per verificare che l'asse batch dinamico funzioni davvero).
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime non installato. Installa con: pip install onnxruntime")
        return

    print(f"\nTest inferenza ONNX: {onnx_path}")
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    for batch_size in batch_sizes:
        dummy_input = np.random.randn(batch_size, 3, resolution, resolution).astype(np.float32)
        outputs = session.run(None, {input_name: dummy_input})
        print(f"batch_size={batch_size}:")
        for output, out_info in zip(outputs, session.get_outputs()):
            print(f"  - {out_info.name}: shape={output.shape}, dtype={output.dtype}")

    print("Test inferenza completato!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Esporta RF-DETR in ONNX")
    parser.add_argument(
        "--model",
        type=str,
        default="RFDETRNano",
        choices=["RFDETRNano", "RFDETRSmall", "RFDETRMedium", "RFDETRBase", "RFDETRLarge", "RFDETRSegNano", "RFDETRSegSmall"],
        help="Classe del modello da esportare",
    )
    parser.add_argument("--weights", type=str, default=None, help="Percorso ai pesi custom (opzionale)")
    parser.add_argument("--num-classes", type=int, default=None, help="Numero di classi del modello")
    parser.add_argument("--output", type=str, default="output/model.onnx", help="Percorso di output per il file ONNX")
    parser.add_argument("--resolution", type=int, default=None, help="Risoluzione dell'immagine di input")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size per il tracciamento dell'export")
    parser.add_argument("--static-batch", action="store_true", help="Disabilita l'asse batch dinamico (default: dinamico)")
    parser.add_argument("--opset", type=int, default=17, help="Versione ONNX opset (default: 17)")
    parser.add_argument("--simplify", action="store_true", help="Semplifica il modello ONNX con onnxsim")
    parser.add_argument("--test", action="store_true", help="Testa l'inferenza dopo l'export")

    args = parser.parse_args()

    output_path = export_rfdetr_to_onnx(
        model_class=args.model,
        weights_path=args.weights,
        num_classes=args.num_classes,
        output_path=args.output,
        resolution=args.resolution,
        batch_size=args.batch_size,
        dynamic_batch=not args.static_batch,
        opset_version=args.opset,
        simplify=args.simplify,
    )

    if args.test:
        test_batch_sizes = (1, 2, 3) if not args.static_batch else (args.batch_size,)
        test_onnx_inference(output_path, args.resolution or 384, batch_sizes=test_batch_sizes)
