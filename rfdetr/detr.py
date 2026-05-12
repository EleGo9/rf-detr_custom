# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------


import json
import os
from collections import defaultdict
from logging import getLogger
from typing import Union, List
from copy import deepcopy

import numpy as np
import supervision as sv
import torch
import torchvision.transforms.functional as F
from PIL import Image
import cv2

try:
    torch.set_float32_matmul_precision('high')
except:
    pass

from rfdetr.config import (
    RFDETRBaseConfig,
    RFDETRLargeConfig,
    RFDETRNanoConfig,
    RFDETRSmallConfig,
    RFDETRMediumConfig,
    RFDETRSegPreviewConfig,
    TrainConfig,
    SegmentationTrainConfig,
    ModelConfig
)
from rfdetr.main import Model, download_pretrain_weights
from rfdetr.util.metrics import MetricsPlotSink, MetricsTensorBoardSink, MetricsWandBSink
from rfdetr.util.coco_classes import COCO_CLASSES

logger = getLogger(__name__)
class RFDETR:
    """
    The base RF-DETR class implements the core methods for training RF-DETR models,
    running inference on the models, optimising models, and uploading trained
    models for deployment.
    """
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    
    # means = [0, 0, 0]
    # stds = [1, 1, 1]
    
    size = None

    def __init__(self, **kwargs):
        self.model_config = self.get_model_config(**kwargs)
        self.maybe_download_pretrain_weights()
        self.model = self.get_model(self.model_config)
        self.callbacks = defaultdict(list)

        self.model.inference_model = None
        self._is_optimized_for_inference = False
        self._has_warned_about_not_being_optimized_for_inference = False
        self._optimized_has_been_compiled = False
        self._optimized_batch_size = None
        self._optimized_resolution = None
        self._optimized_dtype = None
        
        self.last_prediction = None
        self.last_input_tensor = None

    def maybe_download_pretrain_weights(self):
        """
        Download pre-trained weights if they are not already downloaded.
        """
        download_pretrain_weights(self.model_config.pretrain_weights)

    def get_model_config(self, **kwargs):
        """
        Retrieve the configuration parameters used by the model.
        """
        return ModelConfig(**kwargs)

    def train(self, **kwargs):
        """
        Train an RF-DETR model.
        """
        config = self.get_train_config(**kwargs)
        self.train_from_config(config, **kwargs)
    
    def optimize_for_inference(self, compile=True, batch_size=1, dtype=torch.float32):
        self.remove_optimized_model()

        self.model.inference_model = deepcopy(self.model.model)
        self.model.inference_model.eval()
        self.model.inference_model.export()

        self._optimized_resolution = self.model.resolution
        self._is_optimized_for_inference = True

        self.model.inference_model = self.model.inference_model.to(dtype=dtype)
        self._optimized_dtype = dtype

        if compile:
            self.model.inference_model = torch.jit.trace(
                self.model.inference_model,
                torch.randn(
                    batch_size, 3, self.model.resolution, self.model.resolution, 
                    device=self.model.device,
                    dtype=dtype
                )
            )
            self._optimized_has_been_compiled = True
            self._optimized_batch_size = batch_size
    
    def remove_optimized_model(self):
        self.model.inference_model = None
        self._is_optimized_for_inference = False
        self._optimized_has_been_compiled = False
        self._optimized_batch_size = None
        self._optimized_resolution = None
        self._optimized_half = False
    
    def export(self, **kwargs):
        """
        Export your model to an ONNX file.

        See [the ONNX export documentation](https://rfdetr.roboflow.com/learn/train/#onnx-export) for more information.
        """
        self.model.export(**kwargs)

    def train_from_config(self, config: TrainConfig, **kwargs):
        if config.dataset_file == "roboflow":
            with open(
                os.path.join(config.dataset_dir, "train", "_annotations.coco.json"), "r"
            ) as f:
                anns = json.load(f)
                num_classes = len(anns["categories"])
                class_names = [c["name"] for c in anns["categories"] if c["supercategory"] != "none"]
                self.model.class_names = class_names
        elif config.dataset_file == "coco":
            class_names = COCO_CLASSES
            num_classes = 90
        else:
            raise ValueError(f"Invalid dataset file: {config.dataset_file}")

        if self.model_config.num_classes != num_classes:
            self.model.reinitialize_detection_head(num_classes)
        
        train_config = config.dict()
        model_config = self.model_config.dict()
        model_config.pop("num_classes")
        if "class_names" in model_config:
            model_config.pop("class_names")
        
        if "class_names" in train_config and train_config["class_names"] is None:
            train_config["class_names"] = class_names

        for k, v in train_config.items():
            if k in model_config:
                model_config.pop(k)
            if k in kwargs:
                kwargs.pop(k)
        
        all_kwargs = {**model_config, **train_config, **kwargs, "num_classes": num_classes}

        metrics_plot_sink = MetricsPlotSink(output_dir=config.output_dir)
        self.callbacks["on_fit_epoch_end"].append(metrics_plot_sink.update)
        self.callbacks["on_train_end"].append(metrics_plot_sink.save)

        if config.tensorboard:
            metrics_tensor_board_sink = MetricsTensorBoardSink(output_dir=config.output_dir)
            self.callbacks["on_fit_epoch_end"].append(metrics_tensor_board_sink.update)
            self.callbacks["on_train_end"].append(metrics_tensor_board_sink.close)

        if config.wandb:
            metrics_wandb_sink = MetricsWandBSink(
                output_dir=config.output_dir,
                project=config.project,
                run=config.run,
                config=config.model_dump()
            )
            self.callbacks["on_fit_epoch_end"].append(metrics_wandb_sink.update)
            self.callbacks["on_train_end"].append(metrics_wandb_sink.close)

        if config.early_stopping:
            from rfdetr.util.early_stopping import EarlyStoppingCallback
            early_stopping_callback = EarlyStoppingCallback(
                model=self.model,
                patience=config.early_stopping_patience,
                min_delta=config.early_stopping_min_delta,
                use_ema=config.early_stopping_use_ema,
                segmentation_head=config.segmentation_head
            )
            self.callbacks["on_fit_epoch_end"].append(early_stopping_callback.update)

        self.model.train(
            **all_kwargs,
            callbacks=self.callbacks,
        )

    def get_train_config(self, **kwargs):
        """
        Retrieve the configuration parameters that will be used for training.
        """
        return TrainConfig(**kwargs)

    def get_model(self, config: ModelConfig):
        """
        Retrieve a model instance based on the provided configuration.
        """
        return Model(**config.dict())
    
    # Get class_names from the model
    @property
    def class_names(self):
        """
        Retrieve the class names supported by the loaded model.

        Returns:
            dict: A dictionary mapping class IDs to class names. The keys are integers starting from
        """
        if hasattr(self.model, 'class_names') and self.model.class_names:
            return {i+1: name for i, name in enumerate(self.model.class_names)}
            
        return COCO_CLASSES

    def get_last_raw_results(self):
        """
        Stampa e restituisce i logit e i box salvati nell'ultima chiamata a predict().
        """
        if self.last_prediction is None:
            print("Nessuna predizione trovata. Esegui prima model.predict().")
            return None, None

        logits = self.last_prediction["pred_logits"]
        boxes = self.last_prediction["pred_boxes"]
        
        print(f"logits shape: {logits.shape}")
        print(f"boxes shape: {boxes.shape}")

        
        logits_cpu = logits.cpu().to(torch.float32).contiguous()
        logits_cpu.numpy().tofile("logits_dump.bin")
        
        # Prepara e salva i boxes
        boxes_cpu = boxes.cpu().to(torch.float32).contiguous()
        boxes_cpu.numpy().tofile("boxes_dump.bin")

        # Imposta la stampa completa
        torch.set_printoptions(threshold=float('inf'))
        
        # print("\n=== LOGITS ===")
        # print(logits)
        # print(f"Shape: {logits.shape}")
        
        # print("\n=== BOX ===")
        # print(boxes)
        # print(f"Shape: {boxes.shape}")
        
        # Ripristina la stampa standard
        # torch.set_printoptions(profile="default")

        return logits, boxes

    def get_last_input_tensor(self, save_to_bin=True):
        """
        Stampa e restituisce il tensore dell'immagine preprocessata usato nell'ultima predict().
        Se save_to_bin è True, lo salva in formato binario raw per C++.
        """
        if getattr(self, 'last_input_tensor', None) is None:
            print("Nessun tensore di input trovato. Esegui prima model.predict().")
            return None

        # Questo è il tensore finale che entra nella rete: shape (batch_size, 3, H, W)
        input_tensor = self.last_input_tensor

        print("\n=== INPUT TENSOR ===")
        print(f"Shape: {input_tensor.shape}")
        print(f"Min: {input_tensor.min().item():.4f}")
        print(f"Max: {input_tensor.max().item():.4f}")
        print(f"Mean: {input_tensor.mean().item():.4f}")


        tensor_cpu = input_tensor.cpu().to(torch.float32).contiguous()
        filename = "input_tensor_dump.bin"
        tensor_cpu.numpy().tofile(filename)


        # torch.set_printoptions(threshold=10_000_000)

        # print("\n--- INPUT IMAGE ---")
        # print(input_tensor.cpu()) 

        # Ripristina la stampa standard di PyTorch
        torch.set_printoptions(profile="default")

        return input_tensor
    
    def predict(
        self,
        images: Union[str, Image.Image, np.ndarray, torch.Tensor, List[Union[str, np.ndarray, Image.Image, torch.Tensor]]],
        threshold: float = 0.5,
        **kwargs,
    ) -> Union[sv.Detections, List[sv.Detections]]:
        """Performs object detection on the input images and returns bounding box
        predictions.

        This method accepts a single image or a list of images in various formats
        (file path, PIL Image, NumPy array, or torch.Tensor). The images should be in
        RGB channel order. If a torch.Tensor is provided, it must already be normalized
        to values in the [0, 1] range and have the shape (C, H, W).

        Args:
            images (Union[str, Image.Image, np.ndarray, torch.Tensor, List[Union[str, np.ndarray, Image.Image, torch.Tensor]]]):
                A single image or a list of images to process. Images can be provided
                as file paths, PIL Images, NumPy arrays, or torch.Tensors.
            threshold (float, optional):
                The minimum confidence score needed to consider a detected bounding box valid.
            **kwargs:
                Additional keyword arguments.

        Returns:
            Union[sv.Detections, List[sv.Detections]]: A single or multiple Detections
                objects, each containing bounding box coordinates, confidence scores,
                and class IDs.
        """
        
        if not self._is_optimized_for_inference and not self._has_warned_about_not_being_optimized_for_inference:
            logger.warning(
                "Model is not optimized for inference. "
                "Latency may be higher than expected. "
                "You can optimize the model for inference by calling model.optimize_for_inference()."
            )
            self._has_warned_about_not_being_optimized_for_inference = True

            self.model.model.eval()

        if not isinstance(images, list):
            images = [images]

        orig_sizes = []
        processed_images = []

        for img in images:
            
            if isinstance(img, str):
                img = Image.open(img).convert("RGB")

            if isinstance(img, Image.Image):
                img = np.array(img)

            if isinstance(img, np.ndarray):
                h, w = img.shape[:2]
                orig_sizes.append((h, w))
                
                if img.shape[2] != 3:
                    raise ValueError(
                        f"Invalid image shape. Expected 3 channels (RGB), but got "
                        f"{img.shape[2]} channels."
                    )

                # Float conversion
                if img.dtype == np.uint8:
                    img_array = img / 255.0
                else:
                    img_array = img.astype(np.float32)

                # OpenCV Resize
                res = self.model.resolution
                img_array = cv2.resize(img_array, (res, res), interpolation=cv2.INTER_LINEAR)

                # NumPy Normalization
                mean = np.array(self.means, dtype=np.float32)
                std = np.array(self.stds, dtype=np.float32)
                img_array = ((img_array - mean) / std).astype(np.float32)

                img_array = np.transpose(img_array, (2, 0, 1))
                img_tensor = torch.from_numpy(img_array).to(self.model.device)

            elif isinstance(img, torch.Tensor):
                if (img > 1).any():
                    raise ValueError(
                        "Image has pixel values above 1. Please ensure the image is "
                        "normalized (scaled to [0, 1])."
                    )
                if img.shape[0] != 3:
                    raise ValueError(
                        f"Invalid image shape. Expected 3 channels (RGB), but got "
                        f"{img.shape[0]} channels."
                    )
                
                h, w = img.shape[1:]
                orig_sizes.append((h, w))

                img_array = img.cpu().numpy()
                img_array = np.transpose(img_array, (1, 2, 0))  # CHW -> HWC
                
                res = self.model.resolution
                img_array = cv2.resize(img_array, (res, res), interpolation=cv2.INTER_LINEAR)

                mean = np.array(self.means, dtype=np.float32)
                std = np.array(self.stds, dtype=np.float32)
                img_array = ((img_array - mean) / std).astype(np.float32)

                img_array = np.transpose(img_array, (2, 0, 1))
                img_tensor = torch.from_numpy(img_array).to(self.model.device)
                
            else:
                raise TypeError(f"Unsupported image type: {type(img)}")

            processed_images.append(img_tensor)

        batch_tensor = torch.stack(processed_images)
        self.last_input_tensor = batch_tensor

        if self._is_optimized_for_inference:
            if self._optimized_resolution != batch_tensor.shape[2]:
                raise ValueError(f"Resolution mismatch. "
                                 f"Model was optimized for resolution {self._optimized_resolution}, "
                                 f"but got {batch_tensor.shape[2]}. "
                                 "You can explicitly remove the optimized model by calling model.remove_optimized_model().")
            if self._optimized_has_been_compiled:
                if self._optimized_batch_size != batch_tensor.shape[0]:
                    raise ValueError(f"Batch size mismatch. "
                                     f"Optimized model was compiled for batch size {self._optimized_batch_size}, "
                                     f"but got {batch_tensor.shape[0]}. "
                                     "You can explicitly remove the optimized model by calling model.remove_optimized_model(). "
                                     "Alternatively, you can recompile the optimized model for a different batch size "
                                     "by calling model.optimize_for_inference(batch_size=<new_batch_size>).")

        with torch.inference_mode():
            if self._is_optimized_for_inference:
                predictions = self.model.inference_model(batch_tensor.to(dtype=self._optimized_dtype))
            else:
                predictions = self.model.model(batch_tensor)
            if isinstance(predictions, tuple):
                predictions = {
                    "pred_logits": predictions[1],
                    "pred_boxes": predictions[0],
                }

                if len(predictions) == 3:
                    predictions["pred_masks"] = predictions[2]
                    
            self.last_prediction = predictions
            
            target_sizes = torch.tensor(orig_sizes, device=self.model.device)
            results = self.model.postprocess(predictions, target_sizes=target_sizes)

        detections_list = []
        for result in results:
            scores = result["scores"]
            labels = result["labels"]
            boxes = result["boxes"]

            keep = scores > threshold
            scores = scores[keep]
            labels = labels[keep]
            boxes = boxes[keep]

            if "masks" in result:
                masks = result["masks"]
                masks = masks[keep]

                detections = sv.Detections(
                    xyxy=boxes.float().cpu().numpy(),
                    confidence=scores.float().cpu().numpy(),
                    class_id=labels.cpu().numpy(),
                    mask=masks.squeeze(1).cpu().numpy(),
                )
            else:
                detections = sv.Detections(
                    xyxy=boxes.float().cpu().numpy(),
                    confidence=scores.float().cpu().numpy(),
                    class_id=labels.cpu().numpy(),
                )

            detections_list.append(detections)

        return detections_list if len(detections_list) > 1 else detections_list[0]
        
    def deploy_to_roboflow(self, workspace: str, project_id: str, version: str, api_key: str = None, size: str = None):
        """
        Deploy the trained RF-DETR model to Roboflow.

        Deploying with Roboflow will create a Serverless API to which you can make requests.

        You can also download weights into a Roboflow Inference deployment for use in Roboflow Workflows and on-device deployment.

        Args:
            workspace (str): The name of the Roboflow workspace to deploy to.
            project_ids (List[str]): A list of project IDs to which the model will be deployed
            api_key (str, optional): Your Roboflow API key. If not provided,
                it will be read from the environment variable `ROBOFLOW_API_KEY`.
            size (str, optional): The size of the model to deploy. If not provided,
                it will default to the size of the model being trained (e.g., "rfdetr-base", "rfdetr-large", etc.).
            model_name (str, optional): The name you want to give the uploaded model.
            If not provided, it will default to "<size>-uploaded".
        Raises:
            ValueError: If the `api_key` is not provided and not found in the environment
                variable `ROBOFLOW_API_KEY`, or if the `size` is not set for custom architectures.
        """
        from roboflow import Roboflow
        import shutil
        if api_key is None:
            api_key = os.getenv("ROBOFLOW_API_KEY")
            if api_key is None:
                raise ValueError("Set api_key=<KEY> in deploy_to_roboflow or export ROBOFLOW_API_KEY=<KEY>")


        rf = Roboflow(api_key=api_key)
        workspace = rf.workspace(workspace)

        if self.size is None and size is None:
            raise ValueError("Must set size for custom architectures")

        size = self.size or size
        tmp_out_dir = ".roboflow_temp_upload"
        os.makedirs(tmp_out_dir, exist_ok=True)
        outpath = os.path.join(tmp_out_dir, "weights.pt")
        torch.save(
            {
                "model": self.model.model.state_dict(),
                "args": self.model.args
            }, outpath
        )
        project = workspace.project(project_id)
        version = project.version(version)
        version.deploy(
            model_type=size,
            model_path=tmp_out_dir,
            filename="weights.pt"
        )
        shutil.rmtree(tmp_out_dir)



class RFDETRBase(RFDETR):
    """
    Train an RF-DETR Base model (29M parameters).
    """
    size = "rfdetr-base"
    def get_model_config(self, **kwargs):
        return RFDETRBaseConfig(**kwargs)

    def get_train_config(self, **kwargs):
        return TrainConfig(**kwargs)

class RFDETRLarge(RFDETR):
    """
    Train an RF-DETR Large model.
    """
    size = "rfdetr-large"
    def get_model_config(self, **kwargs):
        return RFDETRLargeConfig(**kwargs)

    def get_train_config(self, **kwargs):
        return TrainConfig(**kwargs)

class RFDETRNano(RFDETR):
    """
    Train an RF-DETR Nano model.
    """
    size = "rfdetr-nano"
    def get_model_config(self, **kwargs):
        return RFDETRNanoConfig(**kwargs)

    def get_train_config(self, **kwargs):
        return TrainConfig(**kwargs)

class RFDETRSmall(RFDETR):
    """
    Train an RF-DETR Small model.
    """
    size = "rfdetr-small"
    def get_model_config(self, **kwargs):
        return RFDETRSmallConfig(**kwargs)

    def get_train_config(self, **kwargs):
        return TrainConfig(**kwargs)

class RFDETRMedium(RFDETR):
    """
    Train an RF-DETR Medium model.
    """
    size = "rfdetr-medium"
    def get_model_config(self, **kwargs):
        return RFDETRMediumConfig(**kwargs)

    def get_train_config(self, **kwargs):
        return TrainConfig(**kwargs)

class RFDETRSegPreview(RFDETR):
    size = "rfdetr-seg-preview"
    def get_model_config(self, **kwargs):
        return RFDETRSegPreviewConfig(**kwargs)

    def get_train_config(self, **kwargs):
        return SegmentationTrainConfig(**kwargs)
