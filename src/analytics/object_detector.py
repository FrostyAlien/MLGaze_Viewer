"""Object detection plugin using RF-DETR for MLGaze Viewer."""

import io
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import supervision as sv
import rerun as rr
from rfdetr import RFDETRBase, RFDETRNano, RFDETRSmall, RFDETRMedium
from tqdm import tqdm

from src.plugin_sys.base import AnalyticsPlugin
from src.core import SessionData, BoundingBox
from src.core.data_types import DetectedObject, get_coco_class_name, get_coco_class_id, get_coco_class_color, COCO_CLASSES
from src.utils.logger import MLGazeLogger

logger = MLGazeLogger().get_logger(__name__)


class ObjectDetector(AnalyticsPlugin):
    """Object detection plugin using RF-DETR for real-time object detection.
    
    This plugin processes camera frames through RF-DETR to detect objects,
    caches the results in SessionData, and provides configurable model sizes
    for different accuracy/speed trade-offs.
    """
    
    # Available model sizes and their file names
    MODEL_FILES = {
        "nano": "rf-detr-nano.pth",
        "small": "rf-detr-small.pth", 
        "medium": "rf-detr-medium.pth",
        "base": "rf-detr-base.pth",
        "custom": None  # Will be set dynamically for custom models
    }
    
    def __init__(self, model_size: str = "base", confidence_threshold: float = 0.5, 
                 device: str = "auto", custom_model_path: Optional[str] = None,
                 custom_class_names: Optional[List[str]] = None,
                 nms_threshold: float = 0.5, target_classes: Optional[List[str]] = None,
                 preprocessing_mode: str = "center_crop"):
        """Initialize the ObjectDetector plugin.
        
        Args:
            model_size: RF-DETR model size ("nano", "small", "medium", "base", "custom")
            confidence_threshold: Minimum confidence for detections (0.0-1.0)
            device: Device to use ("auto", "cpu", "cuda", "mps")
            custom_model_path: Path to custom fine-tuned model (required if model_size="custom")
            custom_class_names: Custom class names for fine-tuned model
            nms_threshold: Non-Maximum Suppression threshold (0.0-1.0)
            target_classes: Only detect these specific classes (None = detect all)
            preprocessing_mode: Image preprocessing mode ("none", "center_crop", "padding")
        """
        super().__init__("Object Detector")
        self.model_size = model_size
        self.confidence_threshold = confidence_threshold
        self.device = self._setup_device(device)
        self.custom_model_path = custom_model_path
        self.custom_class_names = custom_class_names or []
        self.nms_threshold = nms_threshold
        self.target_classes = target_classes
        self.preprocessing_mode = preprocessing_mode
        
        # Model and processing state
        self.model = None
        self.model_class_names = {}  # Will store model's class_id -> class_name mapping
        self._processing_stats = {
            'frames_processed': 0,
            'total_detections': 0,
            'processing_time_ms': 0
        }
        
        # Validate model configuration
        self._validate_model_config()
    
    def get_dependencies(self) -> List[str]:
        """Return list of required plugin class names."""
        return []
    
    def get_optional_dependencies(self) -> List[str]:
        """Return list of optional plugin class names."""
        return []
    
    @classmethod
    def get_project_root(cls) -> Path:
        """Get the project root directory."""
        # Navigate up from src/analytics/object_detector.py to project root
        return Path(__file__).parent.parent.parent
    
    @classmethod 
    def get_models_directory(cls) -> Path:
        """Get the models directory path."""
        return cls.get_project_root() / "models" / "object_detection"
    
    @classmethod
    def get_model_path(cls, model_size: str) -> Path:
        """Get the full path to a model file.
        
        Args:
            model_size: Model size identifier
            
        Returns:
            Path to the model file
        """
        # Handle custom models separately (they don't have fixed paths)
        if model_size == "custom":
            # Return a non-existent path for custom models - they're handled differently
            return cls.get_models_directory() / "custom-model.pth"
        
        model_filename = cls.MODEL_FILES.get(model_size, cls.MODEL_FILES["base"])
        return cls.get_models_directory() / model_filename
    
    @classmethod
    def check_model_availability(cls, model_size: str) -> bool:
        """Check if a model file exists locally.
        
        Args:
            model_size: Model size to check
            
        Returns:
            True if model file exists locally, False otherwise
        """
        model_path = cls.get_model_path(model_size)
        return model_path.exists()
    
    @classmethod
    def check_model_or_downloadable(cls, model_size: str, custom_path: Optional[str] = None) -> str:
        """Check model availability including download capability.
        
        Args:
            model_size: Model size to check
            custom_path: Path to custom model (if model_size is "custom")
            
        Returns:
            "local" if model exists locally
            "downloadable" if model can be auto-downloaded by RF-DETR
            "unavailable" if neither option works
        """
        # Handle custom models
        if model_size == "custom":
            if custom_path and Path(custom_path).exists():
                return "local"
            return "unavailable"
        
        # First check if it's a valid model size
        if model_size not in cls.MODEL_FILES:
            return "unavailable"
        
        # Check if we have it locally first
        if cls.check_model_availability(model_size):
            return "local"
        
        # RF-DETR supports all these model sizes for auto-download
        return "downloadable"
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """Get list of available model sizes.
        
        Returns:
            List of model sizes that have corresponding files
        """
        available = []
        for model_size in cls.MODEL_FILES.keys():
            if cls.check_model_availability(model_size):
                available.append(model_size)
        return available
    
    def _download_model(self, model_size: str, target_path: Path) -> None:
        """Download model to our models directory.
        
        Args:
            model_size: Model size (nano, small, medium, base)
            target_path: Path where model should be saved
        """
        from rfdetr.main import HOSTED_MODELS, download_pretrain_weights
        from rfdetr.util.files import download_file
        
        model_filename = f"rf-detr-{model_size}.pth"
        
        if model_filename in HOSTED_MODELS:
            logger.info(f"Downloading {model_filename} to {target_path}")
            target_path.parent.mkdir(parents=True, exist_ok=True)
            download_file(HOSTED_MODELS[model_filename], str(target_path))
        else:
            # Fallback to RF-DETR's download mechanism
            logger.info(f"Using RF-DETR download for {model_filename}")
            downloaded_path = download_pretrain_weights(model_filename)
            if downloaded_path and Path(downloaded_path).exists():
                # Move to our directory
                Path(downloaded_path).rename(target_path)
            else:
                raise ValueError(f"Failed to download model: {model_filename}")
    
    def _validate_model_config(self) -> None:
        """Validate model configuration and setup."""
        if self.model_size == "custom":
            if not self.custom_model_path:
                raise ValueError("custom_model_path is required when model_size='custom'")
            
            custom_path = Path(self.custom_model_path)
            if not custom_path.exists():
                raise FileNotFoundError(f"Custom model not found: {self.custom_model_path}")
                
        elif self.model_size not in self.MODEL_FILES:
            logger.warning(f"Unknown model size '{self.model_size}', using 'base'")
            self.model_size = "base"
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup computing device with MPS support for Apple Silicon.
        
        Args:
            device: Device specification ("auto", "cpu", "cuda", "mps")
            
        Returns:
            torch.device object
        """
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")  # Apple Silicon GPU
            else:
                return torch.device("cpu")
        elif device == "mps":
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                logger.warning("MPS requested but not available, using CPU")
                return torch.device("cpu")
        elif device == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                logger.warning("CUDA requested but not available, using CPU")
                return torch.device("cpu")
        else:
            return torch.device("cpu")
    
    def _load_model(self) -> None:
        """Load and optimize RF-DETR model for inference."""
        if self.model is not None:
            return  # Already loaded
        
        logger.info(f"Loading RF-DETR {self.model_size} model...")
        
        try:
            # Create model instance
            self.model = self._create_model_instance()

            logger.info("Optimizing model for inference...")
            self._optimize_model_for_inference()
            
            # Store model's class names for mapping
            self.model_class_names = self.model.class_names
            logger.info(f"Loaded {len(self.model_class_names)} class mappings from model")
            
            logger.success(f"RF-DETR {self.model_size} model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load RF-DETR model: {e}")
            # Try fallback to available models
            available_models = self.get_available_models()
            if available_models and self.model_size not in available_models:
                fallback_model = available_models[0]
                logger.warning(f"Falling back to available model: {fallback_model}")
                self.model_size = fallback_model
                self._load_model()
            else:
                raise
    
    def _create_model_instance(self):
        """Create RF-DETR model instance based on size."""
        if self.model_size == "custom":
            return self._create_custom_model_instance()
        
        # Get the local model path
        model_path = self.get_model_path(self.model_size)
        
        # Ensure model exists in our directory
        if not model_path.exists():
            logger.info(f"Model not found at {model_path}, downloading...")
            self._download_model(self.model_size, model_path)
        
        # Always use explicit absolute path - never None!
        pretrain_weights = str(model_path)
        logger.info(f"Loading {self.model_size} model from: {pretrain_weights}")

        model_classes = {
            "nano": RFDETRNano,
            "small": RFDETRSmall,
            "medium": RFDETRMedium,
            "base": RFDETRBase
        }
        
        model_class = model_classes.get(self.model_size)
        if not model_class:
            raise ValueError(f"Unknown model size: {self.model_size}")
        
        try:
            model = model_class(pretrain_weights=pretrain_weights)

            if model.model is None:
                raise RuntimeError(f"Model creation failed for {self.model_size}")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to create {self.model_size} model: {e}")
            raise
    
    def _create_custom_model_instance(self):
        """Create RF-DETR instance with custom fine-tuned model."""
        try:
            # Load checkpoint to determine model architecture
            checkpoint = torch.load(self.custom_model_path, map_location='cpu', weights_only=False)
            
            # Try to infer model size from checkpoint or default to base
            model_size = "base"  # Default fallback
            if 'args' in checkpoint and hasattr(checkpoint['args'], 'encoder'):
                # Try to determine size from encoder type or hidden dimensions
                if hasattr(checkpoint['args'], 'hidden_dim'):
                    hidden_dim = checkpoint['args'].hidden_dim
                    if hidden_dim <= 128:
                        model_size = "nano"
                    elif hidden_dim <= 192:
                        model_size = "small"
                    elif hidden_dim <= 320:
                        model_size = "medium"
                    else:
                        model_size = "base"
            
            # Create base model instance
            if model_size == "nano":
                model = RFDETRNano()
            elif model_size == "small":
                model = RFDETRSmall()
            elif model_size == "medium":
                model = RFDETRMedium()
            else:
                model = RFDETRBase()
            
            # Load custom weights
            model.model.load_state_dict(checkpoint['model'], strict=False)
            
            # Set custom class names if provided
            if self.custom_class_names:
                model.class_names = self.custom_class_names
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load custom model: {e}")
            logger.warning("Falling back to base model...")
            return RFDETRBase()
    
    def _optimize_model_for_inference(self) -> None:
        """Optimize model for inference performance."""
        try:
            # Determine compilation based on device
            compile_model = self.device.type == "cuda"  # Only compile for CUDA
            
            # Choose appropriate dtype based on device
            if self.device.type == "cpu":
                dtype = torch.float32
            elif self.device.type == "mps":
                dtype = torch.float32  # MPS works better with float32
            else:  # CUDA
                dtype = torch.float16  # Use half precision for GPU

            self.model.optimize_for_inference(
                compile=compile_model,
                batch_size=1,
                dtype=dtype
            )
            
            logger.info(f"Model optimized for {self.device.type} inference")
            
        except Exception as e:
            logger.warning(f"Could not optimize model: {e}")
            logger.warning("Continuing with unoptimized model (may have higher latency)")
    
    def _preprocess_image(self, image: Image.Image) -> Tuple[Image.Image, Dict]:
        """Preprocess image based on configured mode for better detection accuracy.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Tuple of (processed_image, preprocess_info) where preprocess_info
            contains metadata needed for coordinate remapping
        """
        if self.preprocessing_mode == "none":
            return image, {"mode": "none"}
        elif self.preprocessing_mode == "center_crop":
            return self._center_crop_to_square(image)
        elif self.preprocessing_mode == "padding":
            return self._pad_to_square(image)
        else:
            return image, {"mode": "none"}
    
    def _center_crop_to_square(self, image: Image.Image) -> Tuple[Image.Image, Dict]:
        """Center-crop image to square aspect ratio.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Tuple of (cropped_image, crop_info) for coordinate remapping
        """
        width, height = image.size
        crop_size = min(width, height)
        
        # Calculate crop box for center crop
        left = (width - crop_size) // 2
        top = (height - crop_size) // 2
        right = left + crop_size
        bottom = top + crop_size
        
        # Crop to square
        cropped_image = image.crop((left, top, right, bottom))
        
        # Store crop info for coordinate remapping
        crop_info = {
            "mode": "center_crop",
            "original_size": (width, height),
            "crop_offset": (left, top),
            "crop_size": crop_size
        }
        
        return cropped_image, crop_info
    
    def _pad_to_square(self, image: Image.Image) -> Tuple[Image.Image, Dict]:
        """Pad image to square aspect ratio with black borders.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Tuple of (padded_image, padding_info) for coordinate remapping
        """
        width, height = image.size
        target_size = max(width, height)
        
        # Create square canvas with black background
        padded_image = Image.new('RGB', (target_size, target_size), (0, 0, 0))
        
        # Calculate position to center the original image
        paste_x = (target_size - width) // 2
        paste_y = (target_size - height) // 2
        
        # Paste original image onto the padded canvas
        padded_image.paste(image, (paste_x, paste_y))
        
        # Store padding info for coordinate remapping
        padding_info = {
            "mode": "padding",
            "original_size": (width, height),
            "padding_offset": (paste_x, paste_y),
            "padded_size": target_size
        }
        
        return padded_image, padding_info
    
    def _remap_detections(self, detections: List[DetectedObject], preprocess_info: Dict) -> List[DetectedObject]:
        """Remap detection coordinates from preprocessed image back to original image space.
        
        Args:
            detections: List of detections with coordinates in preprocessed image space
            preprocess_info: Preprocessing metadata from _preprocess_image()
            
        Returns:
            List of detections with coordinates mapped to original image space
        """
        if preprocess_info["mode"] == "none":
            return detections
        
        for detection in detections:
            x, y, w, h = detection.bbox.bounds
            
            # Initialize with original coordinates
            x_original, y_original = x, y
            
            if preprocess_info["mode"] == "center_crop":
                # Add crop offset to get coordinates in original image
                offset_x, offset_y = preprocess_info["crop_offset"]
                x_original = x + offset_x
                y_original = y + offset_y
                
            elif preprocess_info["mode"] == "padding":
                # Subtract padding offset to get coordinates in original image
                offset_x, offset_y = preprocess_info["padding_offset"]
                x_original = x - offset_x
                y_original = y - offset_y
            
            # Update the bounding box with remapped coordinates
            detection.bbox.bounds = np.array([x_original, y_original, w, h])
        
        return detections
    
    def process(self, session: SessionData, config: Optional[Dict[str, Any]] = None) -> Dict:
        """Process camera frames and detect objects.
        
        Args:
            session: SessionData containing camera frames
            config: Optional configuration overrides
            
        Returns:
            Dictionary containing detection results and statistics
        """
        # Update configuration if provided
        if config:
            self.confidence_threshold = config.get('confidence_threshold', self.confidence_threshold)
            self.model_size = config.get('model_size', self.model_size)
        
        # Load model if not already loaded
        self._load_model()
        
        # Reset processing stats
        self._processing_stats = {
            'frames_processed': 0,
            'total_detections': 0,
            'processing_time_ms': 0
        }
        
        results = {
            'detections_per_camera': {},
            'statistics': {},
            'model_info': {
                'model_size': self.model_size,
                'confidence_threshold': self.confidence_threshold,
                'device': str(self.device)
            }
        }
        
        # Process each camera
        for camera_name in session.get_camera_names():
            logger.info(f"Processing camera: {camera_name}")
            camera_detections = self._process_camera(session, camera_name)
            results['detections_per_camera'][camera_name] = len(camera_detections)
            
            # Store results in session
            session.add_detection_results(camera_name, camera_detections)
        
        # Generate final statistics
        results['statistics'] = self._generate_statistics(session)
        results['processing_stats'] = self._processing_stats.copy()
        
        # Include session data for visualization
        results['session_data'] = session
        
        # Store results in session for plugin system
        session.set_plugin_result("ObjectDetector", results)
        
        return results
    
    def _process_camera(self, session: SessionData, camera_name: str) -> Dict[str, List[DetectedObject]]:
        """Process all frames for a specific camera.
        
        Args:
            session: SessionData containing frames
            camera_name: Name of the camera to process
            
        Returns:
            Dictionary mapping frame_id to list of DetectedObject
        """
        camera_detections = {}
        camera_frames = session.get_frames_for_camera(camera_name)
        camera_metadata = session.get_metadata_for_camera(camera_name)
        
        if not camera_frames or camera_metadata is None:
            return camera_detections
        
        # Process frames in batches for efficiency
        batch_size = 8  # Adjust based on available memory
        frame_items = list(camera_frames.items())
        
        # Use progress bar for frame processing
        for i in tqdm(range(0, len(frame_items), batch_size), 
                      desc=f"Detecting objects in {camera_name}", 
                      unit="batch"):
            batch_items = frame_items[i:i + batch_size]
            batch_detections = self._process_frame_batch(
                batch_items, camera_metadata, camera_name
            )
            camera_detections.update(batch_detections)
        
        logger.success(f"Processed {len(camera_detections)} frames for {camera_name}")
        return camera_detections
    
    def _process_frame_batch(self, frame_items: List[Tuple[str, bytes]], 
                           camera_metadata: Any, camera_name: str) -> Dict[str, List[DetectedObject]]:
        """Process a batch of frames for efficiency.
        
        Args:
            frame_items: List of (frame_id, jpeg_bytes) tuples
            camera_metadata: Camera metadata DataFrame
            camera_name: Name of the camera
            
        Returns:
            Dictionary mapping frame_id to list of DetectedObject
        """
        import time
        batch_detections = {}
        
        for frame_id, jpeg_bytes in frame_items:
            start_time = time.time()
            
            # Get frame metadata for timestamp
            frame_metadata = camera_metadata[camera_metadata['frameId'] == frame_id]
            if frame_metadata.empty:
                continue
            
            timestamp = int(frame_metadata.iloc[0]['timestamp'])
            
            # Convert JPEG bytes to PIL Image
            try:
                image = Image.open(io.BytesIO(jpeg_bytes)).convert('RGB')
            except Exception as e:
                logger.error(f"Failed to load frame {frame_id}: {e}")
                continue
            
            # Run object detection
            detections = self._detect_objects(image, frame_id, timestamp, camera_name)
            
            # Apply filtering if specified
            if self.target_classes:
                detections = self._filter_detections_by_class(detections)
            
            batch_detections[frame_id] = detections
            
            # Update stats
            processing_time = (time.time() - start_time) * 1000
            self._processing_stats['frames_processed'] += 1
            self._processing_stats['total_detections'] += len(detections)
            self._processing_stats['processing_time_ms'] += processing_time
        
        return batch_detections
    
    def _detect_objects(self, image: Image.Image, frame_id: str, 
                       timestamp: int, camera_name: str) -> List[DetectedObject]:
        """Run object detection on a single image.
        
        Args:
            image: PIL Image to process
            frame_id: Frame identifier
            timestamp: Frame timestamp in nanoseconds
            
        Returns:
            List of DetectedObject instances
        """
        try:
            # Preprocess image to square aspect ratio for better accuracy
            processed_image, preprocess_info = self._preprocess_image(image)
            
            # Run RF-DETR inference on preprocessed image
            results = self.model.predict(processed_image, threshold=self.confidence_threshold)
            
            # Convert results to DetectedObject instances
            detections = []
            
            # RF-DETR returns supervision.Detections object
            if hasattr(results, 'xyxy') and len(results.xyxy) > 0:
                for i in range(len(results.xyxy)):
                    bbox_coords = results.xyxy[i]  # [x1, y1, x2, y2]
                    confidence = results.confidence[i] if hasattr(results, 'confidence') else 1.0
                    rfdetr_class_id = int(results.class_id[i]) if hasattr(results, 'class_id') else 0
                    
                    # Get RF-DETR class name and map to COCO index
                    class_name = self.model_class_names.get(rfdetr_class_id, "unknown")
                    coco_class_id = get_coco_class_id(class_name)
                    
                    # Skip if unknown class
                    if coco_class_id == -1:
                        logger.warning(f"Unknown class: {class_name} (RF-DETR ID: {rfdetr_class_id})")
                        continue
                    
                    # Convert to our format [x, y, width, height]
                    x1, y1, x2, y2 = bbox_coords
                    bbox_bounds = np.array([x1, y1, x2 - x1, y2 - y1])
                    
                    # Create BoundingBox
                    bbox = BoundingBox(
                        name=f"detection_{i}",
                        bounds=bbox_bounds,
                        confidence=float(confidence),
                        category="detection"
                    )
                    
                    # Create DetectedObject with COCO class ID
                    detected_obj = DetectedObject(
                        camera_name=camera_name,
                        frame_id=frame_id,
                        timestamp=timestamp,
                        bbox=bbox,
                        class_name=class_name,
                        class_id=coco_class_id,  # Use COCO index, not RF-DETR ID
                        confidence=float(confidence)
                    )
                    
                    detections.append(detected_obj)
            
            # Remap coordinates from preprocessed image back to original image space
            detections = self._remap_detections(detections, preprocess_info)
            
            # Apply Non-Maximum Suppression to reduce duplicates
            if detections and hasattr(sv, 'Detections') and len(detections) > 1:
                detections = self._apply_nms(detections)
            
            return detections
            
        except Exception as e:
            logger.error(f"Detection failed for frame {frame_id}: {e}")
            return []
    
    def _apply_nms(self, detections: List[DetectedObject]) -> List[DetectedObject]:
        """Apply Non-Maximum Suppression to reduce duplicate detections."""
        try:
            if len(detections) <= 1:
                return detections
            
            # Convert to format for NMS
            boxes = []
            scores = []
            class_ids = []
            for detection in detections:
                x, y, w, h = detection.bbox.bounds
                boxes.append([x, y, x + w, y + h])  # Convert to XYXY format
                scores.append(detection.confidence)
                class_ids.append(detection.class_id)
            
            boxes_array = np.array(boxes)
            scores_array = np.array(scores)
            class_ids_array = np.array(class_ids)
            
            # Create supervision Detections object with class_id
            sv_detections = sv.Detections(
                xyxy=boxes_array,
                confidence=scores_array,
                class_id=class_ids_array  # Include class_id for NMS
            )
            
            # Apply NMS
            nms_detections = sv_detections.with_nms(threshold=self.nms_threshold)
            
            # Convert back to our format using mask to preserve original detection objects
            filtered_detections = []
            if hasattr(nms_detections, 'mask') and nms_detections.mask is not None:
                # Use the mask to select kept detections
                for i, keep in enumerate(nms_detections.mask):
                    if keep and i < len(detections):
                        filtered_detections.append(detections[i])
            else:
                # Fallback: use indices from NMS result
                kept_indices = set(range(len(nms_detections.xyxy)))
                for i in kept_indices:
                    if i < len(detections):
                        filtered_detections.append(detections[i])
            
            return filtered_detections
            
        except Exception as e:
            logger.warning(f"NMS failed: {e}")
            # If class-aware NMS fails, try class-agnostic NMS as backup
            try:
                sv_detections = sv.Detections(
                    xyxy=boxes_array,
                    confidence=scores_array
                )
                nms_detections = sv_detections.with_nms(threshold=self.nms_threshold, class_agnostic=True)
                
                # Use mask to select kept detections
                filtered_detections = []
                if hasattr(nms_detections, 'mask') and nms_detections.mask is not None:
                    for i, keep in enumerate(nms_detections.mask):
                        if keep and i < len(detections):
                            filtered_detections.append(detections[i])
                else:
                    filtered_detections = detections[:len(nms_detections.xyxy)]
                
                return filtered_detections
                
            except Exception as e2:
                logger.warning(f"Class-agnostic NMS also failed: {e2}")
                return detections
    
    def _filter_detections_by_class(self, detections: List[DetectedObject]) -> List[DetectedObject]:
        """Filter detections to only include target classes."""
        if not self.target_classes:
            return detections
        
        filtered = []
        for detection in detections:
            if detection.class_name.lower() in [cls.lower() for cls in self.target_classes]:
                filtered.append(detection)
        
        return filtered
    
    
    def _generate_statistics(self, session: SessionData) -> Dict[str, Any]:
        """Generate detection statistics.
        
        Args:
            session: SessionData with detection results
            
        Returns:
            Dictionary with statistics
        """
        # Use SessionData's built-in statistics method
        base_stats = session.get_detection_statistics()
        
        # Add our processing-specific stats
        stats = base_stats.copy()
        stats.update({
            'model_size': self.model_size,
            'device': str(self.device),
            'frames_per_second': 0.0,
            'avg_detections_per_frame': 0.0
        })
        
        # Calculate processing rate
        if self._processing_stats['processing_time_ms'] > 0:
            total_time_s = self._processing_stats['processing_time_ms'] / 1000
            stats['frames_per_second'] = self._processing_stats['frames_processed'] / total_time_s
        
        # Calculate average detections per frame
        if self._processing_stats['frames_processed'] > 0:
            stats['avg_detections_per_frame'] = (
                self._processing_stats['total_detections'] / 
                self._processing_stats['frames_processed']
            )
        
        return stats
    
    def get_summary(self, results: Dict) -> str:
        """Generate text summary of detection results.
        
        Args:
            results: Analysis results from process method
            
        Returns:
            Human-readable summary string
        """
        lines = [
            "Object Detection Results",
            "=" * 40
        ]
        
        # Model info
        model_info = results.get('model_info', {})
        lines.extend([
            f"Model: RF-DETR {model_info.get('model_size', 'unknown')}",
            f"Device: {model_info.get('device', 'unknown')}",
            f"Confidence threshold: {model_info.get('confidence_threshold', 0.0):.2f}",
            ""
        ])
        
        # Processing stats
        proc_stats = results.get('processing_stats', {})
        lines.extend([
            f"Frames processed: {proc_stats.get('frames_processed', 0)}",
            f"Total detections: {proc_stats.get('total_detections', 0)}",
            ""
        ])
        
        # Per-camera results
        camera_results = results.get('detections_per_camera', {})
        if camera_results:
            lines.append("Detections per camera:")
            for camera, count in camera_results.items():
                lines.append(f"  {camera}: {count} frames with detections")
            lines.append("")
        
        # Overall statistics
        stats = results.get('statistics', {})
        if stats:
            lines.extend([
                f"Processing rate: {stats.get('frames_per_second', 0):.1f} FPS",
                f"Avg detections per frame: {stats.get('avg_detections_per_frame', 0):.1f}",
                f"Unique classes detected: {len(stats.get('class_distribution', {}))}"
            ])
            
            # Top detected classes
            class_dist = stats.get('class_distribution', {})
            if class_dist:
                sorted_classes = sorted(class_dist.items(), key=lambda x: x[1], reverse=True)
                lines.append("\nTop detected classes:")
                for class_name, count in sorted_classes[:5]:
                    lines.append(f"  {class_name}: {count}")
        
        return "\n".join(lines)
    
    def validate_data(self, session: SessionData) -> bool:
        """Check if session data is valid for object detection.
        
        Args:
            session: SessionData to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        if not session.frames:
            print("No camera frames available for object detection")
            return False
        
        # Check if at least one camera has frames
        total_frames = sum(len(frames) for frames in session.frames.values())
        if total_frames == 0:
            print("No frames found in any camera")
            return False
        
        # Check if metadata is available
        if not session.camera_metadata:
            print("No camera metadata available")
            return False
        
        print(f"Validation passed: {len(session.frames)} cameras, {total_frames} total frames")
        return True
    
    def visualize(self, results: Dict, rr_stream=None) -> None:
        """Visualize object detection results in Rerun.
        
        Args:
            results: Analysis results from process method
            rr_stream: Optional Rerun recording stream (unused, uses global stream)
        """
        # Get the session data with detection results
        if 'session_data' not in results:
            return
        
        session = results['session_data']
        
        # Visualize detections for each camera
        for camera_name in session.get_camera_names():
            self._visualize_camera_detections(session, camera_name)
    
    def _visualize_camera_detections(self, session: SessionData, camera_name: str) -> None:
        """Visualize object detections for a specific camera.
        
        Args:
            session: SessionData containing detection results
            camera_name: Name of the camera to visualize
        """
        camera_metadata = session.get_metadata_for_camera(camera_name)
        if camera_metadata is None or camera_metadata.empty:
            return
        
        # Use the same path where camera images are displayed
        entity_path = f"/cameras/{camera_name}/image"
        detection_count = 0
        
        # Process each frame with detections
        for idx, row in tqdm(camera_metadata.iterrows(), 
                             total=len(camera_metadata),
                             desc=f"Visualizing {camera_name} detections",
                             unit="frame"):
            frame_id = row['frameId']
            timestamp_ns = int(row['timestamp'])
            
            # Set timeline
            rr.set_time("timestamp", timestamp=timestamp_ns * 1e-9)
            
            # Get detections for this frame
            detections = session.get_detections_for_frame(camera_name, frame_id)
            if not detections:
                # CRITICAL FIX: Clear the detections entity when no objects detected
                rr.log(f"{entity_path}/detections", rr.Clear(recursive=False))
                continue
            
            # Prepare bounding boxes and annotations
            boxes = []
            class_ids = []
            labels = []
            
            for detection in detections:
                # Extract bounding box coordinates [x, y, width, height]
                x, y, w, h = detection.bbox.bounds
                
                # Convert to [x_min, y_min, x_max, y_max] format for Rerun
                boxes.append([x, y, x + w, y + h])
                class_ids.append(detection.class_id)
                labels.append(f"{detection.class_name} ({detection.confidence:.2f})")
            
            if boxes:
                # Log bounding boxes to overlay on camera image
                rr.log(
                    f"{entity_path}/detections",
                    rr.Boxes2D(
                        array=boxes,
                        array_format=rr.Box2DFormat.XYXY,  # Fix: specify array format
                        class_ids=class_ids,
                        labels=labels
                    )
                )
                
                # Create annotation context with COCO class colors (only once per camera)
                if idx == 0 or not hasattr(self, f'_logged_context_{camera_name}'):
                    annotation_infos = []
                    for class_id in range(len(COCO_CLASSES)):  # Create context for all COCO classes
                        class_name = get_coco_class_name(class_id)
                        color = get_coco_class_color(class_id)
                        annotation_infos.append(
                            rr.AnnotationInfo(
                                id=class_id,
                                label=class_name,
                                color=color
                            )
                        )
                    
                    # Log annotation context (static for this camera)
                    rr.log(
                        entity_path,
                        rr.AnnotationContext(annotation_infos),
                        static=True
                    )
                    setattr(self, f'_logged_context_{camera_name}', True)
                
                detection_count += len(detections)
    
    def get_required_columns(self) -> Dict[str, List]:
        """Get required columns for object detection.
        
        Returns:
            Dictionary mapping dataframe names to required column lists
        """
        return {
            'camera_metadata': ['frameId', 'timestamp']
        }