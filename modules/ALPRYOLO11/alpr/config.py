"""
Configuration module for the ALPR system.
Handles loading and validating configuration from environment variables and files.
"""
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from codeproject_ai_sdk import ModuleOptions

@dataclass
class ALPRConfig:
    """Configuration for the ALPR system"""
    # Paths
    app_dir: str = field(default_factory=lambda: os.path.normpath(os.getcwd()))
    models_dir: str = field(default_factory=lambda: os.path.normpath(os.path.join(os.getcwd(), "models")))
    
    # Feature flags
    enable_state_detection: bool = False
    enable_vehicle_detection: bool = False
    enable_speed_calculation: bool = True
    
    # Speed calculation parameters
    frame_rate: float = 20.0  # Camera frame rate in FPS
    plate_width_inches: float = 12.0  # US standard license plate width
    plate_height_inches: float = 6.0  # US standard license plate height
    speed_tracking_window_frames: int = 20  # Rolling window size in frames
    speed_min_tracking_frames: int = 2  # Minimum frames before calculating speed
    speed_iou_threshold: float = 0.15  # IoU threshold for matching plates (lowered for moving vehicles)
    speed_centroid_threshold: float = 5.0  # Max normalized centroid distance for matching (5× plate width)
    
    # Confidence thresholds
    plate_detector_confidence: float = 0.45
    state_classifier_confidence: float = 0.45
    char_detector_confidence: float = 0.20  # Lowered from 0.40 to improve character detection
    char_classifier_confidence: float = 0.40
    vehicle_detector_confidence: float = 0.45
    vehicle_classifier_confidence: float = 0.45
    
    # Processing parameters
    plate_aspect_ratio: Optional[float] = 4.0
    corner_dilation_pixels: int = 5
    char_box_dilation_width: int = 0
    char_box_dilation_height: int = 0
    
    # Character organization parameters
    line_separation_threshold: float = 0.6        # Multiple of avg height to consider separate lines
    vertical_aspect_ratio: float = 1.5            # Aspect ratio threshold for vertical characters
    overlap_threshold: float = 0.3                # IoU threshold for determining overlapping characters
    min_chars_for_clustering: int = 6             # Minimum chars before using advanced clustering
    height_filter_threshold: float = 0.6          # Height threshold ratio for filtering characters
    clustering_y_scale_factor: float = 3.0        # Y-coordinate scaling factor for clustering
    
    # Debug options
    save_debug_images: bool = False
    debug_images_dir: str = field(default_factory=lambda: os.path.normpath(os.path.join(os.getcwd(), "debug_images")))
    cropped_plate_save_path: str = field(default_factory=lambda: os.path.normpath(os.path.join(os.getcwd(), "alpr.jpg")))
    
    # Hardware acceleration
    use_cuda: bool = False
    use_mps: bool = False  # Apple Silicon GPU
    use_directml: bool = True  # DirectML for Windows
    device_id: Optional[int] = None  # GPU device ID (None = auto-detect)
    
    # Model format (always ONNX)
    onnx_models_dir: str = field(default_factory=lambda: os.path.normpath(os.path.join(os.getcwd(), "models")))
    
    # Derived properties
    _model_paths: Dict[str, str] = field(default_factory=dict, init=False)
    
    def __post_init__(self):
        """Validate configuration and set derived properties"""
        # Always use ONNX models
        model_ext = ".onnx"
        models_directory = self.onnx_models_dir
        
        self._model_paths = {
            "plate_detector": os.path.join(models_directory, f"plate_detector{model_ext}"),
            "state_classifier": os.path.join(models_directory, f"state_classifier{model_ext}"),
            "char_detector": os.path.join(models_directory, f"char_detector{model_ext}"),
            "char_classifier": os.path.join(models_directory, f"char_classifier{model_ext}"),
            "vehicle_detector": os.path.join(models_directory, f"vehicle_detector{model_ext}"),
            "vehicle_classifier": os.path.join(models_directory, f"vehicle_classifier{model_ext}"),
        }
        
        # Validate configuration after model paths are set
        self.validate()
        
        # Create debug images directory if needed
        if self.save_debug_images and not os.path.exists(self.debug_images_dir):
            os.makedirs(self.debug_images_dir, exist_ok=True)
    
    def validate(self) -> None:
        """Validate the configuration values"""
        # Validate confidence thresholds
        for name, value in {
            "plate_detector_confidence": self.plate_detector_confidence,
            "state_classifier_confidence": self.state_classifier_confidence,
            "char_detector_confidence": self.char_detector_confidence,
            "char_classifier_confidence": self.char_classifier_confidence,
            "vehicle_detector_confidence": self.vehicle_detector_confidence,
            "vehicle_classifier_confidence": self.vehicle_classifier_confidence
        }.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"Confidence threshold {name} must be between 0.0 and 1.0")
        
        # Validate paths
        if not os.path.exists(self.models_dir):
            raise ValueError(f"Models directory does not exist: {self.models_dir}")
        
        # Validate ONNX models directory
        if not os.path.exists(self.onnx_models_dir):
            raise ValueError(f"ONNX models directory does not exist: {self.onnx_models_dir}")
        
        # Validate plate aspect ratio
        if self.plate_aspect_ratio is not None and self.plate_aspect_ratio <= 0:
            raise ValueError(f"Plate aspect ratio must be positive, got {self.plate_aspect_ratio}")
        
        # Validate corner dilation
        if self.corner_dilation_pixels < 0:
            raise ValueError(f"Corner dilation pixels must be non-negative, got {self.corner_dilation_pixels}")
        
        # Validate character organization parameters
        if not 0.1 <= self.line_separation_threshold <= 2.0:
            raise ValueError(f"Line separation threshold must be between 0.1 and 2.0, got {self.line_separation_threshold}")
        
        if not 1.0 <= self.vertical_aspect_ratio <= 5.0:
            raise ValueError(f"Vertical aspect ratio must be between 1.0 and 5.0, got {self.vertical_aspect_ratio}")
        
        if not 0.1 <= self.overlap_threshold <= 1.0:
            raise ValueError(f"Overlap threshold must be between 0.1 and 1.0, got {self.overlap_threshold}")
        
        if self.min_chars_for_clustering < 2:
            raise ValueError(f"Minimum chars for clustering must be at least 2, got {self.min_chars_for_clustering}")
        
        if not 0.3 <= self.height_filter_threshold <= 1.0:
            raise ValueError(f"Height filter threshold must be between 0.3 and 1.0, got {self.height_filter_threshold}")
        
        if not 1.0 <= self.clustering_y_scale_factor <= 10.0:
            raise ValueError(f"Clustering Y scale factor must be between 1.0 and 10.0, got {self.clustering_y_scale_factor}")
        
        # Validate device_id if specified
        if self.device_id is not None and self.device_id < 0:
            raise ValueError(f"Device ID must be non-negative, got {self.device_id}")
        
        # Validate that required model files exist
        required_models = ["plate_detector", "char_detector", "char_classifier", "state_classifier"]
        for model_name in required_models:
            model_path = self._model_paths[model_name]
            if not os.path.exists(model_path):
                raise ValueError(f"Required model file does not exist: {model_path}")
        
        # Check optional vehicle models only if vehicle detection is enabled
        if self.enable_vehicle_detection:
            vehicle_models = ["vehicle_detector", "vehicle_classifier"]
            for model_name in vehicle_models:
                model_path = self._model_paths[model_name]
                if not os.path.exists(model_path):
                    # Don't fail validation, just disable vehicle detection
                    # Note: Logging is handled by the adapter layer
                    self.enable_vehicle_detection = False
                    break
    
    def get_model_path(self, model_name: str) -> str:
        """Get the path to a specific model"""
        if model_name not in self._model_paths:
            raise ValueError(f"Unknown model: {model_name}")
        return self._model_paths[model_name]
    
    def as_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary"""
        return {
            "paths": {
                "app_dir": self.app_dir,
                "models_dir": self.models_dir,
                "onnx_models_dir": self.onnx_models_dir,
                "debug_images_dir": self.debug_images_dir if self.save_debug_images else None,
            },
            "features": {
                "enable_state_detection": self.enable_state_detection,
                "enable_vehicle_detection": self.enable_vehicle_detection,
                "enable_speed_calculation": self.enable_speed_calculation,
                "save_debug_images": self.save_debug_images,
            },
            "confidence_thresholds": {
                "plate_detector": self.plate_detector_confidence,
                "state_classifier": self.state_classifier_confidence,
                "char_detector": self.char_detector_confidence,
                "char_classifier": self.char_classifier_confidence,
                "vehicle_detector": self.vehicle_detector_confidence,
                "vehicle_classifier": self.vehicle_classifier_confidence,
            },
            "processing": {
                "plate_aspect_ratio": self.plate_aspect_ratio,
                "corner_dilation_pixels": self.corner_dilation_pixels,
                "char_box_dilation_width": self.char_box_dilation_width,
                "char_box_dilation_height": self.char_box_dilation_height,
            },
            "speed_calculation": {
                "frame_rate": self.frame_rate,
                "plate_width_inches": self.plate_width_inches,
                "plate_height_inches": self.plate_height_inches,
                "tracking_window_frames": self.speed_tracking_window_frames,
                "min_tracking_frames": self.speed_min_tracking_frames,
                "iou_threshold": self.speed_iou_threshold,
                "centroid_threshold": self.speed_centroid_threshold,
            },
            "character_organization": {
                "line_separation_threshold": self.line_separation_threshold,
                "vertical_aspect_ratio": self.vertical_aspect_ratio,
                "overlap_threshold": self.overlap_threshold,
                "min_chars_for_clustering": self.min_chars_for_clustering,
                "height_filter_threshold": self.height_filter_threshold,
                "clustering_y_scale_factor": self.clustering_y_scale_factor,
            },
            "hardware": {
                "use_cuda": self.use_cuda,
                "use_mps": self.use_mps,
                "use_directml": self.use_directml,
                "device_id": self.device_id,
            }
        }
        
    def __str__(self) -> str:
        """Convert configuration to a string representation"""
        import json
        return json.dumps(self.as_dict(), indent=2)


def load_from_env() -> ALPRConfig:
    """Load configuration from environment variables"""
    app_dir = os.path.normpath(ModuleOptions.getEnvVariable("APPDIR", os.getcwd()))
    models_dir = os.path.normpath(ModuleOptions.getEnvVariable("MODELS_DIR", f"{app_dir}/models"))
    
    # Feature flags
    enable_state_detection = ModuleOptions.getEnvVariable("ENABLE_STATE_DETECTION", "False").lower() == "true"
    enable_vehicle_detection = ModuleOptions.getEnvVariable("ENABLE_VEHICLE_DETECTION", "False").lower() == "true"
    enable_speed_calculation = ModuleOptions.getEnvVariable("ENABLE_SPEED_CALCULATION", "True").lower() == "true"
    
    # Speed calculation parameters
    frame_rate = float(ModuleOptions.getEnvVariable("FRAME_RATE", "20.0"))
    plate_width_inches = float(ModuleOptions.getEnvVariable("PLATE_WIDTH_INCHES", "12.0"))
    plate_height_inches = float(ModuleOptions.getEnvVariable("PLATE_HEIGHT_INCHES", "6.0"))
    speed_tracking_window_frames = int(ModuleOptions.getEnvVariable("SPEED_TRACKING_WINDOW_FRAMES", "20"))
    speed_min_tracking_frames = int(ModuleOptions.getEnvVariable("SPEED_MIN_TRACKING_FRAMES", "3"))
    speed_iou_threshold = float(ModuleOptions.getEnvVariable("SPEED_IOU_THRESHOLD", "0.15"))
    speed_centroid_threshold = float(ModuleOptions.getEnvVariable("SPEED_CENTROID_THRESHOLD", "2.0"))
    
    # Confidence thresholds
    plate_detector_confidence = float(ModuleOptions.getEnvVariable("PLATE_DETECTOR_CONFIDENCE", "0.45"))
    state_classifier_confidence = float(ModuleOptions.getEnvVariable("STATE_CLASSIFIER_CONFIDENCE", "0.45"))
    char_detector_confidence = float(ModuleOptions.getEnvVariable("CHAR_DETECTOR_CONFIDENCE", "0.40"))
    char_classifier_confidence = float(ModuleOptions.getEnvVariable("CHAR_CLASSIFIER_CONFIDENCE", "0.40"))
    vehicle_detector_confidence = float(ModuleOptions.getEnvVariable("VEHICLE_DETECTOR_CONFIDENCE", "0.45"))
    vehicle_classifier_confidence = float(ModuleOptions.getEnvVariable("VEHICLE_CLASSIFIER_CONFIDENCE", "0.45"))
    
    # Processing parameters
    plate_aspect_ratio_str = ModuleOptions.getEnvVariable("PLATE_ASPECT_RATIO", "2.5")
    plate_aspect_ratio = float(plate_aspect_ratio_str) if plate_aspect_ratio_str and plate_aspect_ratio_str != "0" else None
    corner_dilation_pixels = int(ModuleOptions.getEnvVariable("CORNER_DILATION_PIXELS", "5"))
    char_box_dilation_width = int(ModuleOptions.getEnvVariable("CHAR_BOX_DILATION_WIDTH", "0"))
    char_box_dilation_height = int(ModuleOptions.getEnvVariable("CHAR_BOX_DILATION_HEIGHT", "0"))
    
    # Character organization parameters
    line_separation_threshold = float(ModuleOptions.getEnvVariable("LINE_SEPARATION_THRESHOLD", "0.6"))
    vertical_aspect_ratio = float(ModuleOptions.getEnvVariable("VERTICAL_ASPECT_RATIO", "1.5"))
    overlap_threshold = float(ModuleOptions.getEnvVariable("OVERLAP_THRESHOLD", "0.3"))
    min_chars_for_clustering = int(ModuleOptions.getEnvVariable("MIN_CHARS_FOR_CLUSTERING", "6"))
    height_filter_threshold = float(ModuleOptions.getEnvVariable("HEIGHT_FILTER_THRESHOLD", "0.6"))
    clustering_y_scale_factor = float(ModuleOptions.getEnvVariable("CLUSTERING_Y_SCALE_FACTOR", "3.0"))
    
    # Debug options
    save_debug_images = ModuleOptions.getEnvVariable("SAVE_DEBUG_IMAGES", "False").lower() == "true"
    debug_images_dir = os.path.normpath(ModuleOptions.getEnvVariable("DEBUG_IMAGES_DIR", f"{app_dir}/debug_images"))
    cropped_plate_save_path = os.path.normpath(ModuleOptions.getEnvVariable("CROPPED_PLATE_SAVE_PATH", f"{app_dir}/alpr.jpg"))
    
    # Hardware acceleration
    use_cuda = ModuleOptions.getEnvVariable("USE_CUDA", "False").lower() == "true"
    use_mps = False  # Default to false, will be checked for availability later
    use_directml = ModuleOptions.getEnvVariable("USE_DIRECTML", "True").lower() == "true"
    device_id_str = ModuleOptions.getEnvVariable("DEVICE_ID", "None")
    device_id = int(device_id_str) if device_id_str and device_id_str.isdigit() else None
    
    # Model format (always ONNX)
    onnx_models_dir = os.path.normpath(ModuleOptions.getEnvVariable("ONNX_MODELS_DIR", f"{app_dir}/models"))
    
    return ALPRConfig(
        app_dir=app_dir,
        models_dir=models_dir,
        enable_state_detection=enable_state_detection,
        enable_vehicle_detection=enable_vehicle_detection,
        enable_speed_calculation=enable_speed_calculation,
        frame_rate=frame_rate,
        plate_width_inches=plate_width_inches,
        plate_height_inches=plate_height_inches,
        speed_tracking_window_frames=speed_tracking_window_frames,
        speed_min_tracking_frames=speed_min_tracking_frames,
        speed_iou_threshold=speed_iou_threshold,
        speed_centroid_threshold=speed_centroid_threshold,
        plate_detector_confidence=plate_detector_confidence,
        state_classifier_confidence=state_classifier_confidence,
        char_detector_confidence=char_detector_confidence,
        char_classifier_confidence=char_classifier_confidence,
        vehicle_detector_confidence=vehicle_detector_confidence,
        vehicle_classifier_confidence=vehicle_classifier_confidence,
        plate_aspect_ratio=plate_aspect_ratio,
        corner_dilation_pixels=corner_dilation_pixels,
        char_box_dilation_width=char_box_dilation_width,
        char_box_dilation_height=char_box_dilation_height,
        line_separation_threshold=line_separation_threshold,
        vertical_aspect_ratio=vertical_aspect_ratio,
        overlap_threshold=overlap_threshold,
        min_chars_for_clustering=min_chars_for_clustering,
        height_filter_threshold=height_filter_threshold,
        clustering_y_scale_factor=clustering_y_scale_factor,
        save_debug_images=save_debug_images,
        debug_images_dir=debug_images_dir,
        cropped_plate_save_path=cropped_plate_save_path,
        use_cuda=use_cuda,
        use_mps=use_mps,
        use_directml=use_directml,
        device_id=device_id,
        onnx_models_dir=onnx_models_dir
    )
