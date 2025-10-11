"""
Base classes for YOLO models using ONNX runtime.
"""
import os
import numpy as np
import cv2
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

# Import the session manager
from .session_manager import get_session_manager, SessionConfig, ONNX_AVAILABLE as SM_ONNX_AVAILABLE
class YOLOBase:
    """
    Base class for YOLO models using ONNX runtime.
    Uses a centralized session manager for ONNX inference sessions.
    """

    def __init__(self, model_path: str, task: str, use_onnx: bool = True, use_cuda: bool = True):
        """
        Initialize a YOLO model using ONNX runtime.

        Args:
            model_path: Path to the ONNX model file
            task: Task type ('detect', 'classify', 'pose')
            use_onnx: Deprecated parameter, always True (kept for compatibility)
            use_cuda: Whether to use GPU acceleration (DirectML on Windows, CPU fallback otherwise)
        """
        self.model_path = model_path
        self.task = task
        self.use_onnx = True  # Always use ONNX
        self.session_id = None
        self.session_manager = None  # Reference to session manager
        self.names = {}  # Class names dictionary

        # For ONNX runtime metadata (managed by session manager)
        self.input_name = None
        self.output_names = None
        self.input_shape = None

        # Validate model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Initialize ONNX model
        self._init_onnx_model(use_cuda)

        # Load class names
        self._load_class_names()

    def _init_onnx_model(self, use_cuda: bool):
        """Initialize ONNX runtime session using the centralized session manager"""
        if not SM_ONNX_AVAILABLE:
            raise ImportError("ONNX Runtime is not available. Please install it with 'pip install onnxruntime' or 'pip install onnxruntime-directml'")

        try:
            # Get the global session manager
            self.session_manager = get_session_manager()
            
            # Create session configuration
            session_config = SessionConfig(
                model_path=self.model_path,
                use_cuda=use_cuda,
                use_directml=True  # Enable DirectML support
            )
            
            # Create session through the manager
            self.session_id = self.session_manager.create_session(session_config)
            
            # Get session metadata
            metadata = self.session_manager.get_session_metadata(self.session_id)
            self.input_name = metadata['input_name']
            self.output_names = metadata['output_names']
            self.input_shape = metadata['input_shape']
            
            print(f"Initialized ONNX model {self.model_path} with session manager")
            print(f"Session providers: {metadata['providers']}")

        except Exception as e:
            raise RuntimeError(f"Failed to initialize ONNX model {self.model_path}: {e}")

    def _safe_onnx_inference(self, input_tensor: np.ndarray, max_retries: int = 1):
        """
        Perform ONNX inference using the session manager.
        The session manager handles DirectML fallback automatically.

        Args:
            input_tensor: Preprocessed input tensor
            max_retries: Deprecated - session manager handles retries internally

        Returns:
            ONNX model outputs

        Raises:
            RuntimeError: If inference fails
        """
        if not self.session_manager or not self.session_id:
            raise RuntimeError("ONNX session not properly initialized")

        try:
            # Prepare input data
            input_data = {self.input_name: input_tensor}

            # Run inference through session manager
            outputs = self.session_manager.run_inference(self.session_id, input_data)
            return outputs

        except Exception as e:
            raise RuntimeError(f"ONNX inference failed for {self.model_path}: {str(e)}") from e

    def _load_class_names(self):
        """Load class names from JSON file with proper error handling"""
        class_file = os.path.splitext(self.model_path)[0] + '.json'
        if os.path.exists(class_file):
            try:
                import json
                with open(class_file, 'r') as f:
                    data = json.load(f)

                # Handle different JSON formats
                if isinstance(data, dict):
                    # If data is a dict, use it directly or extract 'names' key
                    if 'names' in data:
                        self.names = data['names']
                    else:
                        self.names = data
                elif isinstance(data, list):
                    # If data is a list, convert to dict with indices as keys
                    self.names = {str(i): name for i, name in enumerate(data)}
                else:
                    print(f"Warning: Unexpected JSON format in {class_file}, using default names")
                    self.names = {}

            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Failed to load class names from {class_file}: {e}")
                self.names = {}

        # Create default mapping if no class file or loading failed
        if not self.names:
            # Use a reasonable default based on model type
            model_name = os.path.basename(self.model_path).lower()
            if 'char_classifier' in model_name:
                # Common alphanumeric characters
                chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                self.names = {str(i): char for i, char in enumerate(chars)}
            elif 'state_classifier' in model_name:
                # US states (this is a placeholder - should be loaded from JSON)
                states = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA']
                self.names = {str(i): state for i, state in enumerate(states)}
            else:
                # Generic class mapping
                self.names = {str(i): f"class_{i}" for i in range(1000)}
    
    def _preprocess_image(self, image: np.ndarray, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Preprocess image for ONNX inference with validation and proper formatting.

        Args:
            image: Input image as numpy array (expected to be in BGR format)
            target_size: Target size as (width, height). If None, use model's expected size.

        Returns:
            Preprocessed image as numpy array with shape [1, 3, H, W]
        """
        if image is None:
            raise ValueError("Input image is None")

        # Ensure image is a numpy array
        if not isinstance(image, np.ndarray):
            raise ValueError(f"Expected numpy array, got {type(image)}")

        # Ensure image has correct shape
        if len(image.shape) != 3:
            raise ValueError(f"Expected 3D image array, got shape {image.shape}")

        if image.shape[2] != 3:
            raise ValueError(f"Expected 3-channel image, got {image.shape[2]} channels")

        # Determine target size
        if target_size is None:
            target_size = self._get_model_input_size()

        # Apply resizing if target size is specified
        if target_size is not None:
            width, height = target_size
            resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        else:
            resized = image.copy()

        # Convert BGR to RGB (OpenCV uses BGR, models typically expect RGB)
        if len(resized.shape) == 3 and resized.shape[2] == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1] and convert to float32
        normalized = resized.astype(np.float32) / 255.0

        # Transpose from HWC to CHW format and add batch dimension
        # From (H, W, C) to (1, C, H, W)
        if len(normalized.shape) == 3:
            preprocessed = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]
        else:
            # Fallback for grayscale images
            preprocessed = normalized[np.newaxis, np.newaxis, ...]

        # Ensure output is contiguous in memory for ONNX
        preprocessed = np.ascontiguousarray(preprocessed)

        # Validate output shape matches expected input
        if self.input_shape and preprocessed.shape != tuple(self.input_shape):
            expected_str = f"[{', '.join(str(x) for x in self.input_shape)}]"
            actual_str = f"[{', '.join(str(x) for x in preprocessed.shape)}]"
            print(f"Warning: Preprocessed shape {actual_str} doesn't match expected {expected_str}")

        return preprocessed

    def _get_model_input_size(self) -> Optional[Tuple[int, int]]:
        """
        Get the target input size for the specific model based on the model path or ONNX metadata.

        Returns:
            Target size as (width, height) or None if no specific size required
        """
        # First try to get size from ONNX model metadata
        if self.use_onnx and self.input_shape:
            try:
                # ONNX shape is typically [batch, channels, height, width]
                if len(self.input_shape) >= 4:
                    height = self.input_shape[2]
                    width = self.input_shape[3]
                    if isinstance(height, int) and isinstance(width, int):
                        return (width, height)
            except (IndexError, TypeError):
                pass

        # Fallback to model name-based sizing
        model_name = os.path.basename(self.model_path).lower()

        if 'char_classifier' in model_name:
            return (32, 32)
        elif 'char_detector' in model_name:
            return (128, 128)
        elif 'plate_detector' in model_name:
            return (640, 640)
        elif 'state_classifier' in model_name:
            return (224, 224)
        elif 'vehicle' in model_name:
            return (640, 640)
        else:
            return None

    def _validate_onnx_output(self, outputs: List[np.ndarray], expected_shapes: Optional[List[Tuple]] = None) -> bool:
        """
        Validate ONNX model outputs.

        Args:
            outputs: List of output arrays from ONNX inference
            expected_shapes: Optional list of expected shapes for validation

        Returns:
            True if outputs are valid, False otherwise
        """
        if not outputs or len(outputs) == 0:
            print("Warning: ONNX model returned empty outputs")
            return False

        for i, output in enumerate(outputs):
            if output is None:
                print(f"Warning: ONNX output {i} is None")
                return False

            if not isinstance(output, np.ndarray):
                print(f"Warning: ONNX output {i} is not a numpy array")
                return False

            if expected_shapes and i < len(expected_shapes):
                expected_shape = expected_shapes[i]
                if expected_shape and output.shape != expected_shape:
                    print(f"Warning: ONNX output {i} shape {output.shape} doesn't match expected {expected_shape}")

        return True

    def get_session_info(self) -> Dict[str, Any]:
        """
        Get information about the ONNX session for this model.
        
        Returns:
            Dictionary with session information
        """
        if not self.use_onnx or not self.session_manager or not self.session_id:
            return {"error": "No ONNX session available"}
        
        try:
            session_info = self.session_manager.get_session_info()
            return session_info.get(self.session_id, {"error": "Session not found"})
        except Exception as e:
            return {"error": str(e)}

    def cleanup(self):
        """Clean up resources when the model is no longer needed."""
        if self.use_onnx and self.session_manager and self.session_id:
            try:
                # Note: We don't remove the session here as it might be shared
                # The session manager will handle cleanup when appropriate
                print(f"Cleaned up YOLO model resources for {self.model_path}")
            except Exception as e:
                print(f"Error during cleanup for {self.model_path}: {e}")
        
        # Clear references
        self.session_manager = None
        self.session_id = None

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except:
            pass