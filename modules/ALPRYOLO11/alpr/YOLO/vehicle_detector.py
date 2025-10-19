"""
Vehicle detection and classification using YOLOv8.
"""
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple

from .base import YOLOBase
from ..config import ALPRConfig
from ..exceptions import ModelLoadingError, InferenceError, VehicleDetectionError
from ..utils.image_processing import save_debug_image


class VehicleDetector:
    """
    Vehicle detector and classifier using YOLOv8.
    Detects vehicles and identifies make/model.
    """
    
    def __init__(self, config: ALPRConfig):
        """
        Initialize the vehicle detector and classifier.
        
        Args:
            config: ALPR configuration object
            
        Raises:
            ModelLoadingError: If model loading fails
        """
        self.config = config
        self.vehicle_detector_path = config.get_model_path("vehicle_detector")
        self.vehicle_classifier_path = config.get_model_path("vehicle_classifier")
        self.vehicle_detector_confidence = config.vehicle_detector_confidence
        self.vehicle_classifier_confidence = config.vehicle_classifier_confidence
        self.save_debug_images = config.save_debug_images
        self.debug_images_dir = config.debug_images_dir
        
        # Vehicle classifier still needs a fixed resolution
        self.vehicle_classifier_resolution = (224, 224)
        
        # Skip initialization if vehicle detection is disabled
        if not config.enable_vehicle_detection:
            self.detector = None
            self.classifier = None
            return
            
        # Initialize the models
        try:
            self.detector = VehicleDetectorYOLO(
                model_path=self.vehicle_detector_path,
                task='detect',
                use_onnx=True,
                use_cuda=config.use_cuda,
                confidence=self.vehicle_detector_confidence,
                save_debug_images=self.save_debug_images,
                debug_images_dir=self.debug_images_dir,
                device_id=config.device_id
            )
        except Exception as e:
            raise ModelLoadingError(self.vehicle_detector_path, e)
            
        try:
            self.classifier = VehicleClassifierYOLO(
                model_path=self.vehicle_classifier_path,
                task='classify',
                use_onnx=True,
                use_cuda=config.use_cuda,
                resolution=self.vehicle_classifier_resolution,
                confidence=self.vehicle_classifier_confidence,
                save_debug_images=self.save_debug_images,
                debug_images_dir=self.debug_images_dir,
                device_id=config.device_id
            )
        except Exception as e:
            raise ModelLoadingError(self.vehicle_classifier_path, e)
    
    def detect_vehicles(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect vehicles in the image.

        Args:
            image: Input image as numpy array

        Returns:
            List of dictionaries with vehicle information

        Raises:
            InferenceError: If detection fails
        """
        if self.detector is None:
            return []

        return self.detector.detect(image)

    def classify_vehicle(self, vehicle_image: np.ndarray) -> Dict[str, Any]:
        """
        Classify vehicle make and model.

        Args:
            vehicle_image: Vehicle image to classify

        Returns:
            Dictionary with make, model, and confidence

        Raises:
            InferenceError: If classification fails
            VehicleDetectionError: If vehicle image is invalid
        """
        if self.classifier is None or vehicle_image.size == 0:
            return {"make": "Unknown", "model": "Unknown", "confidence": 0.0}

        return self.classifier.classify(vehicle_image)

    def detect_and_classify(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect vehicles and classify their make/model.

        Args:
            image: Input image

        Returns:
            List of dictionaries with vehicle information
        """
        # Save input image if debug is enabled
        if self.save_debug_images:
            save_debug_image(
                image=image,
                debug_dir=self.debug_images_dir,
                prefix="vehicle_process",
                suffix="input",
                draw_objects=None,
                draw_type=None
            )

        # Detect vehicles
        vehicles = self.detect_vehicles(image)

        # Classify each vehicle
        vehicle_results = []
        for vehicle_info in vehicles:
            try:
                classification = self.classify_vehicle(vehicle_info["image"])
                
                # Create result with bounding box and classification
                result = {
                    "box": vehicle_info["box"],
                    "confidence": vehicle_info["confidence"],
                    "make": classification["make"],
                    "model": classification["model"],
                    "classification_confidence": classification["confidence"]
                }
                vehicle_results.append(result)
            except Exception as e:
                # Skip this vehicle if classification fails
                continue
        
        # Save final detection and classification results if debug is enabled
        if self.save_debug_images and vehicle_results:
            debug_img = image.copy()
            
            # Draw each detected and classified vehicle
            for i, vehicle in enumerate(vehicle_results):
                x1, y1, x2, y2 = vehicle["box"]
                # Draw box
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Draw make/model
                make_model = f"{vehicle['make']} {vehicle['model']}"
                confidence = f"Det: {vehicle['confidence']:.2f}, Cls: {vehicle['classification_confidence']:.2f}"
                cv2.putText(debug_img, make_model, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(debug_img, confidence, (x1, y1-30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Save the debug image
            save_debug_image(
                image=debug_img,
                debug_dir=self.debug_images_dir,
                prefix="vehicle_result",
                suffix=f"detected_{len(vehicle_results)}",
                draw_objects=None,
                draw_type=None
            )
                
        return vehicle_results
    
    def __call__(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Convenience method to call detect_and_classify().
        
        Args:
            image: Input image
            
        Returns:
            Detection and classification results
        """
        return self.detect_and_classify(image)


class VehicleDetectorYOLO(YOLOBase):
    """Vehicle detector using YOLOv8."""
    
    def __init__(self, model_path: str, task: str, use_onnx: bool, use_cuda: bool, 
                 confidence: float, save_debug_images: bool = False, debug_images_dir: str = None,
                 device_id: int = None):
        """
        Initialize the vehicle detector.
        
        Args:
            model_path: Path to the model file
            task: Task type ('detect')
            use_onnx: Whether to use ONNX model
            use_cuda: Whether to use CUDA
            confidence: Confidence threshold
            save_debug_images: Whether to save debug images
            debug_images_dir: Directory for debug images
            device_id: GPU device ID (None for auto-detect)
        """
        super().__init__(model_path, task, use_onnx, use_cuda, device_id)
        self.confidence_threshold = confidence
        self.save_debug_images = save_debug_images
        self.debug_images_dir = debug_images_dir
    
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect vehicles in the image.
        
        Args:
            image: Input image
            
        Returns:
            List of vehicle detections
        """
        # Save input image if debug is enabled
        if self.save_debug_images:
            save_debug_image(
                image=image,
                debug_dir=self.debug_images_dir,
                prefix="vehicle_detector",
                suffix="input",
                draw_objects=None,
                draw_type=None
            )
        
        try:
            return self._detect_onnx(image)
        except Exception as e:
            raise InferenceError("vehicle_detector", e)
    def _detect_onnx(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect vehicles using ONNX model with improved error handling.

        Args:
            image: Input image

        Returns:
            List of vehicle detection results
        """
        try:
            # Preprocess image for ONNX with proper validation
            input_tensor = self._preprocess_image(image)

            # Run inference with DirectML fallback capability
            try:
                outputs = self._safe_onnx_inference(input_tensor)
            except Exception as e:
                print(f"ONNX inference failed for vehicle_detector: {e}")
                return []

            # Validate outputs
            if not self._validate_onnx_output(outputs):
                return []

            # Parse ONNX outputs
            if len(outputs) == 0:
                return []

            # Handle different output formats
            boxes = outputs[0] if outputs[0] is not None else []
            scores = outputs[1] if len(outputs) > 1 and outputs[1] is not None else None

            # Process detections
            vehicles = []

            if boxes is not None and len(boxes) > 0:
                h, w = image.shape[:2]

                for i in range(len(boxes)):
                    try:
                        # Skip detections below confidence threshold
                        if scores is not None:
                            confidence = float(scores[i]) if i < len(scores) else 0.0
                            if confidence < self.confidence_threshold:
                                continue
                        else:
                            confidence = 1.0  # Default confidence if not available

                        # Get box coordinates
                        if len(boxes[i]) >= 4:
                            x1, y1, x2, y2 = boxes[i][:4]
                        else:
                            continue  # Skip invalid boxes

                        # Convert to integers and ensure coordinates are within image bounds
                        x1 = max(0, int(x1))
                        y1 = max(0, int(y1))
                        x2 = min(w, int(x2))
                        y2 = min(h, int(y2))

                        # Skip invalid boxes
                        if x1 >= x2 or y1 >= y2:
                            continue

                        # Extract vehicle image
                        if 0 <= y1 < y2 <= h and 0 <= x1 < x2 <= w:
                            vehicle_img = image[y1:y2, x1:x2]

                            # Validate vehicle image
                            if vehicle_img.size > 0:
                                # Save individual vehicle crop if debug is enabled
                                if self.save_debug_images:
                                    try:
                                        save_debug_image(
                                            image=vehicle_img,
                                            debug_dir=self.debug_images_dir,
                                            prefix="vehicle_crop",
                                            suffix=f"vehicle_{i}_onnx",
                                            draw_objects=None,
                                            draw_type=None
                                        )
                                    except Exception as e:
                                        print(f"Warning: Failed to save debug vehicle image: {e}")

                                # Store detection
                                vehicles.append({
                                    "box": [x1, y1, x2, y2],
                                    "confidence": confidence,
                                    "image": vehicle_img
                                })

                    except Exception as e:
                        print(f"Warning: Error processing vehicle detection {i}: {e}")
                        continue

            return vehicles

        except Exception as e:
            print(f"Error in vehicle detector ONNX inference: {e}")
            return []


class VehicleClassifierYOLO(YOLOBase):
    """Vehicle classifier using YOLOv8."""
    
    def __init__(self, model_path: str, task: str, use_onnx: bool, use_cuda: bool, 
                 resolution: Tuple[int, int], confidence: float,
                 save_debug_images: bool = False, debug_images_dir: str = None,
                 device_id: int = None):
        """
        Initialize the vehicle classifier.
        
        Args:
            model_path: Path to the model file
            task: Task type ('classify')
            use_onnx: Whether to use ONNX model
            use_cuda: Whether to use CUDA
            resolution: Input resolution for the model
            confidence: Confidence threshold
            save_debug_images: Whether to save debug images
            debug_images_dir: Directory for debug images
            device_id: GPU device ID (None for auto-detect)
        """
        super().__init__(model_path, task, use_onnx, use_cuda, device_id)
        # Keep resolution for ONNX preprocessing or if specific size is needed for classification
        self.resolution = resolution
        self.confidence_threshold = confidence
        self.save_debug_images = save_debug_images
        self.debug_images_dir = debug_images_dir
    
    def classify(self, vehicle_image: np.ndarray) -> Dict[str, Any]:
        """
        Classify vehicle make and model.
        
        Args:
            vehicle_image: Vehicle image
            
        Returns:
            Dictionary with make, model, and confidence
        """
        if vehicle_image.size == 0:
            return {"make": "Unknown", "model": "Unknown", "confidence": 0.0}
        
        # Save vehicle image for debugging if enabled
        if self.save_debug_images:
            save_debug_image(
                image=vehicle_image,
                debug_dir=self.debug_images_dir,
                prefix="vehicle_classifier",
                suffix="input",
                draw_objects=None,
                draw_type=None
            )
        
        try:
            result = self._classify_onnx(vehicle_image)
                
            # Save classification result visualization if debug is enabled
            if self.save_debug_images:
                # Create a visualization of the classification result
                result_img = vehicle_image.copy()
                h, w = result_img.shape[:2]
                
                # Add a label at the bottom
                label_bg = np.zeros((80, w, 3), dtype=np.uint8)
                result_img = np.vstack([result_img, label_bg])
                
                # Draw the make and model
                make_model = f"{result['make']} {result['model']}"
                confidence = f"Confidence: {result['confidence']:.2f}"
                
                cv2.putText(result_img, make_model, (10, h+30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(result_img, confidence, (10, h+60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                
                save_debug_image(
                    image=result_img,
                    debug_dir=self.debug_images_dir,
                    prefix="vehicle_classifier",
                    suffix=f"result_{result['make']}_{result['model']}",
                    draw_objects=None,
                    draw_type=None
                )
                
            return result
        except Exception as e:
            raise InferenceError("vehicle_classifier", e)
    
    def _classify_onnx(self, vehicle_image: np.ndarray) -> Dict[str, Any]:
        """
        Classify vehicle using ONNX model with improved error handling.

        Args:
            vehicle_image: Input vehicle image

        Returns:
            Dictionary with vehicle classification results
        """
        try:
            # Validate input image
            if vehicle_image is None or vehicle_image.size == 0:
                return {"make": "Unknown", "model": "Unknown", "confidence": 0.0}

            # Preprocess image for ONNX with proper validation
            input_tensor = self._preprocess_image(vehicle_image)

            # Run inference with DirectML fallback capability
            try:
                outputs = self._safe_onnx_inference(input_tensor)
            except Exception as e:
                print(f"ONNX inference failed for vehicle_classifier: {e}")
                return {"make": "Unknown", "model": "Unknown", "confidence": 0.0}

            # Validate outputs
            if not self._validate_onnx_output(outputs):
                return {"make": "Unknown", "model": "Unknown", "confidence": 0.0}

            # Parse ONNX outputs
            if len(outputs) == 0 or outputs[0] is None:
                return {"make": "Unknown", "model": "Unknown", "confidence": 0.0}

            probs = outputs[0]

            # Handle different output shapes
            if len(probs.shape) > 1 and probs.shape[0] == 1:
                probs = probs[0]  # Remove batch dimension

            if len(probs) == 0:
                return {"make": "Unknown", "model": "Unknown", "confidence": 0.0}

            try:
                # Ensure probabilities are valid
                valid_probs = np.where(np.isfinite(probs), probs, 0.0)

                # Get the index with highest probability
                vehicle_idx = np.argmax(valid_probs)
                confidence = float(valid_probs[vehicle_idx])

                # Ensure confidence is in valid range
                confidence = max(0.0, min(1.0, confidence))

                # Check if confidence meets threshold
                if confidence < self.confidence_threshold:
                    return {"make": "Unknown", "model": "Unknown", "confidence": confidence}

                # Get vehicle name from class index
                vehicle_name = self.names.get(str(vehicle_idx), self.names.get(vehicle_idx, "Unknown"))

                # Split make and model (assuming format "Make_Model")
                try:
                    if "_" in vehicle_name:
                        make, model = vehicle_name.split("_", 1)
                    else:
                        make = vehicle_name
                        model = "Unknown"
                except Exception:
                    make = "Unknown"
                    model = "Unknown"

                return {"make": make, "model": model, "confidence": confidence}

            except Exception as e:
                print(f"Error processing vehicle classification results: {e}")
                return {"make": "Unknown", "model": "Unknown", "confidence": 0.0}

        except Exception as e:
            print(f"Error in vehicle classifier ONNX inference: {e}")
            return {"make": "Unknown", "model": "Unknown", "confidence": 0.0}
