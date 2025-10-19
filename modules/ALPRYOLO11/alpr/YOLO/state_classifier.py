
"""
State classification for license plates using YOLOv8.
"""
import cv2
import numpy as np
import logging
from typing import Dict, Any

from .base import YOLOBase
from ..config import ALPRConfig
from ..exceptions import ModelLoadingError, InferenceError
from ..utils.image_processing import save_debug_image


class StateClassifier(YOLOBase):
    """
    State classifier for license plates using YOLOv8.
    Identifies the state of origin for a license plate.
    """
    
    def __init__(self, config: ALPRConfig):
        """
        Initialize the state classifier.
        
        Args:
            config: ALPR configuration object
            
        Raises:
            ModelLoadingError: If model loading fails
        """
        self.config = config
        self.model_path = config.get_model_path("state_classifier")
        self.confidence_threshold = config.state_classifier_confidence
        self.save_debug_images = config.save_debug_images
        self.debug_images_dir = config.debug_images_dir
        
        # Skip initialization if state detection is disabled
        if not config.enable_state_detection:
            self.session_id = None
            self.session_manager = None
            return
            
        # Initialize the model
        try:
            super().__init__(
                model_path=self.model_path,
                task='classify',
                use_onnx=True,
                use_cuda=config.use_cuda,
                device_id=config.device_id
            )
        except Exception as e:
            raise ModelLoadingError(self.model_path, e)
    
    def classify(self, plate_image: np.ndarray) -> Dict[str, Any]:
        """
        Classify the state of the license plate.

        Args:
            plate_image: License plate image as numpy array

        Returns:
            Dictionary with state name and confidence

        Raises:
            InferenceError: If classification fails
        """
        if self.session_id is None or self.session_manager is None:
            return {"state": "Unknown", "confidence": 0.0}

        # Save input image for debugging if enabled
        if self.save_debug_images:
            save_debug_image(
                image=plate_image,
                debug_dir=self.debug_images_dir,
                prefix="state_classifier",
                suffix="input",
                draw_objects=None,
                draw_type=None
            )

        try:
            result = self._classify_onnx(plate_image)

            # Save classification result visualization if debug is enabled
            if self.save_debug_images:
                # Create a visualization of the state classification result
                result_img = plate_image.copy()
                h, w = result_img.shape[:2]

                # Add some space at the bottom for the label
                label_bg = np.zeros((60, w, 3), dtype=np.uint8)
                result_img = np.vstack([result_img, label_bg])

                # Draw the state and confidence
                state_text = f"State: {result['state']}"
                confidence_text = f"Confidence: {result['confidence']:.2f}"

                cv2.putText(result_img, state_text, (10, h+30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(result_img, confidence_text, (10, h+55),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                save_debug_image(
                    image=result_img,
                    debug_dir=self.debug_images_dir,
                    prefix="state_classifier",
                    suffix=f"result_{result['state']}",
                    draw_objects=None,
                    draw_type=None
                )

            return result
        except Exception as e:
            raise InferenceError("state_classifier", e)
    
    def _classify_onnx(self, plate_image: np.ndarray) -> Dict[str, Any]:
        """
        Classify state using ONNX model with improved error handling.

        Args:
            plate_image: Input plate image

        Returns:
            Dictionary with classification results
        """
        try:
            # Preprocess image for ONNX with proper validation
            input_tensor = self._preprocess_image(plate_image, target_size=(224, 224))

            # Run inference with DirectML fallback capability
            try:
                outputs = self._safe_onnx_inference(input_tensor)
            except Exception as e:
                print(f"ONNX inference failed for state_classifier: {e}")
                return {"state": "Unknown", "confidence": 0.0}

            # Validate outputs
            if not self._validate_onnx_output(outputs):
                return {"state": "Unknown", "confidence": 0.0}

            # Parse ONNX outputs
            # For classification, output is typically class probabilities
            if len(outputs) == 0 or outputs[0] is None:
                return {"state": "Unknown", "confidence": 0.0}

            probs = outputs[0]

            # Handle different output shapes
            if len(probs.shape) > 1 and probs.shape[0] == 1:
                probs = probs[0]  # Remove batch dimension

            if len(probs) == 0:
                return {"state": "Unknown", "confidence": 0.0}

            # Get the index with highest probability
            try:
                state_idx = np.argmax(probs)
                confidence = float(probs[state_idx])

                # Ensure confidence is valid
                if np.isnan(confidence) or np.isinf(confidence):
                    confidence = 0.0

                # Clamp confidence to valid range
                confidence = max(0.0, min(1.0, confidence))

                # Get state name from class index
                state_name = self.names.get(str(state_idx), self.names.get(state_idx, "Unknown"))

                # If confidence is too low, return Unknown
                if confidence < self.confidence_threshold:
                    return {"state": "Unknown", "confidence": confidence}

                # Save debug information if enabled
                if self.save_debug_images:
                    try:
                        self._save_debug_classification(probs, state_name, confidence)
                    except Exception as e:
                        print(f"Warning: Failed to save debug classification: {e}")

                return {"state": state_name, "confidence": confidence}

            except (IndexError, ValueError) as e:
                print(f"Error processing state classification results: {e}")
                return {"state": "Unknown", "confidence": 0.0}

        except Exception as e:
            print(f"Error in state classifier ONNX inference: {e}")
            return {"state": "Unknown", "confidence": 0.0}

    def _save_debug_classification(self, probs: np.ndarray, predicted_state: str, confidence: float):
        """Save debug visualization for classification results"""
        try:
            # Sort indices by probability
            sorted_indices = np.argsort(-probs)[:5]  # Top 5 predictions

            # Create a blank image to show alternatives
            alt_img = np.ones((200, 400, 3), dtype=np.uint8) * 255

            # Title
            cv2.putText(alt_img, "Top State Predictions (ONNX):", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            # Draw each state prediction
            for i, idx in enumerate(sorted_indices):
                if idx < len(probs):
                    conf = float(probs[idx])
                    # Ensure confidence is valid for display
                    if np.isnan(conf) or np.isinf(conf):
                        conf = 0.0
                    conf = max(0.0, min(1.0, conf))

                    name = self.names.get(str(idx), self.names.get(idx, f"class_{idx}"))
                    text = f"{name}: {conf:.4f}"

                    # Highlight the predicted class
                    color = (0, 150, 0) if name == predicted_state else (0, 0, 0)

                    cv2.putText(alt_img, text, (20, 70 + i*25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

            # Save the alternatives image
            save_debug_image(
                image=alt_img,
                debug_dir=self.debug_images_dir,
                prefix="state_classifier",
                suffix="top_probs_onnx",
                draw_objects=None,
                draw_type=None
            )
        except Exception as e:
            print(f"Error creating state classification debug visualization: {e}")
    
    def __call__(self, plate_image: np.ndarray) -> Dict[str, Any]:
        """
        Convenience method to call classify().
        
        Args:
            plate_image: License plate image
            
        Returns:
            Classification results
        """
        return self.classify(plate_image)
