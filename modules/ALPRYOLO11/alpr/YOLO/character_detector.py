"""
Character detection and recognition for license plates.
This is the main entry point that coordinates detection, classification, and organization.
"""
import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Tuple

from .base import YOLOBase
from ..config import ALPRConfig
from ..exceptions import ModelLoadingError, InferenceError
from ..utils.image_processing import save_debug_image
from .char_organizer import CharacterOrganizer
from .char_classifier_manager import CharacterClassifierManager

# Set up logger for this module
logger = logging.getLogger(__name__)


# Import the YOLO implementations
class CharDetector(YOLOBase):
    """Character detector for license plates using YOLOv8 or ONNX."""
    
    def __init__(self, model_path: str, task: str, use_onnx: bool, use_cuda: bool, 
                 confidence: float, save_debug_images: bool = False, debug_images_dir: str = None,
                 dilation_width: int = 0, dilation_height: int = 0):
        """
        Initialize the character detector.
        
        Args:
            model_path: Path to the model file
            task: Task type
            use_onnx: Whether to use ONNX model
            use_cuda: Whether to use CUDA
            confidence: Confidence threshold
            save_debug_images: Whether to save debug images
            debug_images_dir: Directory for debug images
            dilation_width: Number of pixels to dilate the character box horizontally
            dilation_height: Number of pixels to dilate the character box vertically
        """
        super().__init__(model_path, task, use_onnx, use_cuda)
        self.confidence_threshold = confidence
        self.save_debug_images = save_debug_images
        self.debug_images_dir = debug_images_dir
        self.dilation_width = dilation_width
        self.dilation_height = dilation_height
    
    def detect(self, plate_image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect characters in the license plate image.

        Args:
            plate_image: License plate image

        Returns:
            List of character detections
        """
        # Save input image if debug is enabled
        if self.save_debug_images:
            save_debug_image(
                image=plate_image,
                debug_dir=self.debug_images_dir,
                prefix="char_detector",
                suffix="input",
                draw_objects=None,
                draw_type=None
            )

        try:
            return self._detect_onnx(plate_image)
        except Exception as e:
            raise InferenceError("char_detector", e)
    
    def _apply_nms_xyxy(self, boxes: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
        """Apply Non-Maximum Suppression on XYXY boxes and return kept indices."""
        if boxes.size == 0:
            return np.array([], dtype=int)

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            if order.size == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            intersection = w * h

            union = areas[i] + areas[order[1:]] - intersection
            iou = np.zeros_like(intersection)
            valid_union = union > 0
            iou[valid_union] = intersection[valid_union] / union[valid_union]

            remaining_indices = np.where(iou <= threshold)[0]
            order = order[remaining_indices + 1]

        return np.array(keep, dtype=int)

    def _detect_onnx(self, plate_image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect characters using ONNX model with improved error handling.

        Args:
            plate_image: Input plate image

        Returns:
            List of character detection results
        """
        try:
            # Preprocess image for ONNX with proper validation
            input_tensor = self._preprocess_image(plate_image, target_size=(256, 256))

            # Store original image dimensions for coordinate scaling
            original_h, original_w = plate_image.shape[:2]
            target_w, target_h = 256, 256

            # Run inference with DirectML fallback capability
            try:
                outputs = self._safe_onnx_inference(input_tensor)
            except Exception as e:
                print(f"ONNX inference failed for char_detector: {e}")
                return []

            # Validate outputs
            if not self._validate_onnx_output(outputs):
                return []

            # Parse ONNX outputs - handle YOLOv8 format
            if len(outputs) == 0 or outputs[0] is None:
                return []

            output = outputs[0]  # Main output tensor

            # Handle different output shapes
            if len(output.shape) == 3 and output.shape[0] == 1:
                output = output[0]  # Remove batch dimension

            # Ensure output is in the correct format [num_detections, num_outputs]
            if output.shape[0] < output.shape[1]:
                output = output.T   # Transpose to [num_detections, num_outputs]

            # Extract detection information
            if output.shape[1] >= 5:  # Standard format: [cx, cy, w, h, conf]
                boxes = output[:, :4]           # First 4 columns: bbox coordinates
                confidences = output[:, 4]      # 5th column: objectness/confidence score

                # Filter by confidence threshold
                valid_indices = confidences >= self.confidence_threshold
                valid_boxes = boxes[valid_indices]
                valid_confidences = confidences[valid_indices]

            elif output.shape[1] == 4:  # Only bbox coordinates, no confidence
                boxes = output[:, :4]
                confidences = np.ones(len(boxes))  # Assume all detections are valid
                valid_boxes = boxes
                valid_confidences = confidences

            else:
                # Unexpected output format
                print(f"Warning: Unexpected character detector output shape: {output.shape}")
                return []

            # Process detections
            characters = []

            # Check if we have valid detections
            if len(valid_boxes) == 0:
                return characters

            # Convert from center to corner format for NMS and later processing
            boxes_xyxy = []
            for box in valid_boxes:
                center_x, center_y, width, height = box
                width = max(width, 1.0)
                height = max(height, 1.0)
                x1 = center_x - width / 2
                y1 = center_y - height / 2
                x2 = center_x + width / 2
                y2 = center_y + height / 2
                boxes_xyxy.append([x1, y1, x2, y2])

            boxes_xyxy = np.array(boxes_xyxy, dtype=np.float32)

            if boxes_xyxy.size == 0:
                return characters

            # Apply NMS with IoU threshold tuned for densely packed characters
            keep_indices = self._apply_nms_xyxy(boxes_xyxy, valid_confidences, threshold=0.6)
            if keep_indices.size == 0:
                return characters

            valid_boxes = valid_boxes[keep_indices]
            valid_confidences = valid_confidences[keep_indices]
            boxes_xyxy = boxes_xyxy[keep_indices]

            # Scale coordinates back to original image size
            scale_x = original_w / target_w
            scale_y = original_h / target_h

            for i in range(len(valid_boxes)):
                try:
                    x1, y1, x2, y2 = boxes_xyxy[i]

                    x1 = x1 * scale_x
                    y1 = y1 * scale_y
                    x2 = x2 * scale_x
                    y2 = y2 * scale_y

                    # Convert to integers and ensure coordinates are within image bounds
                    x1 = max(0, int(x1))
                    y1 = max(0, int(y1))
                    x2 = min(original_w, int(x2))
                    y2 = min(original_h, int(y2))

                    # Skip invalid boxes
                    if x1 >= x2 or y1 >= y2:
                        continue

                    # Apply dilation to the box if needed
                    if self.dilation_width > 0 or self.dilation_height > 0:
                        try:
                            from ..utils.image_processing import dilate_char_box
                            dilated_box = dilate_char_box(
                                [x1, y1, x2, y2],
                                dilation_width=self.dilation_width,
                                dilation_height=self.dilation_height,
                                image_size=(original_w, original_h)
                            )
                            x1, y1, x2, y2 = dilated_box
                        except Exception as e:
                            print(f"Warning: Failed to dilate character box: {e}")

                    # Extract character image
                    if 0 <= y1 < y2 <= original_h and 0 <= x1 < x2 <= original_w:
                        char_img = plate_image[y1:y2, x1:x2]

                        # Validate character image
                        if char_img.size > 0:
                            # Save individual character crop if debug is enabled
                            if self.save_debug_images:
                                try:
                                    save_debug_image(
                                        image=char_img,
                                        debug_dir=self.debug_images_dir,
                                        prefix="char_crop",
                                        suffix=f"char_{i}_onnx" + ("_dilated" if (self.dilation_width > 0 or self.dilation_height > 0) else ""),
                                        draw_objects=None,
                                        draw_type=None
                                    )
                                except Exception as e:
                                    print(f"Warning: Failed to save debug character image: {e}")

                            # Store detection
                            characters.append({
                                "box": [x1, y1, x2, y2],
                                "confidence": float(valid_confidences[i]),
                                "image": char_img
                            })

                except Exception as e:
                    print(f"Warning: Error processing character detection {i}: {e}")
                    continue

            # FIXED: Removed premature sorting - let organize_characters() handle proper ordering
            # This was causing character ordering issues for multi-line, overlapping, or misaligned characters
            # characters.sort(key=lambda char: char["box"][0])

            return characters

        except Exception as e:
            print(f"Error in character detector ONNX inference: {e}")
            return []


class CharClassifier(YOLOBase):
    """Character classifier for license plates using YOLOv8 or ONNX."""
    
    def __init__(self, model_path: str, task: str, use_onnx: bool, use_cuda: bool, 
                 resolution: tuple, confidence: float,
                 save_debug_images: bool = False, debug_images_dir: str = None):
        """
        Initialize the character classifier.
        
        Args:
            model_path: Path to the model file
            task: Task type
            use_onnx: Whether to use ONNX model
            use_cuda: Whether to use CUDA
            resolution: Input resolution for the model
            confidence: Confidence threshold
            save_debug_images: Whether to save debug images
            debug_images_dir: Directory for debug images
        """
        super().__init__(model_path, task, use_onnx, use_cuda)
        self.resolution = resolution  # Keep this for character classification, as specific size may be needed
        self.confidence_threshold = confidence
        self.save_debug_images = save_debug_images
        self.debug_images_dir = debug_images_dir
    
    def _resolve_label(self, idx: int) -> str:
        """Resolve a class index to its label, falling back to known name tables."""
        # Use names loaded from JSON manifest (ONNX models don't have embedded names)
        return self._lookup_name(getattr(self, "names", {}), idx)

    @staticmethod
    def _lookup_name(names: Any, idx: int) -> str:
        """Lookup a label from a YOLO names structure (dict, list, tuple)."""
        if isinstance(names, dict):
            return names.get(idx) or names.get(str(idx), "?")
        if isinstance(names, (list, tuple)):
            if 0 <= idx < len(names):
                return names[idx]
        return "?"

    def classify(self, char_image: np.ndarray) -> List[tuple]:
        """
        Classify a character and return top predictions.

        Args:
            char_image: Character image

        Returns:
            List of (character, confidence) tuples
        """
        if char_image.size == 0:
            return [("?", 0.0)]

        char_image = cv2.cvtColor(char_image, cv2.COLOR_BGR2GRAY)
        char_image = cv2.cvtColor(char_image, cv2.COLOR_GRAY2BGR)

        # Use letterbox resizing for consistent sizing while maintaining aspect ratio
        try:
            from ..utils.image_processing import letterbox_resize
            char_resized, _, _ = letterbox_resize(char_image, self.resolution, fill_value=0)

            # Save resized character image if debug is enabled
            if self.save_debug_images:
                save_debug_image(
                    image=char_resized,
                    debug_dir=self.debug_images_dir,
                    prefix="char_classifier",
                    suffix="resized_input",
                    draw_objects=None,
                    draw_type=None
                )
        except Exception as e:
            from ..exceptions import CharacterRecognitionError
            raise CharacterRecognitionError(f"Failed to resize character image with letterbox: {str(e)}")

        try:
            result = self._classify_onnx(char_resized)
                
            # Save classification result visualization if debug is enabled
            if self.save_debug_images:
                # Create a visualization of the classification result
                result_img = char_resized.copy()
                # Scale the image for better visibility (3x)
                result_img = cv2.resize(result_img, (self.resolution[0] * 3, self.resolution[1] * 3))
                
                # Draw the top prediction
                top_char, top_conf = result[0] if result else ("?", 0.0)
                label = f"{top_char}: {top_conf:.2f}"
                cv2.putText(result_img, label, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Add other top predictions
                y_offset = 60
                for i, (char, conf) in enumerate(result[1:]):
                    alt_label = f"{char}: {conf:.2f}"
                    cv2.putText(result_img, alt_label, (10, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 1)
                    y_offset += 25
                
                save_debug_image(
                    image=result_img,
                    debug_dir=self.debug_images_dir,
                    prefix="char_classifier",
                    suffix=f"result_{top_char}",
                    draw_objects=None,
                    draw_type=None
                )
                
            return result
        except Exception as e:
            raise InferenceError("char_classifier", e)
    
    def _classify_onnx(self, char_image: np.ndarray) -> List[tuple]:
        """
        Classify character using ONNX model with improved error handling.

        Args:
            char_image: Input character image

        Returns:
            List of (character, confidence) tuples sorted by confidence
        """
        if char_image is None or char_image.size == 0:
            return [("?", 0.0)]

        try:
            input_tensor = self._preprocess_image(char_image, target_size=(64, 64))
        except Exception as exc:
            logger.warning("Failed to preprocess character image for ONNX classification: %s", exc)
            return [("?", 0.0)]

        try:
            outputs = self._safe_onnx_inference(input_tensor)
        except Exception as exc:
            logger.warning("ONNX inference failed for char_classifier: %s", exc)
            return [("?", 0.0)]

        if not self._validate_onnx_output(outputs):
            return [("?", 0.0)]

        if not outputs or outputs[0] is None:
            return [("?", 0.0)]

        probs = outputs[0]

        if len(probs.shape) > 1 and probs.shape[0] == 1:
            probs = probs[0]

        if probs.size == 0:
            return [("?", 0.0)]

        try:
            valid_probs = np.where(np.isfinite(probs), probs, 0.0).astype(np.float32)
        except Exception as exc:
            logger.warning("Invalid probabilities from ONNX character classifier: %s", exc)
            return [("?", 0.0)]

        top_predictions = []

        try:
            top_indices = np.argsort(valid_probs)[::-1][:5]
            for idx in top_indices:
                conf = float(np.clip(valid_probs[int(idx)], 0.0, 1.0))
                if conf >= 0.02:
                    top_predictions.append((self._resolve_label(int(idx)), conf))
        except Exception as exc:
            logger.warning("Error processing ONNX character classification results: %s", exc)
            return [("?", 0.0)]

        if top_predictions:
            return top_predictions

        best_idx = int(np.argmax(valid_probs))
        best_conf = float(np.clip(valid_probs[best_idx], 0.0, 1.0))
        return [(self._resolve_label(best_idx), best_conf)]


class CharacterDetector:
    """
    Character detector for license plates using YOLOv8.
    Detects individual characters on license plates and recognizes them.
    """
    
    def __init__(self, config: ALPRConfig):
        """
        Initialize the character detector.

        Args:
            config: ALPR configuration object

        Raises:
            ModelLoadingError: If model loading fails
        """
        self.config = config
        self.char_detector_path = config.get_model_path("char_detector")
        self.char_classifier_path = config.get_model_path("char_classifier")
        self.char_detector_confidence = config.char_detector_confidence
        self.char_classifier_confidence = config.char_classifier_confidence
        self.save_debug_images = config.save_debug_images
        self.debug_images_dir = config.debug_images_dir

        # Character classifier still needs a fixed resolution
        self.char_classifier_resolution = (64, 64)

        # Initialize attributes to None first
        self.char_detector = None
        self.char_classifier = None
        
        # Initialize helper components
        self.organizer = CharacterOrganizer(config)
        self.classifier_manager = None  # Will be initialized after char_classifier

        # Initialize the detector model
        try:
            self.char_detector = CharDetector(
                model_path=self.char_detector_path,
                task='detect',
                use_onnx=True,
                use_cuda=config.use_cuda,
                confidence=self.char_detector_confidence,
                save_debug_images=self.save_debug_images,
                debug_images_dir=self.debug_images_dir,
                dilation_width=config.char_box_dilation_width,
                dilation_height=config.char_box_dilation_height
            )
        except Exception as e:
            self.char_detector = None
            self.char_classifier = None
            raise ModelLoadingError(self.char_detector_path, e)

        # Initialize the classifier model
        try:
            self.char_classifier = CharClassifier(
                model_path=self.char_classifier_path,
                task='classify',
                use_onnx=True,
                use_cuda=config.use_cuda,
                resolution=self.char_classifier_resolution,
                confidence=self.char_classifier_confidence,
                save_debug_images=self.save_debug_images,
                debug_images_dir=self.debug_images_dir
            )
            
            # Initialize classifier manager with the classifier
            self.classifier_manager = CharacterClassifierManager(self.char_classifier)
            
        except Exception as e:
            self.char_detector = None
            self.char_classifier = None
            raise ModelLoadingError(self.char_classifier_path, e)
    
    def detect_characters(self, plate_image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect characters in the license plate image.

        Args:
            plate_image: License plate image as numpy array

        Returns:
            List of dictionaries with character box coordinates and images

        Raises:
            InferenceError: If detection fails
        """
        if self.char_detector is None:
            raise InferenceError("Character detector not properly initialized. Check model loading.")

        # Get raw detections from the character detector
        raw_detections = self.char_detector.detect(plate_image)
        
        # Filter duplicate detections using IoU threshold
        duplicate_iou_threshold = getattr(self.config, 'char_duplicate_iou_threshold', 0.5)
        filtered_detections = self._filter_duplicate_detections(raw_detections, duplicate_iou_threshold)
        
        return filtered_detections
    
    def organize_characters(self, characters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Organize characters into a coherent structure, handling multiple lines and vertical characters.
        
        Args:
            characters: List of character dictionaries from detect_characters()
            
        Returns:
            List of characters in reading order
        """
        return self.organizer.organize_characters(characters)

    def classify_character(self, char_image: np.ndarray):
        """
        Classify a character using OCR and return top predictions.

        Args:
            char_image: Character image to classify

        Returns:
            List of tuples containing (character, confidence) for top predictions

        Raises:
            InferenceError: If classification fails
        """
        if self.classifier_manager is None:
            raise InferenceError("Character classifier not properly initialized. Check model loading.")
        
        return self.classifier_manager.classify_character(char_image)
    
    def _filter_duplicate_detections(self, characters: List[Dict[str, Any]], iou_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Filter duplicate character detections by removing overlapping boxes.
        
        This method implements a custom Non-Maximum Suppression (NMS) to handle cases
        where the YOLO model detects multiple bounding boxes for the same character.
        
        Args:
            characters: List of character detections from detect_characters()
            iou_threshold: IoU threshold for considering detections as duplicates (default: 0.5)
            
        Returns:
            Filtered list of character detections with duplicates removed
        """
        if len(characters) <= 1:
            return characters
        
        # Extract boxes and confidences
        boxes = []
        confidences = []
        valid_chars = []
        
        for char in characters:
            box = char.get("box", None)
            confidence = char.get("confidence", 0.0)
            
            if box and len(box) == 4 and box[2] > box[0] and box[3] > box[1]:
                boxes.append(box)
                confidences.append(confidence)
                valid_chars.append(char)
        
        if len(valid_chars) <= 1:
            return valid_chars
        
        # Convert to numpy arrays for easier processing
        boxes = np.array(boxes)
        confidences = np.array(confidences)
        
        # Sort by confidence (descending order)
        sorted_indices = np.argsort(confidences)[::-1]
        
        # Keep track of which detections to keep
        keep = []
        suppressed = set()
        
        for i in sorted_indices:
            if i in suppressed:
                continue
                
            keep.append(i)
            
            # Suppress overlapping detections
            for j in sorted_indices:
                if j == i or j in suppressed:
                    continue
                    
                # Calculate IoU between boxes i and j
                iou = self._calculate_iou(boxes[i], boxes[j])
                
                # If IoU is above threshold, suppress the lower confidence detection
                if iou > iou_threshold:
                    suppressed.add(j)
        
        # Return filtered detections
        filtered_chars = [valid_chars[i] for i in keep]
        
        if self.save_debug_images and len(filtered_chars) < len(characters):
            print(f"*** DUPLICATE FILTERING: Reduced {len(characters)} detections to {len(filtered_chars)} ***")
        
        # Additional validation: ensure we haven't over-filtered
        if len(filtered_chars) == 0 and len(characters) > 0:
            best_idx = np.argmax(confidences)
            return [valid_chars[best_idx]]
        
        return filtered_chars
    
    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) of two bounding boxes."""
        x_overlap = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0]))
        y_overlap = max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
        intersection_area = x_overlap * y_overlap
        
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - intersection_area
        
        if union_area <= 0:
            return 0.0
        
        return intersection_area / union_area

    def process_plate(self, plate_image: np.ndarray) -> Dict[str, Any]:
        """
        Detect and recognize characters on a license plate.

        Args:
            plate_image: License plate image

        Returns:
            Dictionary with character detections and plate number
        """
        # Convert to grayscale and back to BGR for consistency
        plate_image = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        plate_image = cv2.cvtColor(plate_image, cv2.COLOR_GRAY2BGR)

        if self.save_debug_images:
            save_debug_image(
                image=plate_image,
                debug_dir=self.debug_images_dir,
                prefix="char_process",
                suffix="plate_input",
                draw_objects=None,
                draw_type=None
            )

        # 1. Detect characters
        characters = self.detect_characters(plate_image)
        
        # 2. Organize characters (handle multiple lines and vertical characters)
        organized_chars = self.organize_characters(characters)
        
        # 3. Classify characters and generate plate reading
        char_results, license_number, avg_confidence, top_plates = \
            self.classifier_manager.process_characters_for_plate(organized_chars)
        
        return {
            "characters": char_results,
            "license_number": license_number,
            "confidence": avg_confidence,
            "top_plates": top_plates
        }
    
    def cleanup(self):
        """Clean up resources when the detector is no longer needed."""
        try:
            if self.char_detector:
                self.char_detector.cleanup()
            if self.char_classifier:
                self.char_classifier.cleanup()
        except Exception as e:
            logger.warning(f"Error during CharacterDetector cleanup: {e}")
