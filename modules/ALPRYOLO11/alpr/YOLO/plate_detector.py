"""
License plate detection module using YOLOv8 keypoint detection model.
"""
import os
import cv2
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

from .base import YOLOBase
from ..config import ALPRConfig
from ..exceptions import ModelLoadingError, InferenceError
from ..utils.image_processing import dilate_corners, save_debug_image


class PlateDetector(YOLOBase):
    """
    License plate detector using YOLOv8 keypoint detection model.
    Detects both day and night license plates in images.
    """
    
    def __init__(self, config: ALPRConfig):
        """
        Initialize the license plate detector.
        
        Args:
            config: ALPR configuration object
            
        Raises:
            ModelLoadingError: If model loading fails
        """
        self.config = config
        self.model_path = config.get_model_path("plate_detector")
        self.confidence_threshold = config.plate_detector_confidence
        self.corner_dilation_pixels = config.corner_dilation_pixels
        self.save_debug_images = config.save_debug_images
        self.debug_images_dir = config.debug_images_dir
        
        # Initialize the model
        try:
            super().__init__(
                model_path=self.model_path,
                task='pose',
                use_onnx=True,
                use_cuda=config.use_cuda,
                device_id=config.device_id
            )
        except Exception as e:
            raise ModelLoadingError(self.model_path, e)
    
    def detect(self, image: np.ndarray) -> Dict[str, List[Dict[str, Any]]]:
        """
        Detect license plates in the image.

        Args:
            image: Input image as numpy array (BGR format)

        Returns:
            Dictionary with 'day_plates' and 'night_plates' lists containing detection results

        Raises:
            InferenceError: If detection fails
        """
        # Get original image dimensions
        h, w = image.shape[:2]

        # Save input image for debugging if enabled
        if self.save_debug_images:
            save_debug_image(
                image=image,
                debug_dir=self.debug_images_dir,
                prefix="plate_detector",
                suffix="input",
                draw_objects=None,
                draw_type=None
            )

        try:
            results = self._detect_onnx(image)
        except Exception as e:
            raise InferenceError("plate_detector", e)
        
        # Process the results to extract plate corners
        day_plates = []
        night_plates = []
        
        if self.use_onnx:
            # Process ONNX results
            if 'keypoints' in results and len(results['keypoints']) > 0:
                for i, keypoints in enumerate(results['keypoints']):
                    if len(keypoints) >= 4:  # Ensure we have at least 4 keypoints (corners)
                        # Get class and confidence
                        class_id = results['classes'][i]
                        confidence = results['confidences'][i]
                        
                        # Get the bounding box if available
                        detection_box = None
                        if 'boxes' in results and i < len(results['boxes']):
                            box = results['boxes'][i]
                            x1, y1, x2, y2 = box
                            detection_box = [
                                int(x1) - 15,
                                int(y1) - 15,
                                int(x2) + 15,
                                int(y2) + 15
                            ]
                        
                        # Use keypoints directly without scaling
                        corners = []
                        for kp in keypoints[:4]:
                            if len(kp) >= 2:
                                x, y = kp[0], kp[1]
                                corners.append([float(x), float(y)])
                            else:
                                corners.append([0.0, 0.0])
                        
                        # Convert to numpy array for dilation
                        corners_np = np.array(corners, dtype=np.float32)
                        
                        # Apply dilation to the corners
                        dilated_corners_np = dilate_corners(corners_np, self.corner_dilation_pixels)
                        
                        # Convert back to list format
                        dilated_corners = dilated_corners_np.tolist()
                        original_corners = corners.copy()
                        
                        plate_info = {
                            "corners": dilated_corners,
                            "original_corners": original_corners,
                            "detection_box": detection_box,
                            "confidence": float(confidence)
                        }
                        
                        if class_id == 0:  # Day plate
                            day_plates.append(plate_info)
                        else:  # Night plate
                            night_plates.append(plate_info)
        else:
            # Process PyTorch YOLOv8 results
            # The model returns keypoints for each detected plate (4 corners)
            if hasattr(results, 'keypoints') and results.keypoints is not None:
                for i, keypoints in enumerate(results.keypoints.data):
                    if len(keypoints) >= 4:  # Ensure we have at least 4 keypoints (corners)
                        # Get the 4 corner points
                        corners = keypoints[:4].cpu().numpy()  # Format: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                        
                        # No scaling needed since we're using original image
                        scaled_corners = []
                        for corner in corners:
                            # Handle different possible formats of keypoint data
                            try:
                                if len(corner) >= 3:  # Format may include confidence value or other data
                                    x, y = corner[0], corner[1]
                                else:
                                    x, y = corner
                                
                                scaled_corners.append([float(x), float(y)])
                            except Exception as e:
                                # Use a default value to avoid breaking the pipeline
                                scaled_corners.append([0.0, 0.0])
                        
                        # Convert to numpy array for dilation
                        scaled_corners_np = np.array(scaled_corners, dtype=np.float32)
                        
                        # Apply dilation to the corners
                        dilated_corners_np = dilate_corners(scaled_corners_np, self.corner_dilation_pixels)
                        
                        # Convert back to list format
                        dilated_corners = dilated_corners_np.tolist()
                        
                        # Store both original and dilated corners for visualization
                        original_corners = scaled_corners.copy()
                        
                        # Get the detection box if available
                        detection_box = None
                        if hasattr(results.boxes, 'xyxy') and i < len(results.boxes.xyxy):
                            box = results.boxes.xyxy[i].cpu().numpy()
                            # Add padding to the box
                            x1, y1, x2, y2 = box
                            x1 = int(x1) - 15
                            y1 = int(y1) - 15
                            x2 = int(x2) + 15
                            y2 = int(y2) + 15
                            detection_box = [x1, y1, x2, y2]  # [x1, y1, x2, y2] format
                        
                        # Determine if it's a day plate or night plate based on the class
                        # Assuming class 0 is day plate and class 1 is night plate
                        if hasattr(results.boxes, 'cls') and i < len(results.boxes.cls):
                            plate_class = int(results.boxes.cls[i].item())
                            
                            plate_info = {
                                "corners": dilated_corners,  # Use dilated corners for processing
                                "original_corners": original_corners,  # Keep original corners for reference
                                "detection_box": detection_box,
                                "confidence": float(results.boxes.conf[i].item()) if hasattr(results.boxes, 'conf') else 0.0
                            }
                            
                            if plate_class == 0:  # Day plate
                                day_plates.append(plate_info)
                            else:  # Night plate
                                night_plates.append(plate_info)
        
        # Save debug images with both day and night plates if enabled
        if self.save_debug_images:
            # Create debug image with plate detections
            debug_img = image.copy()
            
            # Draw day plates in green
            for plate in day_plates:
                corners = np.array(plate['corners'], dtype=np.int32)
                cv2.polylines(debug_img, [corners], True, (0, 255, 0), 2)
                
                # Add "Day" label and confidence
                if len(corners) > 0:
                    x, y = corners[0]
                    cv2.putText(debug_img, f"Day ({plate['confidence']:.2f})", (int(x), int(y) - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw night plates in blue
            for plate in night_plates:
                corners = np.array(plate['corners'], dtype=np.int32)
                cv2.polylines(debug_img, [corners], True, (255, 0, 0), 2)
                
                # Add "Night" label and confidence
                if len(corners) > 0:
                    x, y = corners[0]
                    cv2.putText(debug_img, f"Night ({plate['confidence']:.2f})", (int(x), int(y) - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Save the debug image
            save_debug_image(
                image=debug_img,
                debug_dir=self.debug_images_dir,
                prefix="plate_detector",
                suffix="processed_output",
                draw_objects=None,
                draw_type=None
            )
        
        return {
            "day_plates": day_plates,
            "night_plates": night_plates
        }
    
    def _detect_onnx(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect plates using ONNX model with improved error handling.

        Args:
            image: Input image

        Returns:
            Dictionary with detection results
        """
        try:
            # Preprocess image for ONNX with proper validation
            original_h, original_w = image.shape[:2]
            target_w, target_h = 640, 640

            # Ensure input image is valid
            if len(image.shape) != 3 or image.shape[2] != 3:
                raise ValueError(f"Expected BGR image with shape (H, W, 3), got {image.shape}")

            input_tensor = self._preprocess_image(image, target_size=(target_w, target_h))

            # Validate preprocessed tensor shape
            expected_shape = (1, 3, 640, 640)
            if input_tensor.shape != expected_shape:
                raise ValueError(f"Preprocessed tensor has wrong shape. Expected {expected_shape}, got {input_tensor.shape}")

            # Ensure tensor is float32 and normalized
            if input_tensor.dtype != np.float32:
                input_tensor = input_tensor.astype(np.float32)

            # Run inference with DirectML fallback capability
            try:
                outputs = self._safe_onnx_inference(input_tensor)
            except Exception as e:
                error_msg = f"ONNX inference failed for plate_detector. "
                error_msg += f"Input tensor shape: {input_tensor.shape}, "
                error_msg += f"Input name: {self.input_name}, "
                error_msg += f"Expected input shape: [1, 3, 640, 640]. "
                error_msg += f"Original error: {str(e)}"
                raise RuntimeError(error_msg) from e

            # Validate outputs
            if not self._validate_onnx_output(outputs):
                raise RuntimeError("Invalid ONNX output from plate detector")

            # Structure to hold results similar to YOLO output format
            results = {
                'boxes': [],        # Format: [x1, y1, x2, y2]
                'classes': [],      # Class IDs
                'confidences': [],  # Confidence scores
                'keypoints': []     # Keypoints (4 corners for plates)
            }

            # Parse ONNX outputs for YOLOv8 pose detection format
            if len(outputs) > 0 and outputs[0] is not None:
                output = outputs[0]  # Main output tensor

                # Handle different output shapes
                if len(output.shape) == 3 and output.shape[0] == 1:
                    output = output[0]  # Remove batch dimension: [num_outputs, num_detections]

                # Ensure output is in the correct format [num_detections, num_outputs]
                if output.shape[0] < output.shape[1]:
                    output = output.T   # Transpose to [num_detections, num_outputs]

                # Expected format: [num_detections, 17] where 17 = 4(bbox) + 1(conf) + 12(keypoints 4*3)
                if output.shape[1] < 5:
                    print(f"Warning: Unexpected output shape {output.shape}, expected at least 5 columns")
                    return results

                # Extract detection information
                boxes = output[:, :4]           # First 4 columns: bbox coordinates (center_x, center_y, width, height)
                confidences = output[:, 4]      # 5th column: objectness score

                # Handle keypoints if available
                if output.shape[1] > 5:
                    keypoints = output[:, 5:]   # Remaining columns: keypoints
                else:
                    keypoints = np.zeros((len(boxes), 12))  # Default empty keypoints

                # Filter by confidence threshold
                valid_indices = confidences >= self.confidence_threshold

                if not np.any(valid_indices):
                    return results  # No detections above threshold

                valid_boxes = boxes[valid_indices]
                valid_confidences = confidences[valid_indices]
                valid_keypoints = keypoints[valid_indices]

                # Apply Non-Maximum Suppression (NMS) to filter overlapping detections
                if len(valid_boxes) > 0:
                    # Convert from center format to corner format for NMS
                    boxes_xyxy = []
                    for box in valid_boxes:
                        center_x, center_y, width, height = box
                        # Ensure positive dimensions
                        width = max(width, 1.0)
                        height = max(height, 1.0)
                        x1 = center_x - width / 2
                        y1 = center_y - height / 2
                        x2 = center_x + width / 2
                        y2 = center_y + height / 2
                        boxes_xyxy.append([x1, y1, x2, y2])

                    boxes_xyxy = np.array(boxes_xyxy)

                    # Apply NMS using OpenCV
                    try:
                        nms_indices = cv2.dnn.NMSBoxes(
                            boxes_xyxy.tolist(),
                            valid_confidences.tolist(),
                            self.confidence_threshold,
                            0.4  # NMS threshold
                        )

                        if len(nms_indices) > 0:
                            # Filter boxes, confidences, and keypoints based on NMS results
                            if isinstance(nms_indices, np.ndarray):
                                nms_indices = nms_indices.flatten()
                            else:
                                nms_indices = [idx[0] if isinstance(idx, (list, tuple)) else idx for idx in nms_indices]

                            valid_boxes = valid_boxes[nms_indices]
                            valid_confidences = valid_confidences[nms_indices]
                            valid_keypoints = valid_keypoints[nms_indices]
                        else:
                            # No boxes survived NMS
                            valid_boxes = np.array([])
                            valid_confidences = np.array([])
                            valid_keypoints = np.array([])
                    except Exception as e:
                        print(f"Warning: NMS failed, using all detections: {e}")
                        # Continue without NMS

                # Scale coordinates back to original image size
                scale_x = original_w / target_w
                scale_y = original_h / target_h

                for i in range(len(valid_boxes)):
                    # Convert from center format to corner format and scale
                    center_x, center_y, width, height = valid_boxes[i]

                    # Ensure positive dimensions
                    width = max(width, 1.0)
                    height = max(height, 1.0)

                    x1 = max(0, (center_x - width/2) * scale_x)
                    y1 = max(0, (center_y - height/2) * scale_y)
                    x2 = min(original_w, (center_x + width/2) * scale_x)
                    y2 = min(original_h, (center_y + height/2) * scale_y)

                    # Ensure valid box
                    if x2 > x1 and y2 > y1:
                        scaled_box = [x1, y1, x2, y2]
                        results['boxes'].append(scaled_box)
                        results['confidences'].append(float(valid_confidences[i]))
                        results['classes'].append(0)  # Assume single class for plates

                        # Scale keypoints (4 points with x,y,confidence each)
                        if len(valid_keypoints) > i:
                            kpts = valid_keypoints[i]
                            scaled_kpts = []
                            # Process keypoints in groups of 3 (x,y,conf)
                            for j in range(0, min(12, len(kpts)), 3):
                                if j+1 < len(kpts):
                                    x = max(0, min(original_w, kpts[j] * scale_x))
                                    y = max(0, min(original_h, kpts[j+1] * scale_y))
                                    scaled_kpts.append([x, y])

                            # Ensure we have 4 keypoints (corners)
                            while len(scaled_kpts) < 4:
                                scaled_kpts.append([0, 0])  # Default corner

                            results['keypoints'].append(scaled_kpts[:4])  # Limit to 4 corners
                        else:
                            # Default keypoints if none available
                            results['keypoints'].append([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])

            return results

        except Exception as e:
            print(f"Error in plate detector ONNX inference: {e}")
            # Return empty results on error
            return {
                'boxes': [],
                'classes': [],
                'confidences': [],
                'keypoints': []
            }
    
    def __call__(self, image: np.ndarray) -> Dict[str, List[Dict[str, Any]]]:
        """
        Convenience method to call detect().
        
        Args:
            image: Input image
            
        Returns:
            Detection results
        """
        return self.detect(image)
