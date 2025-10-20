"""
Core ALPR system that coordinates the detection and recognition pipeline.
"""
import os
import cv2
import numpy as np
import time

from typing import Dict, List, Any
from .config import ALPRConfig
from .YOLO.plate_detector import PlateDetector
from .YOLO.character_detector import CharacterDetector
from .YOLO.state_classifier import StateClassifier
from .YOLO.vehicle_detector import VehicleDetector
from .utils.image_processing import four_point_transform, save_debug_image
from .utils.speed_tracker import VehicleSpeedTracker


class ALPRSystem:
    """
    Automatic License Plate Recognition system.
    Coordinates the detection and recognition pipeline.
    """
    
    def __init__(self, config: ALPRConfig):
        """
        Initialize the ALPR system.
        
        Args:
            config: Configuration for the ALPR system
            
        Raises:
            ModelLoadingError: If any model fails to load
        """
        self.config = config
        
        # Create debug directory if needed
        if config.save_debug_images and not os.path.exists(config.debug_images_dir):
            os.makedirs(config.debug_images_dir, exist_ok=True)
        
        # Initialize detector components
        self.plate_detector = PlateDetector(config)
        self.character_detector = CharacterDetector(config)
        
        # Initialize optional components based on configuration
        self.state_classifier = None
        if config.enable_state_detection:
            self.state_classifier = StateClassifier(config)
            
        self.vehicle_detector = None
        if config.enable_vehicle_detection:
            self.vehicle_detector = VehicleDetector(config)
        
        # Initialize speed tracker if enabled
        self.speed_tracker = None
        if config.enable_speed_calculation:
            self.speed_tracker = VehicleSpeedTracker(
                frame_rate=config.frame_rate,
                plate_width_inches=config.plate_width_inches,
                plate_height_inches=config.plate_height_inches,
                tracking_window_frames=config.speed_tracking_window_frames,
                min_tracking_frames=config.speed_min_tracking_frames,
                iou_threshold=config.speed_iou_threshold,
                centroid_threshold=config.speed_centroid_threshold
            )
            print(f"Speed tracking ENABLED: frame_rate={config.frame_rate}, plate_width={config.plate_width_inches}, window={config.speed_tracking_window_frames} frames, iou_threshold={config.speed_iou_threshold}, centroid_threshold={config.speed_centroid_threshold}")
        else:
            print(f"Speed tracking DISABLED: enable_speed_calculation={config.enable_speed_calculation}")
            
        # Ensure the parent directory for saving cropped plates exists
        cropped_plate_dir = os.path.dirname(config.cropped_plate_save_path)
        if cropped_plate_dir:
            os.makedirs(cropped_plate_dir, exist_ok=True)
        
    def detect_license_plates(self, image: np.ndarray) -> Dict[str, List[Dict[str, Any]]]:
        """
        Detect license plates in the image.

        Args:
            image: Input image as numpy array (BGR format)

        Returns:
            Dictionary with 'day_plates' and 'night_plates' lists
        """
        plate_detections = self.plate_detector.detect(image)
        
        # Save debug image if enabled
        if self.config.save_debug_images:
            # Combine day and night plates for visualization
            all_plates = []
            all_plates.extend([{'corners': plate['corners'], 'is_day_plate': True, 'confidence': plate['confidence']} 
                              for plate in plate_detections['day_plates']])
            all_plates.extend([{'corners': plate['corners'], 'is_day_plate': False, 'confidence': plate['confidence']} 
                              for plate in plate_detections['night_plates']])
            
            # Save debug image with plate detections
            save_debug_image(
                image=image,
                debug_dir=self.config.debug_images_dir,
                prefix="plate_detector",
                suffix="detection",
                draw_objects=all_plates,
                draw_type="plates"
            )
        
        return plate_detections

    def process_plate(self,
                     image: np.ndarray,
                     plate_info: Dict[str, Any],
                     is_day_plate: bool) -> Dict[str, Any]:
        """
        Process a single license plate.

        Args:
            image: Original image
            plate_info: Plate detection information
            is_day_plate: Whether it's a day plate or night plate

        Returns:
            Dictionary with plate processing results
        """
        # Extract corners
        plate_corners = plate_info["corners"]

        # Crop the license plate using 4-point transform
        plate_image = four_point_transform(
            image,
            plate_corners,
            self.config.plate_aspect_ratio
        )

        # Save the cropped plate image to the configured location
        try:
            # Always save the most recent plate (will overwrite previous ones)
            cv2.imwrite(self.config.cropped_plate_save_path, plate_image)
            # Note: Logging is handled by the adapter layer
        except Exception as e:
            # Log error but continue processing
            # Note: Logging is handled by the adapter layer
            pass

        # Save debug image of the cropped plate if enabled
        if self.config.save_debug_images:
            plate_type = "day" if is_day_plate else "night"
            save_debug_image(
                image=plate_image,
                debug_dir=self.config.debug_images_dir,
                prefix="plate_crop",
                suffix=f"{plate_type}_plate",
                draw_objects=None,
                draw_type=None
            )


        # Initialize result dictionary with basic information
        plate_result = {
            "type": "day" if is_day_plate else "night",
            "corners": plate_corners,
            "confidence": plate_info["confidence"],
            "is_day_plate": is_day_plate,
        }

        # Include original corners if available
        if "original_corners" in plate_info:
            plate_result["original_corners"] = plate_info["original_corners"]

        # Include detection box if available
        if "detection_box" in plate_info:
            plate_result["detection_box"] = plate_info["detection_box"]

        # If it's a day plate, also determine the state
        if is_day_plate and self.state_classifier:
            state_result = self.state_classifier.classify(plate_image)
            plate_result["state"] = state_result["state"]
            plate_result["state_confidence"] = state_result["confidence"]

            # Save debug image with state classification result if enabled
            if self.config.save_debug_images:
                # Add state information to the plate image as text
                state_debug_img = plate_image.copy()
                state_text = f"State: {state_result['state']} ({state_result['confidence']:.2f})"
                cv2.putText(state_debug_img, state_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                save_debug_image(
                    image=state_debug_img,
                    debug_dir=self.config.debug_images_dir,
                    prefix="state_classifier",
                    suffix=f"{state_result['state']}",
                    draw_objects=None,
                    draw_type=None
                )

        # Detect and recognize characters in the plate
        char_result = self.character_detector.process_plate(plate_image)
        
        # Save debug image with character detections if enabled
        if self.config.save_debug_images:
            save_debug_image(
                image=plate_image,
                debug_dir=self.config.debug_images_dir,
                prefix="char_detector",
                suffix=f"{char_result['license_number']}",
                draw_objects=char_result['characters'],
                draw_type="characters"
            )
        
        # Update plate result with character recognition data
        plate_result.update({
            "characters": char_result["characters"],
            "license_number": char_result["license_number"],
            "confidence": char_result["confidence"],
            "top_plates": char_result["top_plates"]
        })
        
        # Store plate dimensions for debugging
        if plate_image is not None:
            h, w = plate_image.shape[:2]
            plate_result["dimensions"] = {"width": w, "height": h, "actual_ratio": w/h if h > 0 else 0}
        
        return plate_result
    
    def process_image(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Process an image to detect and recognize license plates, vehicle make/model.

        Args:
            image: Input image

        Returns:
            Dictionary with processing results
        """
        # Create a copy of the image to avoid modifying the original
        image_copy = image.copy()

        start_time = time.perf_counter()

        # Detect license plates in the image
        plate_detection = self.detect_license_plates(image_copy)

        # Initialize results
        results = {
            "day_plates": [],
            "night_plates": [],
            "vehicles": []
        }

        # Process plates sequentially (no multi-threading)
        # Process day plates
        for plate in plate_detection["day_plates"]:
            try:
                plate_result = self.process_plate(image_copy, plate, True)
                results["day_plates"].append(plate_result)
            except Exception as e:
                # Error handling is done by the adapter layer
                continue

        # If day plates were detected and vehicle detection is enabled,
        # detect and classify vehicles
        # Process night plates
        for plate in plate_detection["night_plates"]:
            try:
                plate_result = self.process_plate(image_copy, plate, False)
                results["night_plates"].append(plate_result)
            except Exception as e:
                # Error handling is done by the adapter layer
                continue

        # If day plates were detected and vehicle detection is enabled,
        # detect and classify vehicles
        if results["day_plates"] and self.vehicle_detector:
            vehicle_results = self.vehicle_detector.detect_and_classify(image_copy)
            results["vehicles"] = vehicle_results
            
            # Save debug image with vehicle detections if enabled
            if self.config.save_debug_images and vehicle_results:
                save_debug_image(
                    image=image_copy,
                    debug_dir=self.config.debug_images_dir,
                    prefix="vehicle_detector",
                    suffix="detection",
                    draw_objects=vehicle_results,
                    draw_type="vehicles"
                )
        
        # Add timing information
        results["processing_time_ms"] = int((time.perf_counter() - start_time) * 1000)
        
        return results
    
    def detect_license_plate(self,
                            image: np.ndarray,
                            threshold: float = 0.4) -> Dict[str, Any]:
        """
        Detect license plates and prepare response for the API.

        Args:
            image: Input image as PIL Image
            threshold: Minimum confidence threshold

        Returns:
            API response with predictions
        """
        start_process_time = time.perf_counter()

        # Convert PIL Image to numpy array for OpenCV
        image_np = np.array(image)
        # Convert RGB to BGR (OpenCV format)
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:  # Color image
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Save input image for debugging if enabled
        if self.config.save_debug_images:
            save_debug_image(
                image=image_np,
                debug_dir=self.config.debug_images_dir,
                prefix="input",
                suffix="original",
                draw_objects=None,
                draw_type=None
            )

        # Process the image
        start_inference_time = time.perf_counter()

        # Process the image to find license plates
        plate_detection = self.detect_license_plates(image_np)
        # plate_detection = self.detect_license_plates(image_np)

        # Process each plate
        results = {
            "day_plates": [],
            "night_plates": []
        }

        for plate_type in ["day_plates", "night_plates"]:
            for plate_info in plate_detection[plate_type]:
                if plate_info["confidence"] >= threshold:
                    plate_result = self.process_plate(image_np, plate_info, plate_type == "day_plates")
                    if plate_result["confidence"] >= threshold:
                        results[plate_type].append(plate_result)

        inferenceMs = int((time.perf_counter() - start_inference_time) * 1000)

        # Extract plate numbers and coordinates for client response
        plates = []
        for plate_type in ["day_plates", "night_plates"]:
            for plate in results[plate_type]:
                # Only include plates with confidence above threshold
                if plate["confidence"] >= threshold:
                    # Use detection_box if available, otherwise calculate from corners
                    if "detection_box" in plate and plate["detection_box"] is not None:
                        # If detection_box is available, use it directly
                        x1, y1, x2, y2 = plate["detection_box"]
                        plate_data = {
                            "confidence": plate["confidence"],
                            "is_day_plate": plate["is_day_plate"],
                            "label": plate["license_number"],
                            "plate": plate["license_number"],
                            "x_min": x1,
                            "y_min": y1,
                            "x_max": x2,
                            "y_max": y2
                        }
                    else:
                        # Otherwise, calculate the bounding box from the corners
                        corners = plate["corners"]
                        # Convert corners to numpy array if not already
                        corners_arr = np.array(corners)
                        x_min = np.min(corners_arr[:, 0])
                        y_min = np.min(corners_arr[:, 1])
                        x_max = np.max(corners_arr[:, 0])
                        y_max = np.max(corners_arr[:, 1])
                        
                        plate_data = {
                            "confidence": plate["confidence"],
                            "is_day_plate": plate["is_day_plate"],
                            "label": plate["license_number"],
                            "plate": plate["license_number"],
                            "x_min": float(x_min),
                            "y_min": float(y_min),
                            "x_max": float(x_max),
                            "y_max": float(y_max)
                        }
                    
                    if "state" in plate:
                        plate_data["state"] = plate["state"]
                        plate_data["state_confidence"] = plate["state_confidence"]
                    
                    # Add top plate alternatives
                    if "top_plates" in plate:
                        plate_data["top_plates"] = plate["top_plates"]
                    
                    # Add corners for perspective-aware speed calculation
                    if "corners" in plate:
                        plate_data["corners"] = plate["corners"]
                        
                    plates.append(plate_data)
        
        # Apply speed tracking if enabled
        if self.speed_tracker and plates:
            # print(f"[DEBUG] Applying speed tracking to {len(plates)} plates")
            plates = self.speed_tracker.update(plates)
            # print(f"[DEBUG] After speed tracking, got {len(plates)} plates back")
            # Print speed for each plate
            for plate in plates:
                speed = plate.get("speed_mph")
                tracking_frames = plate.get("tracking_frames", 0)
                track_id = plate.get("track_id", "?")
                # print(f"[DEBUG] Plate {plate['label']}: speed={speed}, tracking_frames={tracking_frames}, track_id={track_id}")
                if speed is not None:
                    print(f"Plate {plate['label']}: {speed} mph (tracked {tracking_frames} frames)")
                else:
                    print(f"Plate {plate['label']}: calculating speed... ({tracking_frames} frames)")
        
        # Save final result image with all plates if enabled
        if self.config.save_debug_images and plates:
            # Draw all detected plates on the original image
            final_results = []
            for plate in plates:
                final_results.append({
                    'corners': [[plate['x_min'], plate['y_min']], 
                               [plate['x_max'], plate['y_min']], 
                               [plate['x_max'], plate['y_max']], 
                               [plate['x_min'], plate['y_max']]],
                    'license_number': plate['plate'],
                    'confidence': plate['confidence']
                })
                
            save_debug_image(
                image=image_np,
                debug_dir=self.config.debug_images_dir,
                prefix="final",
                suffix="result",
                draw_objects=final_results,
                draw_type="plates"
            )
        
        # Create a response message
        if len(plates) > 0:
            message = f"Found {len(plates)} license plates"
            if len(plates) <= 3:
                message += ": " + ", ".join([p["label"] for p in plates])
        else:
            message = "No license plates detected"
            
        return {
            "success": True,
            "processMs": int((time.perf_counter() - start_process_time) * 1000),
            "inferenceMs": inferenceMs,
            "predictions": plates,
            "message": message,
            "count": len(plates)
        }

    def cleanup(self):
        """
        Clean up resources when the ALPR system is no longer needed.
        This includes cleaning up all detector/classifier instances.
        """
        try:
            # Clean up individual detector instances
            if hasattr(self, 'plate_detector') and self.plate_detector:
                self.plate_detector.cleanup()

            if hasattr(self, 'character_detector') and self.character_detector:
                self.character_detector.cleanup()

            if hasattr(self, 'state_classifier') and self.state_classifier:
                self.state_classifier.cleanup()

            if hasattr(self, 'vehicle_detector') and self.vehicle_detector:
                self.vehicle_detector.cleanup()

            # Session manager cleanup is handled by the adapter layer; no call here.

            print("ALPR system cleanup completed")

        except Exception as e:
            print(f"Error during ALPR system cleanup: {e}")

    def get_session_info(self) -> Dict[str, Any]:
        """
        Get information about all ONNX sessions used by the ALPR system.

        Returns:
            Dictionary with session information for all models
        """
        session_info = {}

        # Get session info from each detector/classifier
        if hasattr(self, 'plate_detector') and self.plate_detector:
            session_info['plate_detector'] = self.plate_detector.get_session_info()

        if hasattr(self, 'character_detector') and self.character_detector:
            session_info['character_detector'] = self.character_detector.get_session_info()

        if hasattr(self, 'state_classifier') and self.state_classifier:
            session_info['state_classifier'] = self.state_classifier.get_session_info()

        if hasattr(self, 'vehicle_detector') and self.vehicle_detector:
            session_info['vehicle_detector'] = self.vehicle_detector.get_session_info()

        return session_info

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except:
            pass
