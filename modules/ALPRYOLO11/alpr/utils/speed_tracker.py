"""
Vehicle Speed Tracker for ALPR System.
Calculates vehicle speed using license plate dimensions and tracking across frames.
"""
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import deque


@dataclass
class PlateTrack:
    """Represents a tracked license plate across frames."""
    plate_number: str
    track_id: int
    positions: deque  # Store (timestamp, centroid_x, centroid_y, width_pixels, height_pixels, corners)
    last_seen: float
    speeds: deque  # Store calculated speeds for smoothing
    
    def __init__(self, plate_number: str, track_id: int, max_history: int = 20):
        self.plate_number = plate_number
        self.track_id = track_id
        self.positions = deque(maxlen=max_history)
        self.speeds = deque(maxlen=5)  # Smaller window for faster response to speed changes
        self.last_seen = time.time()
        self.frames_since_update = 0


class VehicleSpeedTracker:
    """
    Tracks license plates across frames and calculates vehicle speed.
    
    Uses the known physical dimensions of a license plate (12" x 6") to calculate
    distance and speed based on pixel measurements and frame rate.
    """
    
    def __init__(self,
                 frame_rate: float = 20.0,
                 plate_width_inches: float = 12.0,
                 plate_height_inches: float = 6.0,
                 tracking_window_frames: int = 20,
                 min_tracking_frames: int = 3,
                 iou_threshold: float = 0.15,
                 centroid_threshold: float = 5.0,
                 frame_timeout_ms: float = 500.0):
        """
        Initialize the speed tracker.
        
        Args:
            frame_rate: Camera frame rate in FPS
            plate_width_inches: Real-world license plate width in inches
            plate_height_inches: Real-world license plate height in inches
            tracking_window_frames: Rolling window size in frames for tracking
            min_tracking_frames: Minimum frames needed before calculating speed
            iou_threshold: IoU threshold for matching plates between frames
            centroid_threshold: Max normalized centroid distance for matching
            frame_timeout_ms: Reset frame_count if no frames received for this duration (ms)
        """
        self.frame_rate = frame_rate
        self.plate_width_inches = plate_width_inches
        self.plate_height_inches = plate_height_inches
        self.tracking_window_frames = tracking_window_frames
        self.min_tracking_frames = min_tracking_frames
        self.iou_threshold = iou_threshold
        self.centroid_threshold = centroid_threshold
        self.frame_timeout_ms = frame_timeout_ms
        
        # Time between frames
        self.time_per_frame = 1.0 / frame_rate
        
        # Active tracks
        self.tracks: Dict[int, PlateTrack] = {}
        self.next_track_id = 1
        self.frame_count = 0  # Track frame number for accurate timing
        self.last_frame_time = time.time()  # Track last frame timestamp for timeout detection
        
        # Conversion constants
        self.INCHES_PER_MILE = 63360.0
        self.SECONDS_PER_HOUR = 3600.0
        
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            box1, box2: [x_min, y_min, x_max, y_max]
            
        Returns:
            IoU value between 0 and 1
        """
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Calculate intersection area
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # Calculate union area
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    def _calculate_centroid_distance(self, pos1: Tuple[float, float], 
                                     pos2: Tuple[float, float]) -> float:
        """
        Calculate Euclidean distance between two centroids.
        
        Args:
            pos1, pos2: (x, y) centroids
            
        Returns:
            Distance in pixels
        """
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def _extract_plate_info(self, plate_data: Dict[str, Any]) -> Tuple[str, List[float], Tuple[float, float], float, float, List[List[float]]]:
        """
        Extract relevant information from plate detection data.
        
        Args:
            plate_data: Plate detection dictionary
            
        Returns:
            Tuple of (plate_number, bbox, centroid, width_pixels, height_pixels, corners)
        """
        plate_number = plate_data.get("license_number", plate_data.get("plate", ""))
        
        # Extract corners if available for perspective correction
        corners = plate_data.get("corners", None)
        
        # Extract bounding box
        x_min = plate_data["x_min"]
        y_min = plate_data["y_min"]
        x_max = plate_data["x_max"]
        y_max = plate_data["y_max"]
        bbox = [x_min, y_min, x_max, y_max]
        
        # Calculate centroid
        centroid_x = (x_min + x_max) / 2.0
        centroid_y = (y_min + y_max) / 2.0
        centroid = (centroid_x, centroid_y)
        
        # Calculate dimensions in pixels
        width_pixels = x_max - x_min
        height_pixels = y_max - y_min
        
        return plate_number, bbox, centroid, width_pixels, height_pixels, corners
    
    def _match_plate_to_track(self, plate_data: Dict[str, Any]) -> Optional[int]:
        """
        Match a detected plate to an existing track.
        
        Args:
            plate_data: Plate detection dictionary
            
        Returns:
            Track ID if matched, None otherwise
        """
        plate_number, bbox, centroid, _, _, _ = self._extract_plate_info(plate_data)
        
        best_match_id = None
        best_score = 0.0
        
        current_time = time.time()
        
        for track_id, track in self.tracks.items():
            # Skip expired tracks (more than window frames since last update)
            if track.frames_since_update > self.tracking_window_frames:
                continue
            
            # Get last known position
            if len(track.positions) == 0:
                continue
            
            last_pos = track.positions[-1]
            last_timestamp = last_pos[0]
            last_cx = last_pos[1]
            last_cy = last_pos[2]
            last_w = last_pos[3]
            last_h = last_pos[4]
            # Ignore corners (last_pos[5]) for matching - only used for speed calc
            
            last_bbox = [
                last_cx - last_w / 2,
                last_cy - last_h / 2,
                last_cx + last_w / 2,
                last_cy + last_h / 2
            ]
            
            # Calculate IoU
            iou = self._calculate_iou(bbox, last_bbox)
            
            # Calculate centroid distance (normalized by plate size)
            cent_dist = self._calculate_centroid_distance(centroid, (last_cx, last_cy))
            max_dimension = max(last_w, last_h)
            normalized_dist = cent_dist / max_dimension if max_dimension > 0 else float('inf')
            
            # Match by EITHER IoU OR centroid proximity (for horizontal movement)
            # This handles both stationary vehicles and vehicles moving across frame
            
            # Accept match if EITHER condition is met:
            # 1. IoU is high enough (overlapping bboxes)
            # 2. Centroid distance is small enough (moving vehicle)
            if iou >= self.iou_threshold or normalized_dist <= self.centroid_threshold:
                # Calculate score that prioritizes IoU when available, but allows centroid matching
                # Give bonus to matches that pass threshold to ensure positive score
                if iou >= self.iou_threshold:
                    score = iou + 1.0  # IoU match: score 1.0-2.0
                else:
                    score = 1.0 - (normalized_dist / self.centroid_threshold)  # Centroid match: score 0.0-1.0
                
                if score > best_score:
                    best_score = score
                    best_match_id = track_id
        
        return best_match_id
    
    def _create_new_track(self, plate_data: Dict[str, Any]) -> int:
        """
        Create a new track for a detected plate.
        
        Args:
            plate_data: Plate detection dictionary
            
        Returns:
            New track ID
        """
        plate_number, _, centroid, width_pixels, height_pixels, corners = self._extract_plate_info(plate_data)
        
        track_id = self.next_track_id
        self.next_track_id += 1
        
        track = PlateTrack(plate_number, track_id, max_history=self.tracking_window_frames)
        track.positions.append((self.frame_count, centroid[0], centroid[1], width_pixels, height_pixels, corners))
        track.last_seen = self.frame_count
        track.frames_since_update = 0
        
        self.tracks[track_id] = track
        
        return track_id
    
    def _update_track(self, track_id: int, plate_data: Dict[str, Any]):
        """
        Update an existing track with new plate detection.
        
        Args:
            track_id: Track ID to update
            plate_data: Plate detection dictionary
        """
        if track_id not in self.tracks:
            return
        
        track = self.tracks[track_id]
        _, _, centroid, width_pixels, height_pixels, corners = self._extract_plate_info(plate_data)
        
        track.positions.append((self.frame_count, centroid[0], centroid[1], width_pixels, height_pixels, corners))
        track.last_seen = self.frame_count
        track.frames_since_update = 0
    
    def _calculate_speed(self, track_id: int) -> Optional[float]:
        """
        Calculate vehicle speed for a track.
        
        Args:
            track_id: Track ID
            
        Returns:
            Speed in MPH, or None if insufficient data
        """
        if track_id not in self.tracks:
            return None
        
        track = self.tracks[track_id]
        
        # Need at least min_tracking_frames frames
        if len(track.positions) < self.min_tracking_frames:
            return None
        
        # Get first and last positions (frame_number, cx, cy, w, h, corners)
        first_pos = track.positions[0]
        last_pos = track.positions[-1]
        
        first_frame = first_pos[0]
        first_cx = first_pos[1]
        first_cy = first_pos[2]
        first_w = first_pos[3]
        first_h = first_pos[4]
        first_corners = first_pos[5] if first_pos[5] is not None else None
        
        last_frame = last_pos[0]
        last_cx = last_pos[1]
        last_cy = last_pos[2]
        last_w = last_pos[3]
        last_h = last_pos[4]
        last_corners = last_pos[5] if last_pos[5] is not None else None
        
        # Calculate frames elapsed
        frames_elapsed = last_frame - first_frame
        if frames_elapsed <= 0:
            return 0.0  # Same frame or no movement
        
        # Calculate time elapsed using frame rate
        time_elapsed = frames_elapsed / self.frame_rate
        
        # Use corners for perspective-corrected distance if available
        if first_corners and last_corners and len(first_corners) >= 4 and len(last_corners) >= 4:
            # Calculate plate width from corners for each position (use bottom edge)
            first_corners_arr = np.array(first_corners)
            last_corners_arr = np.array(last_corners)
            
            # Calculate actual plate width at each position using bottom-left to bottom-right corners
            first_plate_width_px = np.linalg.norm(first_corners_arr[3] - first_corners_arr[2])
            last_plate_width_px = np.linalg.norm(last_corners_arr[3] - last_corners_arr[2])
            
            if first_plate_width_px <= 0 or last_plate_width_px <= 0:
                return None
            
            # Calculate center of plate from corners
            first_center = np.mean(first_corners_arr, axis=0)
            last_center = np.mean(last_corners_arr, axis=0)
            
            # Calculate pixel distance between centers
            pixel_distance = np.linalg.norm(last_center - first_center)
            
            # Use average plate width for scale
            avg_plate_width_px = (first_plate_width_px + last_plate_width_px) / 2
            pixels_per_inch = avg_plate_width_px / self.plate_width_inches
            distance_inches = pixel_distance / pixels_per_inch
            avg_plate_width_pixels = avg_plate_width_px  # For debug logging
        else:
            # Fallback: use bounding box method
            avg_plate_width_pixels = np.mean([pos[3] for pos in track.positions])
            
            if avg_plate_width_pixels <= 0:
                return None
            
            # Calculate pixel distance traveled
            pixel_distance = np.sqrt((last_cx - first_cx)**2 + (last_cy - first_cy)**2)
            
            # Convert pixels to inches using average plate width as reference
            pixels_per_inch = avg_plate_width_pixels / self.plate_width_inches
            distance_inches = pixel_distance / pixels_per_inch
        
        # Check for minimal movement (stationary vehicle)
        if pixel_distance < 1.0:  # Less than 1 pixel movement
            return 0.0  # Stationary vehicle
        
        # Calculate speed in inches per second
        speed_inches_per_second = distance_inches / time_elapsed
        
        # Convert to MPH (inches/sec * miles/inch * seconds/hour)
        speed_mph = (speed_inches_per_second / self.INCHES_PER_MILE) * self.SECONDS_PER_HOUR
        
        # Store speed for smoothing
        track.speeds.append(speed_mph)
        
        # Return smoothed speed (average of recent speeds)
        return float(np.mean(track.speeds))
    
    def update(self, plates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Update tracks with new plate detections and calculate speeds.
        
        Args:
            plates: List of plate detection dictionaries
            
        Returns:
            List of plates with added speed information
        """
        # Check for frame timeout and reset if needed
        current_time = time.time()
        time_since_last_frame_ms = (current_time - self.last_frame_time) * 1000.0
        
        if time_since_last_frame_ms > self.frame_timeout_ms:
            # Timeout detected - reset frame counter and clear all tracks
            self.frame_count = 0
            self.tracks.clear()
            self.next_track_id = 1
        
        # Update last frame time
        self.last_frame_time = current_time
        
        # Increment frame counter
        self.frame_count += 1
        
        # Process each detected plate
        enriched_plates = []
        matched_tracks = set()  # Track which plates were matched this frame
        
        for plate_data in plates:
            # Create a copy to avoid modifying original
            enriched_plate = plate_data.copy()
            
            # Try to match to existing track
            track_id = self._match_plate_to_track(plate_data)
            
            if track_id is None:
                # Create new track
                track_id = self._create_new_track(plate_data)
                enriched_plate["track_id"] = track_id
                enriched_plate["speed_mph"] = None
                enriched_plate["tracking_frames"] = 1
                matched_tracks.add(track_id)
            else:
                # Update existing track
                self._update_track(track_id, plate_data)
                matched_tracks.add(track_id)
                
                # Calculate speed
                speed_mph = self._calculate_speed(track_id)
                
                enriched_plate["track_id"] = track_id
                enriched_plate["speed_mph"] = round(speed_mph, 1) if speed_mph is not None else None
                enriched_plate["tracking_frames"] = len(self.tracks[track_id].positions)
            
            enriched_plates.append(enriched_plate)
        
        # Increment frame counter for tracks that were NOT matched this frame
        for track_id, track in self.tracks.items():
            if track_id not in matched_tracks:
                track.frames_since_update += 1
        
        # Clean up expired tracks (beyond window)
        expired_tracks = [
            track_id for track_id, track in self.tracks.items()
            if track.frames_since_update > self.tracking_window_frames
        ]
        for track_id in expired_tracks:
            del self.tracks[track_id]
        
        return enriched_plates
    
    def reset(self):
        """Clear all tracks."""
        self.tracks.clear()
        self.next_track_id = 1
        self.frame_count = 0
        self.last_frame_time = time.time()
    
    def get_track_info(self) -> Dict[str, Any]:
        """
        Get information about active tracks.
        
        Returns:
            Dictionary with track information
        """
        current_time = time.time()
        
        active_tracks = []
        for track_id, track in self.tracks.items():
            if track.frames_since_update <= self.tracking_window_frames:
                speed = self._calculate_speed(track_id)
                active_tracks.append({
                    "track_id": track_id,
                    "plate_number": track.plate_number,
                    "tracking_frames": len(track.positions),
                    "speed_mph": round(speed, 1) if speed is not None else None,
                    "frames_since_update": track.frames_since_update
                })
        
        return {
            "active_tracks": len(active_tracks),
            "tracks": active_tracks
        }
