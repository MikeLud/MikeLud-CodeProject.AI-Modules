"""
Image processing utilities for the ALPR system.
"""
import os
import cv2
import numpy as np
from datetime import datetime
from typing import List, Tuple, Optional, Union
from ..exceptions import ImageProcessingError


def letterbox_resize(image: np.ndarray, target_size: Tuple[int, int], fill_value: int = 114) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Resize image with letterboxing to maintain aspect ratio.
    
    Args:
        image: Input image as numpy array
        target_size: Target size as (width, height)
        fill_value: Fill value for padding (default 114 for gray)
        
    Returns:
        Tuple of (resized_image, scale_factor, (pad_x, pad_y))
        
    Raises:
        ImageProcessingError: If resizing fails
    """
    try:
        target_width, target_height = target_size
        h, w = image.shape[:2]
        
        # Calculate scaling factor to fit image within target size
        scale = min(target_width / w, target_height / h)
        
        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create letterboxed image with padding
        letterboxed = np.full((target_height, target_width, 3), fill_value, dtype=np.uint8)
        
        # Handle grayscale images
        if len(image.shape) == 2:
            letterboxed = np.full((target_height, target_width), fill_value, dtype=np.uint8)
            
        # Calculate padding offsets to center the image
        pad_x = (target_width - new_w) // 2
        pad_y = (target_height - new_h) // 2
        
        # Place resized image in center of letterboxed image
        if len(image.shape) == 2:
            letterboxed[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        else:
            letterboxed[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
            
        return letterboxed, scale, (pad_x, pad_y)
        
    except Exception as e:
        raise ImageProcessingError("letterbox_resize", e)


def resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """
    Resize an image to the specified dimensions.
    
    Args:
        image: Input image as numpy array
        size: Target size as (width, height)
        
    Returns:
        Resized image
        
    Raises:
        ImageProcessingError: If resizing fails
    """
    try:
        return cv2.resize(image, size)
    except Exception as e:
        raise ImageProcessingError("resize", e)


def order_points(points: np.ndarray) -> np.ndarray:
    """
    Order points in the sequence: top-left, top-right, bottom-right, bottom-left.
    
    Args:
        points: Array of points (x, y)
        
    Returns:
        Ordered points
        
    Raises:
        ImageProcessingError: If ordering fails
    """
    try:
        # Initialize a list of coordinates that will be ordered
        rect = np.zeros((4, 2), dtype=np.float32)
        
        # The top-left point will have the smallest sum
        # The bottom-right point will have the largest sum
        s = points.sum(axis=1)
        rect[0] = points[np.argmin(s)]
        rect[2] = points[np.argmax(s)]
        
        # Now compute the difference between the points
        # The top-right point will have the smallest difference
        # The bottom-left point will have the largest difference
        diff = np.diff(points, axis=1)
        rect[1] = points[np.argmin(diff)]
        rect[3] = points[np.argmax(diff)]
        
        return rect
    except Exception as e:
        raise ImageProcessingError("order_points", e)


def four_point_transform(
    image: np.ndarray, 
    corners: Union[List[List[float]], np.ndarray], 
    aspect_ratio: Optional[float] = None
) -> np.ndarray:
    """
    Apply a 4-point perspective transform to extract a region (e.g., license plate).
    If aspect_ratio is set, the output will have that aspect ratio with fixed height.
    
    Args:
        image: Original image
        corners: List of 4 corner points [x, y]
        aspect_ratio: Optional width/height ratio for the output
        
    Returns:
        Warped image of the region
        
    Raises:
        ImageProcessingError: If transformation fails
    """
    try:
        # Convert corners to numpy array
        corners = np.array(corners, dtype=np.float32)
        
        # Ensure we have exactly 4 points
        if corners.shape[0] != 4:
            if corners.shape[0] > 4:
                corners = corners[:4]
            else:
                # Not enough points, pad with zeros
                padded_corners = np.zeros((4, 2), dtype=np.float32)
                padded_corners[:corners.shape[0]] = corners
                corners = padded_corners
                
        # Order the points correctly
        rect = order_points(corners)
        (tl, tr, br, bl) = rect
        
        # Compute the width of the new image
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        max_width = max(int(widthA), int(widthB))
        
        # Compute the height of the new image
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_height = max(int(heightA), int(heightB))
        
        # Create output dimensions
        output_width = max_width
        output_height = max_height
        
        # Apply aspect ratio if specified (width/height)
        if aspect_ratio is not None and aspect_ratio > 0:
            # Keep height fixed and calculate width based on the desired aspect ratio
            output_width = int(output_height * aspect_ratio)
        
        # Ensure dimensions are at least 1 pixel
        output_width = max(1, output_width)
        output_height = max(1, output_height)
        
        # Construct the set of destination points for the transform
        dst = np.array([
            [0, 0],
            [output_width - 1, 0],
            [output_width - 1, output_height - 1],
            [0, output_height - 1]
        ], dtype=np.float32)
        
        # Compute the perspective transform matrix
        M = cv2.getPerspectiveTransform(rect, dst)
        
        # Apply the transformation
        warped = cv2.warpPerspective(image, M, (output_width, output_height))
        
        return warped
    except Exception as e:
        raise ImageProcessingError("four_point_transform", e)


def dilate_corners(
    corners: np.ndarray, 
    dilation_pixels: int = 5
) -> np.ndarray:
    """
    Dilate the corners by moving them outward from the centroid.
    
    Args:
        corners: Numpy array of shape (4, 2) containing the corner coordinates
        dilation_pixels: Number of pixels to dilate the corners
        
    Returns:
        Dilated corners as a numpy array of the same shape
        
    Raises:
        ImageProcessingError: If dilation fails
    """
    try:
        if dilation_pixels <= 0:
            return corners.copy()
            
        # Calculate the centroid
        centroid = np.mean(corners, axis=0)
        
        # Create a copy of the corners that we will modify
        dilated_corners = corners.copy()
        
        # For each corner, move it away from the centroid
        for i in range(len(corners)):
            # Vector from centroid to corner
            vector = corners[i] - centroid
            
            # Normalize the vector
            vector_length = np.sqrt(np.sum(vector**2))
            if vector_length > 0:  # Avoid division by zero
                unit_vector = vector / vector_length
                
                # Extend the corner by the dilation amount in the direction of the unit vector
                dilated_corners[i] = corners[i] + unit_vector * dilation_pixels
        
        return dilated_corners
    except Exception as e:
        raise ImageProcessingError("dilate_corners", e)


def dilate_char_box(
    box: List[int],
    dilation_width: int = 0,
    dilation_height: int = 0,
    image_size: Tuple[int, int] = None
) -> List[int]:
    """
    Dilate a character bounding box with independent width and height dilation.
    
    Args:
        box: Character bounding box in format [x1, y1, x2, y2]
        dilation_width: Number of pixels to dilate horizontally on each side
        dilation_height: Number of pixels to dilate vertically on each side
        image_size: Optional (width, height) of the image to ensure box stays within bounds
        
    Returns:
        Dilated bounding box as [x1, y1, x2, y2]
        
    Raises:
        ImageProcessingError: If dilation fails
    """
    try:
        # Extract original coordinates
        x1, y1, x2, y2 = box
        
        # Apply dilation
        x1_new = x1 - dilation_width
        y1_new = y1 - dilation_height
        x2_new = x2 + dilation_width
        y2_new = y2 + dilation_height
        
        # Ensure box coordinates remain valid (non-negative width and height)
        if x1_new >= x2_new:
            mid_x = (x1 + x2) / 2
            x1_new = mid_x - 1
            x2_new = mid_x + 1
            
        if y1_new >= y2_new:
            mid_y = (y1 + y2) / 2
            y1_new = mid_y - 1
            y2_new = mid_y + 1
        
        # Ensure box stays within image bounds if image_size is provided
        if image_size is not None:
            img_width, img_height = image_size
            x1_new = max(0, x1_new)
            y1_new = max(0, y1_new)
            x2_new = min(img_width, x2_new)
            y2_new = min(img_height, y2_new)
        
        return [int(x1_new), int(y1_new), int(x2_new), int(y2_new)]
        
    except Exception as e:
        raise ImageProcessingError("dilate_char_box", e)


def save_debug_image(
    image: np.ndarray,
    debug_dir: str,
    prefix: str,
    suffix: str = "",
    draw_objects: List[dict] = None,
    draw_type: str = None
) -> str:
    """
    Save an image for debugging purposes with optional object annotations.
    
    Args:
        image: Input image as numpy array (BGR format)
        debug_dir: Directory to save debug images
        prefix: Prefix for the filename (usually the model name)
        suffix: Optional suffix for the filename
        draw_objects: Optional list of objects to annotate (e.g., plates, characters, vehicles)
        draw_type: Type of objects to draw ('plates', 'characters', 'vehicles')
        
    Returns:
        Path to the saved image
        
    Raises:
        ImageProcessingError: If saving fails
    """
    try:
        # Create a copy of the image to avoid modifying the original
        debug_image = image.copy()
        
        # Draw detected objects if provided
        if draw_objects and draw_type:
            if draw_type == 'plates':
                # Draw license plates
                for plate in draw_objects:
                    if 'corners' in plate:
                        # Draw plate corners
                        corners = np.array(plate['corners'], dtype=np.int32)
                        cv2.polylines(debug_image, [corners], True, (0, 255, 0), 2)
                        
                        # Add plate number if available
                        if 'license_number' in plate:
                            # Get top-left corner position
                            x = int(np.min(corners[:, 0]))
                            y = int(np.min(corners[:, 1])) - 10
                            plate_text = f"{plate['license_number']} ({plate['confidence']:.2f})"
                            cv2.putText(debug_image, plate_text, (x, y), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    elif 'detection_box' in plate:
                        # Draw plate box if corners not available
                        x1, y1, x2, y2 = plate['detection_box']
                        cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            elif draw_type == 'characters':
                # Draw character boxes
                for char in draw_objects:
                    if 'box' in char:
                        x1, y1, x2, y2 = char['box']
                        cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        
                        # Add character and confidence if available
                        if 'char' in char:
                            char_text = f"{char['char']} ({char['confidence']:.2f})"
                            cv2.putText(debug_image, char_text, (x1, y1 - 5), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            elif draw_type == 'vehicles':
                # Draw vehicle boxes
                for vehicle in draw_objects:
                    if 'box' in vehicle:
                        x1, y1, x2, y2 = vehicle['box']
                        cv2.rectangle(debug_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        
                        # Add make/model and confidence if available
                        if 'make' in vehicle and 'model' in vehicle:
                            vehicle_text = f"{vehicle['make']} {vehicle['model']} ({vehicle['confidence']:.2f})"
                            cv2.putText(debug_image, vehicle_text, (x1, y1 - 5), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Create a unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{prefix}_{timestamp}"
        if suffix:
            filename += f"_{suffix}"
        filename += ".jpg"
        
        # Ensure the debug directory exists
        os.makedirs(debug_dir, exist_ok=True)
        
        # Save the image
        output_path = os.path.join(debug_dir, filename)
        cv2.imwrite(output_path, debug_image)
        
        return output_path
    
    except Exception as e:
        # Log the error but don't raise - debug images are not critical
        print(f"Error saving debug image: {str(e)}")
        return ""
