"""
Character organization module for license plates.
Handles sorting and organizing detected characters into proper reading order.
"""
import cv2
import numpy as np
import logging
import os
from typing import List, Dict, Any

from ..utils.image_processing import save_debug_image

# Set up logger for this module
logger = logging.getLogger(__name__)


class CharacterOrganizer:
    """
    Organizes detected characters into proper reading order.
    Handles multiple lines, vertical characters, and complex layouts.
    """
    
    def __init__(self, config):
        """
        Initialize the character organizer.
        
        Args:
            config: ALPR configuration object
        """
        self.config = config
        self.save_debug_images = config.save_debug_images
        self.debug_images_dir = config.debug_images_dir
        
        # Get configurable thresholds from config or use defaults
        self.line_separation_threshold = getattr(config, 'line_separation_threshold', 0.6)
        self.vertical_aspect_ratio = getattr(config, 'vertical_aspect_ratio', 1.3)
        self.overlap_threshold = getattr(config, 'overlap_threshold', 0.3)
        self.min_chars_for_clustering = getattr(config, 'min_chars_for_clustering', 6)
        self.height_filter_threshold = getattr(config, 'height_filter_threshold', 0.8)
        self.clustering_y_scale_factor = getattr(config, 'clustering_y_scale_factor', 3.0)
    
    def organize_characters(self, characters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Organize characters into a coherent structure, handling multiple lines and vertical characters.
        
        Implements a robust algorithm for ordering characters left-to-right and top-to-bottom,
        with special handling for various edge cases.
        
        Args:
            characters: List of character dictionaries from detect_characters()
            
        Returns:
            List of characters in reading order
        """
        if not characters:
            return []
        
        try:
            # Extract bounding box coordinates with error handling
            # Add original index to each character for stable sorting
            valid_chars = []
            boxes = []
            original_indices = []
            for i, char in enumerate(characters):
                box = char.get("box", None)
                if box and len(box) == 4 and box[2] > box[0] and box[3] > box[1]:
                    # Store original index for stable sorting
                    char_with_index = char.copy()
                    char_with_index['_original_index'] = i
                    valid_chars.append(char_with_index)
                    boxes.append([box[0], box[1], box[2], box[3]])
                    original_indices.append(i)
            
            if not valid_chars:
                return characters  # Return original if no valid characters
            
            boxes = np.array(boxes)
            original_indices = np.array(original_indices)
            
            # Debug: Print raw box coordinates
            if self.save_debug_images:
                print(f"\n*** RAW CHARACTER BOXES (before sorting) ***")
                print(f"  Total characters: {len(valid_chars)}")
                for i, (char, box) in enumerate(zip(valid_chars, boxes)):
                    x1, y1, x2, y2 = box
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    print(f"  Char {i}: box=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}], center=({center_x:.1f}, {center_y:.1f})")
                print(f"  Box format assumption: [x1, y1, x2, y2]")
                print(f"  Centers calculation: [(box[2]+box[0])/2, (box[3]+box[1])/2]")
            
            # Calculate important metrics
            # CRITICAL: Verify box format is [x1, y1, x2, y2]
            # box[0], box[2] should be X coordinates (left, right)
            # box[1], box[3] should be Y coordinates (top, bottom)
            centers = np.array([[(box[2] + box[0]) / 2, (box[3] + box[1]) / 2] for box in boxes])
            heights = boxes[:, 3] - boxes[:, 1]
            widths = boxes[:, 2] - boxes[:, 0]
            
            # Special case: only one character
            if len(valid_chars) == 1:
                return valid_chars
                
            # Special case: only two characters - simple left-to-right ordering
            # Use stable sort with Y as secondary key
            if len(valid_chars) == 2:
                # Sort by X, then Y
                sorted_indices = sorted(range(2), key=lambda i: (centers[i][0], centers[i][1]))
                return [valid_chars[sorted_indices[0]], valid_chars[sorted_indices[1]]]
            
            # Determine if there are multiple lines using adaptive thresholding
            reference_height = self._calculate_reference_height(heights)
            
            # Check for multi-line text
            is_multiline, max_gap, y_gaps = self._detect_multiline(centers, reference_height)
            
            # Detect horizontal vs vertical characters
            epsilon = 1e-6  # Avoid division by zero
            aspect_ratios = heights / (widths + epsilon)
            vertical_chars = aspect_ratios > self.vertical_aspect_ratio
            
            # Check for overlapping characters
            has_overlaps = self._detect_overlapping_characters(boxes, self.overlap_threshold)
            
            # Check for mixed horizontal/vertical layout (like ABC123MD)
            has_mixed_layout = self._detect_mixed_layout(centers, boxes, vertical_chars)
            
            # Debug logging
            if self.save_debug_images:
                print(f"*** PROCESSING PATH DEBUG ***")
                print(f"  Mixed layout: {has_mixed_layout}")
                print(f"  Is multiline: {is_multiline}")
                print(f"  Has overlaps: {has_overlaps}")
                print(f"  Character count: {len(valid_chars)}")
                print(f"  Character centers (X, Y): {[(f'{c[0]:.1f}', f'{c[1]:.1f}') for c in centers]}")
                print(f"  Character boxes: {[char.get('box', []) for char in valid_chars]}")
            
            # Handle differently based on layout complexity
            if has_mixed_layout:
                if self.save_debug_images:
                    print("  -> Using mixed layout processing")
                organized_chars = self._organize_mixed_layout_characters(
                    valid_chars, centers, boxes, vertical_chars, original_indices)
            elif is_multiline:
                if self.save_debug_images:
                    print("  -> Using multiline processing")
                organized_chars = self._organize_multiline_simple_and_robust(
                    valid_chars, centers, reference_height, original_indices)
            else:
                if self.save_debug_images:
                    print("  -> Using single-line processing")
                if has_overlaps:
                    if self.save_debug_images:
                        print("    -> Handling overlaps")
                    organized_chars = self._handle_overlapping_characters(
                        valid_chars, boxes, centers)
                else:
                    if self.save_debug_images:
                        print("    -> Standard single-line")
                    organized_chars = self._organize_single_line_characters(
                        valid_chars, centers, vertical_chars, original_indices)
            
            # Save debug image of the organized characters if enabled
            if self.save_debug_images and organized_chars:
                self._save_character_order_debug_image(characters, organized_chars)
                if is_multiline:
                    self._save_multiline_debug_info(characters, organized_chars, centers, reference_height)
            
            # Remove temporary original_index field before returning
            for char in organized_chars:
                char.pop('_original_index', None)
            
            # Debug: Show final ordering
            if self.save_debug_images:
                print(f"\n*** FINAL CHARACTER ORDER ***")
                print(f"  Sorting method used: {('mixed layout' if has_mixed_layout else 'multiline' if is_multiline else 'single-line with overlaps' if has_overlaps else 'single-line')}")
                for i, char in enumerate(organized_chars):
                    box = char.get('box', [0, 0, 0, 0])
                    center_x = (box[0] + box[2]) / 2
                    center_y = (box[1] + box[3]) / 2
                    print(f"  Position {i}: box={[f'{b:.1f}' for b in box]}, center_x={center_x:.1f}, center_y={center_y:.1f}")
                    
                # Show expected left-to-right order
                char_boxes = [(char.get('box', [0,0,0,0]), i) for i, char in enumerate(organized_chars)]
                sorted_by_x = sorted(char_boxes, key=lambda x: (x[0][0] + x[0][2])/2)
                expected_order = [idx for _, idx in sorted_by_x]
                actual_order = list(range(len(organized_chars)))
                if expected_order != actual_order:
                    print(f"  WARNING: Order mismatch!")
                    print(f"    Expected (by X coord): {expected_order}")
                    print(f"    Actual order: {actual_order}")
            
            return organized_chars
        
        except ValueError as e:
            logger.warning(f"Invalid character data in organize_characters: {e}")
            return characters
        except IndexError as e:
            logger.warning(f"Index error in character organization: {e}")
            return characters
        except AttributeError as e:
            logger.warning(f"Missing attribute in character data: {e}")
            return characters
        except Exception as e:
            logger.error(f"Unexpected error in organize_characters: {e}")
            return characters

    def _calculate_reference_height(self, heights):
        """Calculate reference height using robust statistics."""
        median_height = np.median(heights)
        q1_height = np.percentile(heights, 25)
        q3_height = np.percentile(heights, 75)
        
        iqr = q3_height - q1_height
        if iqr < 0.3 * median_height:
            return median_height
        else:
            return q1_height + 0.5 * iqr

    def _detect_multiline(self, centers, reference_height):
        """Detect if characters are arranged in multiple lines."""
        y_sorted = np.sort(centers[:, 1])
        y_gaps = y_sorted[1:] - y_sorted[:-1]
        
        if len(y_gaps) == 0:
            return False, 0, y_gaps
            
        max_gap = np.max(y_gaps)
        simple_multiline_check = max_gap > (reference_height * 0.4)
        
        if self.save_debug_images:
            print(f"*** MULTILINE DETECTION DEBUG ***")
            print(f"  Max gap: {max_gap:.2f}")
            print(f"  Reference height: {reference_height:.2f}")
            print(f"  Simple check: {simple_multiline_check} (gap > {reference_height * 0.4:.2f})")
        
        return simple_multiline_check, max_gap, y_gaps

    def _detect_overlapping_characters(self, boxes, threshold):
        """Detect if there are overlapping characters."""
        for i in range(len(boxes)):
            for j in range(i+1, len(boxes)):
                iou = self._calculate_iou(boxes[i], boxes[j])
                if iou > threshold:
                    return True
        return False
    
    def _detect_mixed_layout(self, centers, boxes, vertical_chars):
        """Detect mixed horizontal/vertical layouts like ABC123MD."""
        if len(centers) < 4:
            return False
            
        num_vertical = np.sum(vertical_chars)
        num_horizontal = len(vertical_chars) - num_vertical
        
        if num_vertical == 0 or num_horizontal == 0:
            return False
            
        if num_vertical > num_horizontal:
            return False
            
        vertical_indices = np.where(vertical_chars)[0]
        if len(vertical_indices) < 2:
            return False
            
        vertical_centers = centers[vertical_indices]
        vertical_x_coords = vertical_centers[:, 0]
        
        avg_char_width = np.mean(boxes[:, 2] - boxes[:, 0])
        x_variance = np.var(vertical_x_coords)
        
        if x_variance > (0.5 * avg_char_width) ** 2:
            return False
            
        all_x_coords = centers[:, 0]
        min_x, max_x = np.min(all_x_coords), np.max(all_x_coords)
        vertical_x_avg = np.mean(vertical_x_coords)
        
        total_width = max_x - min_x
        left_threshold = min_x + 0.3 * total_width
        right_threshold = max_x - 0.3 * total_width
        
        is_at_edge = (vertical_x_avg <= left_threshold) or (vertical_x_avg >= right_threshold)
        return is_at_edge
    
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

    def _handle_overlapping_characters(self, characters, boxes, centers):
        """Special handling for overlapping characters."""
        # Stable sort: X, then Y
        x_sorted_indices = np.lexsort((centers[:, 1], centers[:, 0]))
        
        groups = []
        current_group = [x_sorted_indices[0]]
        
        avg_width = np.mean(boxes[:, 2] - boxes[:, 0])
        threshold = 0.3 * avg_width
        
        for i in range(1, len(x_sorted_indices)):
            curr_idx = x_sorted_indices[i]
            prev_idx = x_sorted_indices[i-1]
            
            if abs(centers[curr_idx][0] - centers[prev_idx][0]) < threshold:
                current_group.append(curr_idx)
            else:
                groups.append(current_group)
                current_group = [curr_idx]
        
        if current_group:
            groups.append(current_group)
        
        organized_indices = []
        for group in groups:
            if len(group) > 1:
                # Stable sort within group: Y coordinate only
                sorted_group = sorted(group, key=lambda idx: centers[idx][1])
            else:
                sorted_group = group
            organized_indices.extend(sorted_group)
        
        return [characters[idx] for idx in organized_indices]

    def _organize_mixed_layout_characters(self, characters, centers, boxes, vertical_chars, original_indices):
        """Organize characters in mixed horizontal/vertical layouts (like ABC123MD)."""
        if len(characters) == 0:
            return []
            
        vertical_indices = np.where(vertical_chars)[0]
        horizontal_indices = np.where(~vertical_chars)[0]
        
        if len(vertical_indices) == 0:
            return self._organize_single_line_characters(characters, centers, vertical_chars, original_indices)
        
        all_x_coords = centers[:, 0]
        vertical_x_avg = np.mean(centers[vertical_indices, 0])
        horizontal_x_avg = np.mean(centers[horizontal_indices, 0]) if len(horizontal_indices) > 0 else 0
        
        vertical_on_left = vertical_x_avg < horizontal_x_avg
        
        horizontal_chars = [characters[i] for i in horizontal_indices]
        horizontal_centers = centers[horizontal_indices] if len(horizontal_indices) > 0 else np.array([])
        horizontal_orig_indices = original_indices[horizontal_indices] if len(horizontal_indices) > 0 else np.array([])
        
        if len(horizontal_chars) > 0:
            # Stable sort: X, then Y
            h_sorted_indices = np.lexsort((horizontal_centers[:, 1], horizontal_centers[:, 0]))
            sorted_horizontal = [horizontal_chars[i] for i in h_sorted_indices]
        else:
            sorted_horizontal = []
        
        vertical_chars_list = [characters[i] for i in vertical_indices]
        vertical_centers_array = centers[vertical_indices]
        vertical_orig_indices = original_indices[vertical_indices]
        
        # Stable sort: Y, then X
        v_sorted_indices = np.lexsort((vertical_centers_array[:, 0], vertical_centers_array[:, 1]))
        sorted_vertical = [vertical_chars_list[i] for i in v_sorted_indices]
        
        if vertical_on_left:
            organized_chars = sorted_vertical + sorted_horizontal
        else:
            organized_chars = sorted_horizontal + sorted_vertical
        
        return organized_chars

    def _organize_multiline_simple_and_robust(self, characters, centers, reference_height, original_indices):
        """Simplified and robust multi-line character organization."""
        if self.save_debug_images:
            print(f"*** SIMPLE METHOD CALLED with {len(characters)} characters ***")
        
        y_coords = centers[:, 1]
        y_sorted_indices = np.argsort(y_coords)
        y_sorted = y_coords[y_sorted_indices]
        
        if len(y_sorted) < 2:
            # Edge case: 0 or 1 characters - sort by X, then Y
            sorted_indices = np.lexsort((centers[:, 1], centers[:, 0]))
            return [characters[idx] for idx in sorted_indices]
        
        gaps = []
        for i in range(len(y_sorted) - 1):
            gap = y_sorted[i + 1] - y_sorted[i]
            gaps.append((gap, y_sorted[i], y_sorted[i + 1]))
        
        gaps.sort(reverse=True)
        largest_gap, gap_start, gap_end = gaps[0]
        
        if largest_gap > reference_height * 0.4:
            split_threshold = (gap_start + gap_end) / 2.0
            
            top_line = []
            bottom_line = []
            
            for i, (x, y) in enumerate(centers):
                if y <= split_threshold:
                    top_line.append(i)
                else:
                    bottom_line.append(i)
            
            # Sort each line by X coordinate (with Y for close X values)
            top_line.sort(key=lambda idx: (centers[idx][0], centers[idx][1]))
            bottom_line.sort(key=lambda idx: (centers[idx][0], centers[idx][1]))
            
            organized_chars = []
            for idx in top_line:
                organized_chars.append(characters[idx])
            for idx in bottom_line:
                organized_chars.append(characters[idx])
            
            if self.save_debug_images:
                print(f"SIMPLE METHOD: Split at Y={split_threshold:.1f}")
                print(f"  Top line indices: {top_line}")
                print(f"  Bottom line indices: {bottom_line}")
            
            return organized_chars
        else:
            if self.save_debug_images:
                print(f"SIMPLE METHOD: Single line detected")
            # Stable sort: X, then Y
            sorted_indices = np.lexsort((centers[:, 1], centers[:, 0]))
            return [characters[idx] for idx in sorted_indices]

    def _organize_single_line_characters(self, characters, centers, vertical_chars, original_indices):
        """Organize characters in a single line."""
        if len(characters) == 0:
            return []

        heights = []
        for char in characters:
            box = char.get("box", [0, 0, 0, 0])
            height = box[3] - box[1] if len(box) >= 4 else 0
            heights.append(height)
        heights = np.array(heights)

        mode_height = np.median(heights) if len(heights) > 0 else 0
        calculated_height_threshold = self.height_filter_threshold * mode_height

        if len(characters) > self.min_chars_for_clustering:
            x_coords = centers[:, 0]
            sort_indices = np.argsort(x_coords)

            filtered_indices = []
            for i, idx in enumerate(sort_indices):
                is_first = (i == 0)
                is_last = (i == len(sort_indices) - 1)
                meets_height_req = heights[idx] >= calculated_height_threshold

                if is_first or is_last or meets_height_req:
                    filtered_indices.append(idx)

            if len(filtered_indices) < len(characters) and len(filtered_indices) >= 0.8 * len(characters):
                characters = [characters[i] for i in filtered_indices]
                centers = centers[filtered_indices]
                vertical_chars = vertical_chars[filtered_indices]
                original_indices = original_indices[filtered_indices]
        
        if np.all(vertical_chars) or not np.any(vertical_chars):
            # Stable sort: X, then Y (original_index only for true ties)
            if self.save_debug_images:
                print(f"\n*** SINGLE-LINE PATH: All same orientation ***")
                print(f"  Centers before sort: {[(f'{c[0]:.1f}', f'{c[1]:.1f}') for c in centers]}")
            indices = np.lexsort((centers[:, 1], centers[:, 0]))
            if self.save_debug_images:
                print(f"  Sorted indices: {indices}")
                print(f"  X coords in sorted order: {[f'{centers[idx][0]:.1f}' for idx in indices]}")
            return [characters[idx] for idx in indices]

        num_vertical = np.sum(vertical_chars)
        
        if num_vertical <= 2:
            if self.save_debug_images:
                print(f"*** SINGLE LINE: Treating as horizontal plate (only {num_vertical} vertical chars) ***")
            # Stable sort: X, then Y
            indices = np.lexsort((centers[:, 1], centers[:, 0]))
            return [characters[idx] for idx in indices]
        
        total_chars = len(characters)

        if num_vertical / total_chars > 0.7:
            if self.save_debug_images:
                print(f"*** SINGLE LINE: Treating as rotated text ({num_vertical}/{total_chars} vertical) ***")
            # Stable sort: X, then Y
            indices = np.lexsort((centers[:, 1], centers[:, 0]))
            return [characters[idx] for idx in indices]

        if self.save_debug_images:
            print(f"*** SINGLE LINE: Using mixed character logic ({num_vertical} vertical chars) ***")
        
        x_median = np.median(centers[:, 0])

        start_vertical = []
        end_vertical = []
        horizontal = []

        for i in range(len(characters)):
            if vertical_chars[i]:
                if centers[i][0] < x_median:
                    start_vertical.append(i)
                else:
                    end_vertical.append(i)
            else:
                horizontal.append(i)

        # Stable sort by spatial coordinates only
        start_vertical.sort(key=lambda idx: (centers[idx][0], centers[idx][1]))
        horizontal.sort(key=lambda idx: (centers[idx][0], centers[idx][1]))
        end_vertical.sort(key=lambda idx: (centers[idx][0], centers[idx][1]))

        all_indices = start_vertical + horizontal + end_vertical
        return [characters[idx] for idx in all_indices]

    def _save_character_order_debug_image(self, characters, organized_chars):
        """Generate and save a debug image showing character reading order."""
        try:
            plate_with_char_order = None
            for char in characters:
                if 'plate_image' in char:
                    plate_with_char_order = char['plate_image'].copy()
                    break
            
            if plate_with_char_order is None:
                return
            
            for i, char in enumerate(organized_chars):
                if 'box' in char:
                    x1, y1, x2, y2 = char['box']
                    cv2.rectangle(plate_with_char_order, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    cv2.putText(plate_with_char_order, str(i+1), (x1, y1-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    cv2.circle(plate_with_char_order, (center_x, center_y), 2, (255, 0, 0), -1)
            
            save_debug_image(
                image=plate_with_char_order,
                debug_dir=self.debug_images_dir,
                prefix="char_organizer",
                suffix="reading_order",
                draw_objects=None,
                draw_type=None
            )
            
            if len(organized_chars) > 2:
                line_visualization = plate_with_char_order.copy()
                
                centers = []
                for char in organized_chars:
                    if 'box' in char:
                        x1, y1, x2, y2 = char['box']
                        centers.append((int((x1 + x2) / 2), int((y1 + y2) / 2)))
                
                for i in range(len(centers) - 1):
                    cv2.line(line_visualization, centers[i], centers[i+1], (0, 0, 255), 1)
                
                save_debug_image(
                    image=line_visualization,
                    debug_dir=self.debug_images_dir,
                    prefix="char_organizer",
                    suffix="character_flow",
                    draw_objects=None,
                    draw_type=None
                )
                
        except Exception as e:
            logger.debug(f"Error generating debug image: {e}")

    def _save_multiline_debug_info(self, characters, organized_chars, centers, reference_height):
        """Save detailed debug information for multiline character organization."""
        try:
            debug_info = []
            debug_info.append("=== MULTILINE DEBUG INFO ===")
            debug_info.append(f"Total characters: {len(characters)}")
            debug_info.append(f"Reference height: {reference_height:.2f}")
            
            y_coords = centers[:, 1]
            y_sorted = np.sort(y_coords)
            y_gaps = np.diff(y_sorted)
            
            debug_info.append(f"Y-coordinate range: {np.min(y_coords):.1f} to {np.max(y_coords):.1f}")
            debug_info.append(f"Max Y gap: {np.max(y_gaps):.2f}")
            debug_info.append(f"Median Y gap: {np.median(y_gaps):.2f}")
            
            debug_text = '\n'.join(debug_info)
            
            if self.debug_images_dir and os.path.exists(self.debug_images_dir):
                debug_file = os.path.join(self.debug_images_dir, "multiline_debug.txt")
                with open(debug_file, 'w') as f:
                    f.write(debug_text)
            
            logger.info(f"Multiline debug: {len(characters)} chars, max gap: {np.max(y_gaps):.2f}")
            
        except Exception as e:
            logger.debug(f"Error generating multiline debug info: {e}")
