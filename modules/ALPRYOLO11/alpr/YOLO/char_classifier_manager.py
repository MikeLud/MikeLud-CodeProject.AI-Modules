"""
Character classification manager for license plates.
Handles character recognition and generates alternative plate readings.
"""
import numpy as np
import logging
from typing import List, Dict, Any, Tuple
import itertools

# Set up logger for this module
logger = logging.getLogger(__name__)


class CharacterClassifierManager:
    """
    Manages character classification and generates alternative plate readings.
    """
    
    def __init__(self, char_classifier):
        """
        Initialize the character classifier manager.
        
        Args:
            char_classifier: The character classifier instance
        """
        self.char_classifier = char_classifier
    
    def classify_character(self, char_image: np.ndarray) -> List[Tuple[str, float]]:
        """
        Classify a character using OCR and return top predictions.

        Args:
            char_image: Character image to classify

        Returns:
            List of tuples containing (character, confidence) for top predictions

        Raises:
            InferenceError: If classification fails
        """
        return self.char_classifier.classify(char_image)
    
    def generate_top_plates(self, 
                           char_results: List[Dict[str, Any]], 
                           max_combinations: int = 5) -> List[Dict[str, Any]]:
        """
        Generate multiple possible license plate combinations using top character predictions.
        
        Args:
            char_results: List of character results with top_predictions
            max_combinations: Maximum number of combinations to return
            
        Returns:
            List of alternative plate combinations with plate number and confidence
        """
        if not char_results:
            return []
        
        # Identify positions with uncertain character predictions
        uncertain_positions = []
        for i, char_result in enumerate(char_results):
            top_preds = char_result.get("top_predictions", [])
            
            # If we have at least 2 predictions with good confidence
            if len(top_preds) >= 2 and top_preds[1][1] >= 0.01:
                confidence_diff = top_preds[0][1] - top_preds[1][1]
                uncertain_positions.append((i, confidence_diff))
        
        # Sort by smallest confidence difference (most uncertain first)
        uncertain_positions.sort(key=lambda x: x[1])
        
        # Create base plate using top1 predictions
        base_plate = ''.join(cr["char"] for cr in char_results)
        base_confidence = sum(cr["confidence"] for cr in char_results) / len(char_results) if char_results else 0.0
        
        # Start with the base plate
        combinations = [{"plate": base_plate, "confidence": base_confidence}]
        
        # Generate alternative plates by substituting at uncertain positions
        for pos_idx, _ in uncertain_positions[:min(3, len(uncertain_positions))]:
            char_result = char_results[pos_idx]
            top_preds = char_result.get("top_predictions", [])[1:3]  # Use 2nd and 3rd predictions
            
            # Generate new plates by substituting at this position
            new_combinations = []
            for existing in combinations:
                for alt_char, alt_conf in top_preds:
                    if alt_conf >= 0.02:
                        plate_chars = list(existing["plate"])
                        if pos_idx < len(plate_chars):
                            # Calculate new confidence
                            old_char_conf = char_results[pos_idx]["confidence"]
                            plate_chars[pos_idx] = alt_char
                            
                            # Adjust overall confidence
                            # Reduce confidence proportional to the substitution
                            confidence_adjustment = (alt_conf - old_char_conf) / len(char_results)
                            new_confidence = existing["confidence"] + confidence_adjustment
                            
                            new_plate = ''.join(plate_chars)
                            
                            # Avoid duplicates
                            if not any(c["plate"] == new_plate for c in combinations + new_combinations):
                                new_combinations.append({
                                    "plate": new_plate,
                                    "confidence": max(0.0, new_confidence)
                                })
            
            # Merge new combinations
            combinations.extend(new_combinations)
            
            # Limit total combinations
            if len(combinations) >= max_combinations * 2:
                break
        
        # Sort by confidence and return top combinations
        combinations.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Format confidence as percentage
        for combo in combinations[:max_combinations]:
            combo["confidence"] = round(combo["confidence"], 4)
        
        return combinations[:max_combinations]
    
    def process_characters_for_plate(self, organized_chars) -> tuple:
        """
        Process organized characters to generate plate reading and alternatives.
        
        Args:
            organized_chars: List of organized character detections
            
        Returns:
            Tuple of (char_results, license_number, avg_confidence, top_plates)
        """
        # Classify each character
        char_results = []
        for char_info in organized_chars:
            top_chars = self.classify_character(char_info["image"])
            char_results.append({
                "char": top_chars[0][0] if top_chars else "?",
                "confidence": top_chars[0][1] if top_chars else 0.0,
                "top_predictions": top_chars,
                "box": char_info["box"]
            })
        
        # Construct the license number by concatenating the characters
        license_number = ''.join(cr["char"] for cr in char_results)
        avg_confidence = sum(cr["confidence"] for cr in char_results) / len(char_results) if char_results else 0.0
        
        # Generate alternative plate combinations
        top_plates = self.generate_top_plates(char_results)
        
        return char_results, license_number, avg_confidence, top_plates
