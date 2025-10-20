#!/usr/bin/env python3
"""
Extract character class mappings from PyTorch models.

This utility script extracts class mappings from YOLO PyTorch models 
and saves them as JSON files for use with ONNX models.
"""
import os
import sys
import json
from pathlib import Path
from typing import Dict

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics package not found. Please install it with 'pip install ultralytics'")
    sys.exit(1)


def extract_character_mappings() -> None:
    """Extract character class mappings from PyTorch models."""
    print("Extracting character class mappings...")
    
    # Define models to process
    models = {
        'char_detector': 'models/char_detector.pt',
        'char_classifier': 'models/char_classifier.pt',
        'plate_detector': 'models/plate_detector.pt',
        'state_classifier': 'models/state_classifier.pt'
    }
    
    # Ensure ONNX directory exists
    onnx_dir = Path("models/onnx")
    onnx_dir.mkdir(exist_ok=True)
    
    success_count = 0
    total_count = len(models)
    
    for model_name, model_path in models.items():
        print(f"\n=== Processing {model_name} ===")
        
        if not os.path.exists(model_path):
            print(f"Warning: Model file not found - {model_path}")
            continue
            
        try:
            model = YOLO(model_path)
            
            if not hasattr(model, 'names'):
                print("Warning: No 'names' attribute found in model")
                continue
                
            print(f"Classes found: {model.names}")
            
            # Save to JSON file for ONNX
            onnx_json_path = onnx_dir / f"{model_name}.json"
            with open(onnx_json_path, 'w', encoding='utf-8') as f:
                json.dump(model.names, f, indent=2, ensure_ascii=False)
                
            print(f"✓ Saved classes to: {onnx_json_path}")
            success_count += 1
            
        except Exception as e:
            print(f"✗ Error loading {model_name}: {e}")
    
    print(f"\n=== Summary ===")
    print(f"Successfully processed: {success_count}/{total_count} models")
    
    if success_count == 0:
        print("Warning: No models were processed successfully.")
        sys.exit(1)


if __name__ == "__main__":
    extract_character_mappings()
