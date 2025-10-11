# Automatic License Plate Recognition (ALPR) Module for CodeProject.AI Server

This is an Automatic License Plate Recognition (ALPR) module for [CodeProject.AI Server](https://www.codeproject.com/Articles/5322557/CodeProject-AI-Server-AI-the-easy-way). The module can detect license plates in images, recognize characters, identify states, and detect vehicles with make/model classifications.

## Features

- License plate detection for both day and night plates
- Character recognition on license plates with **stable, deterministic ordering**
- State classification for license plates (when enabled)
- Vehicle detection and make/model classification (when enabled)
- Support for GPU acceleration via DirectML (Windows) or MPS (Apple Silicon)
- Configurable confidence thresholds and plate aspect ratios
- ONNX model format for optimized inference performance
- Debug image saving for visualizing detection pipeline stages
- Robust handling of multi-line, vertical, and mixed-layout license plates

## API Endpoint

The module provides a single API endpoint with different operation modes:

```http
POST /v1/vision/alpr
```

**Parameters:**

- `operation`: The type of analysis to perform
  - `plate`: Detect only license plates
  - `vehicle`: Detect only vehicles and their make/model
  - `full`: Complete analysis (both license plates and vehicles)
- `min_confidence`: Minimum confidence threshold for detections (0.0 to 1.0)

## Technical Details

This module uses YOLOv8 models with ONNX Runtime for various detection and recognition tasks:

- **plate_detector.onnx**: Detects license plates in the image
- **state_classifier.onnx**: Identifies the US state for a license plate
- **char_detector.onnx**: Detects individual characters on the license plate
- **char_classifier.onnx**: Recognizes each character (OCR)
- **vehicle_detector.onnx**: Detects vehicles in the image
- **vehicle_classifier.onnx**: Identifies vehicle make and model

All models use the ONNX format for optimized inference performance.

## Configuration Options

The module supports several configuration options through environment variables:

### Core Settings

- `ENABLE_STATE_DETECTION`: Enable/disable state identification (default: true)
- `ENABLE_VEHICLE_DETECTION`: Enable/disable vehicle detection (default: true)
- `PLATE_ASPECT_RATIO`: Set a specific aspect ratio for license plates (default: 4.0)
- `CORNER_DILATION_PIXELS`: Configure corner dilation for license plate extraction (default: 5)

### Model Configuration

- `USE_ONNX`: Always true - this module only uses ONNX models for optimized inference
- `ONNX_MODELS_DIR`: Directory path for ONNX models (default: "models/onnx")

### Debug Options

- `SAVE_DEBUG_IMAGES`: Enable/disable saving debug images (default: false)
- `DEBUG_IMAGES_DIR`: Directory path for debug images (default: "debug_images")

### Confidence Thresholds

- `PLATE_DETECTOR_CONFIDENCE`: Plate detection confidence (default: 0.45)
- `STATE_CLASSIFIER_CONFIDENCE`: State classification confidence (default: 0.45)
- `CHAR_DETECTOR_CONFIDENCE`: Character detection confidence (default: 0.40)
- `CHAR_CLASSIFIER_CONFIDENCE`: Character recognition confidence (default: 0.40)
- `VEHICLE_DETECTOR_CONFIDENCE`: Vehicle detection confidence (default: 0.45)
- `VEHICLE_CLASSIFIER_CONFIDENCE`: Vehicle classification confidence (default: 0.45)

### Character Organization

Advanced parameters for fine-tuning character ordering (rarely need adjustment):

- `LINE_SEPARATION_THRESHOLD`: Threshold for detecting multi-line plates (default: 0.6)
- `VERTICAL_ASPECT_RATIO`: Threshold for identifying vertical characters (default: 1.5)
- `OVERLAP_THRESHOLD`: IoU threshold for overlapping character detection (default: 0.3)
- `MIN_CHARS_FOR_CLUSTERING`: Minimum characters before using advanced clustering (default: 6)
- `HEIGHT_FILTER_THRESHOLD`: Height ratio for filtering characters (default: 0.6)
- `CLUSTERING_Y_SCALE_FACTOR`: Y-coordinate scaling in clustering (default: 3.0)

## Character Ordering Stability

This module implements **deterministic character ordering** to ensure 100% consistent results across multiple reads of the same plate. The system uses stable multi-key sorting with the following features:

### Key Benefits

✅ **Deterministic Results**: Same plate image always produces identical character order  
✅ **Floating-Point Safe**: Handles characters with nearly identical coordinates reliably  
✅ **Edge Case Handling**: Special logic for 2-character plates, multi-line plates, vertical characters  
✅ **Robust Sorting**: Original index used as final tiebreaker in all sorting operations  

### Supported Plate Layouts

- **Single-line horizontal** (e.g., "ABC1234"): Left-to-right ordering
- **Multi-line plates** (e.g., "ABC" over "1234"): Top-to-bottom, then left-to-right per line
- **Vertical characters** (e.g., "ABC123MD" with vertical "MD"): Proper vertical section ordering
- **Mixed layouts**: Automatic detection and handling of horizontal/vertical combinations
- **Overlapping characters**: Grouping and ordering based on spatial proximity

### Debug Features

Enable `SAVE_DEBUG_IMAGES=True` to visualize character ordering:

- `char_organizer_reading_order.jpg`: Shows numbered character sequence (1, 2, 3...)
- `char_organizer_character_flow.jpg`: Displays arrows showing reading order path
- Console output logs processing path taken (mixed layout, multiline, single-line, etc.)

For detailed technical information, see [CHARACTER_ORDERING_FIX.md](CHARACTER_ORDERING_FIX.md).

## Debug Image Support

This module includes a comprehensive debug image feature that saves intermediate results during license plate detection and recognition. This is invaluable for:

- **Troubleshooting**: Identify where detection pipeline fails
- **Visualization**: See each stage of the processing pipeline
- **Parameter Tuning**: Optimize confidence thresholds and other settings
- **Quality Assurance**: Verify model performance on specific images

### Debug Image Types

When enabled, debug images are saved with descriptive names:

- `input_*`: Original input images
- `plate_detector_*`: Plate detection results with bounding boxes
- `plate_crop_*`: Cropped license plates
- `char_detector_*`: Character detection within plates
- `char_organizer_reading_order.jpg`: Numbered character sequence visualization
- `char_organizer_character_flow.jpg`: Arrow-based reading order path
- `state_classifier_*`: State classification results
- `vehicle_detector_*`: Vehicle detection and classification
- `final_*`: Complete annotated results

**Note**: Debug images can consume significant disk space. Use only for development and debugging.

## ONNX Runtime

This module uses ONNX models exclusively for optimized inference performance.

### Benefits of ONNX

- **Performance**: Faster inference, especially on CPU and GPU
- **Compatibility**: Better hardware compatibility across platforms
- **Optimization**: Runtime optimizations for specific hardware
- **Deployment**: Lighter weight for production deployments
- **DirectML Support**: GPU acceleration on Windows without CUDA requirements

### Setup

1. Ensure ONNX models are available in `models/onnx/` directory
2. Install ONNX Runtime with DirectML: `pip install onnxruntime-directml==1.23.0`
3. Models will automatically use GPU acceleration when available

## Project Structure

```text
alpr/
├── __init__.py
├── adapter.py          # CodeProject.AI integration
├── config.py           # Configuration management
├── core.py             # Main ALPR processing pipeline
├── exceptions.py       # Custom exception classes
├── utils/
│   ├── __init__.py
│   └── image_processing.py  # Image processing utilities
└── YOLO/
    ├── __init__.py
    ├── base.py                 # Base YOLO model class
    ├── session_manager.py      # ONNX session management with DirectML fallback
    ├── plate_detector.py       # License plate detection
    ├── character_detector.py   # Character detection and recognition
    ├── char_organizer.py       # Stable character ordering logic
    ├── char_classifier_manager.py  # Character classification
    ├── state_classifier.py     # State identification
    └── vehicle_detector.py     # Vehicle detection and classification
```

## Troubleshooting

### Common Issues

#### No license plates detected

- Check image quality and lighting
- Adjust `PLATE_DETECTOR_CONFIDENCE` threshold
- Enable debug images to see detection pipeline
- Verify license plate is clearly visible and not too small

#### Poor character recognition

- Increase image resolution if possible
- Adjust `CHAR_DETECTOR_CONFIDENCE` and `CHAR_CLASSIFIER_CONFIDENCE`
- Check `PLATE_ASPECT_RATIO` setting for your region's plates
- Review debug images to see character detection boxes

#### Characters in wrong order

- This issue has been resolved with stable multi-key sorting
- If encountered, enable `SAVE_DEBUG_IMAGES=True` and review `char_organizer_reading_order.jpg`
- Check for extreme lighting conditions or unusual plate layouts
- See [CHARACTER_ORDERING_FIX.md](CHARACTER_ORDERING_FIX.md) for technical details
- Adjust character organization parameters if needed (rare)

#### Slow performance

- Enable GPU acceleration if available (DirectML on Windows)
- Check that debug image saving is disabled in production
- Reduce image size if extremely large
- Ensure ONNX Runtime DirectML is installed for GPU acceleration

#### Model loading errors

- Verify all ONNX model files exist in the models/onnx directory
- Check file permissions
- Ensure sufficient disk space and memory
- Verify ONNX Runtime DirectML is properly installed

### Debug Mode

Enable debug mode for detailed troubleshooting:

1. Set `SAVE_DEBUG_IMAGES=True`
2. Check debug images in the configured directory
3. Review the processing pipeline step by step
4. Adjust parameters based on visual feedback

## Requirements

- **Python**: 3.8 or higher
- **ONNX Runtime DirectML**: 1.23.0 (for Windows GPU acceleration)
- **OpenCV**: 4.6.0 or higher
- **NumPy**: For numerical operations
- **CodeProject.AI SDK**: For integration with CodeProject.AI Server

## Recent Improvements

### Character Ordering Stability (v2.0)

**Problem Solved**: Rare (1 in 100) character ordering failures due to floating-point precision in sorting.

**Solution**: Implemented stable multi-key sorting with original index as final tiebreaker. All character ordering operations now use deterministic 3-level sort keys: `(primary_coordinate, secondary_coordinate, original_index)`.

**Impact**: 100% consistent character ordering across all plate types and layouts.

See [CHARACTER_ORDERING_FIX.md](CHARACTER_ORDERING_FIX.md) for complete technical documentation.
