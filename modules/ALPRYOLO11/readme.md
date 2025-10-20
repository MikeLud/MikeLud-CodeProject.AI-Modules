# Automatic License Plate Recognition (ALPR) Module for CodeProject.AI Server

This is an Automatic License Plate Recognition (ALPR) module for [CodeProject.AI Server](https://www.codeproject.com/Articles/5322557/CodeProject-AI-Server-AI-the-easy-way). The module can detect license plates in images, recognize characters, identify states, detect vehicles with make/model classifications, and calculate vehicle speed.

## Features

- License plate detection for both day and night plates
- Character recognition on license plates with **stable, deterministic ordering**
- State classification for license plates (when enabled)
- Vehicle detection and make/model classification (when enabled)
- **Vehicle speed calculation using license plate dimensions as reference** (when enabled)
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

This module uses YOLO11 models with ONNX Runtime for various detection and recognition tasks:

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

- `ENABLE_STATE_DETECTION`: Enable/disable state identification (default: False)
- `ENABLE_VEHICLE_DETECTION`: Enable/disable vehicle detection (default: False)
- `ENABLE_SPEED_CALCULATION`: Enable/disable vehicle speed calculation (default: True)
- `PLATE_ASPECT_RATIO`: Set a specific aspect ratio for license plates (default: 2.5)
- `CORNER_DILATION_PIXELS`: Configure corner dilation for license plate extraction (default: 5)
- `CHAR_BOX_DILATION_WIDTH`: Horizontal dilation for character bounding boxes (default: 0)
- `CHAR_BOX_DILATION_HEIGHT`: Vertical dilation for character bounding boxes (default: 0)

### Speed Calculation Settings

- `FRAME_RATE`: Camera frame rate in frames per second (default: 20.0)
- `PLATE_WIDTH_INCHES`: Real-world license plate width in inches (default: 12.0 for US plates)
- `PLATE_HEIGHT_INCHES`: Real-world license plate height in inches (default: 6.0 for US plates)
- `SPEED_TRACKING_WINDOW_FRAMES`: Rolling window size in frames for tracking (default: 20)
- `SPEED_MIN_TRACKING_FRAMES`: Minimum frames needed before calculating speed (default: 2)
- `SPEED_IOU_THRESHOLD`: IoU threshold for matching plates across frames (default: 0.15)
- `SPEED_CENTROID_THRESHOLD`: Maximum normalized centroid distance for matching (default: 2.0)

### Model Configuration

- `USE_ONNX`: Always true - this module only uses ONNX models for optimized inference
- `ONNX_MODELS_DIR`: Directory path for ONNX models (default: "models")
- `USE_DIRECTML`: Enable DirectML GPU acceleration on Windows (default: True)
- `DEVICE_ID`: GPU device ID to use (0-3, default: 0)

### Debug Options

- `SAVE_DEBUG_IMAGES`: Enable/disable saving debug images (default: False)
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

✅ **Correct Spatial Ordering**: Characters sorted purely by position (X, Y coordinates)  
✅ **Detection-Order Independent**: Unaffected by YOLO model's arbitrary output order  
✅ **Edge Case Handling**: Special logic for multi-line plates, vertical characters, mixed layouts  
✅ **Coordinate-Based Sorting**: Trusts bounding box positions, not detection sequence  

### Supported Plate Layouts

- **Single-line horizontal** (e.g., "ABC1234"): Strict left-to-right by X coordinate
- **Multi-line plates** (e.g., "ABC" over "1234"): Top-to-bottom by Y, then left-to-right per line
- **Vertical characters** (e.g., "ABC123MD" with vertical "MD"): Top-to-bottom by Y coordinate
- **Mixed layouts**: Automatic detection and handling of horizontal/vertical combinations
- **Overlapping characters**: Grouping by X proximity, ordering by Y coordinate

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

## Vehicle Speed Calculation

This module includes an innovative vehicle speed calculation feature that uses the known dimensions of a license plate (12" x 6" for US standard plates) as a reference to estimate vehicle speed.

### How It Works

1. **Plate Tracking**: The system tracks license plates across consecutive frames using IoU (Intersection over Union) matching
2. **Distance Estimation**: Uses the plate's pixel width compared to its known real-world width (12 inches) to calculate the camera-to-vehicle distance
3. **Position Tracking**: Monitors the plate's centroid position changes between frames
4. **Speed Calculation**: Converts pixel movement to real-world distance and calculates speed in MPH using the frame rate

### Configuration

To enable speed calculation:

```bash
ENABLE_SPEED_CALCULATION=True
FRAME_RATE=20.0  # Your camera's frame rate
PLATE_WIDTH_INCHES=12.0  # Standard US plate width
PLATE_HEIGHT_INCHES=6.0  # Standard US plate height
SPEED_IOU_THRESHOLD=0.15  # IoU threshold for plate matching
SPEED_CENTROID_THRESHOLD=2.0  # Centroid distance threshold
```

### API Response

When speed calculation is enabled, each detected plate in the JSON response includes:

- `speed_mph`: Calculated speed in miles per hour (null if insufficient tracking data)
- `track_id`: Unique identifier for tracking the plate across frames
- `tracking_frames`: Number of frames the plate has been tracked

Example response:

```json
{
  "success": true,
  "predictions": [
    {
      "label": "ABC1234",
      "plate": "ABC1234",
      "confidence": 0.95,
      "x_min": 150,
      "y_min": 200,
      "x_max": 350,
      "y_max": 260,
      "speed_mph": 35.2,
      "track_id": 1,
      "tracking_frames": 5
    }
  ]
}

```

### Accuracy Considerations

Speed calculation accuracy depends on several factors:

- **Camera Angle**: Best results with vehicles moving toward or away from camera; less accurate for perpendicular motion
- **Frame Rate**: Higher frame rates (20+ FPS) provide more accurate measurements
- **Plate Visibility**: Requires clear plate visibility across multiple consecutive frames
- **Distance**: Most accurate at moderate distances (20-100 feet)
- **Calibration**: Uses standard US license plate dimensions (12" × 6")

For non-US plates, adjust `PLATE_WIDTH_INCHES` and `PLATE_HEIGHT_INCHES` to match your region's standard plate dimensions.

## ONNX Runtime

This module uses ONNX models exclusively for optimized inference performance.

### Benefits of ONNX

- **Performance**: Faster inference, especially on CPU and GPU
- **Compatibility**: Better hardware compatibility across platforms
- **Optimization**: Runtime optimizations for specific hardware
- **Deployment**: Lighter weight for production deployments
- **DirectML Support**: GPU acceleration on Windows without CUDA requirements

### Setup

1. Ensure ONNX models are available in `models/` directory
2. Install ONNX Runtime with DirectML: `pip install onnxruntime-directml==1.23.0`
3. Models will automatically use GPU acceleration when available

## Project Structure

```text
ALPRYOLO11/
├── alpr_adapter.py     # Main entry point and CodeProject.AI integration
├── __init__.py
├── alpr/
│   ├── __init__.py
│   ├── config.py           # Configuration management
│   ├── core.py             # Main ALPR processing pipeline
│   ├── exceptions.py       # Custom exception classes
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── image_processing.py  # Image processing utilities
│   │   └── speed_tracker.py     # Vehicle speed tracking
│   └── YOLO/
│       ├── __init__.py
│       ├── base.py                 # Base YOLO model class
│       ├── session_manager.py      # ONNX session management with DirectML fallback
│       ├── plate_detector.py       # License plate detection
│       ├── character_detector.py   # Character detection and recognition
│       ├── char_organizer.py       # Stable character ordering logic
│       ├── char_classifier_manager.py  # Character classification
│       ├── state_classifier.py     # State identification
│       └── vehicle_detector.py     # Vehicle detection and classification
├── models/              # ONNX model files
├── history/             # Documentation history
└── readme.md
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

- Verify all ONNX model files exist in the models/ directory
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
- **OpenCV**: 4.12.0 or higher
- **NumPy**: 2.2.6 or higher
- **CodeProject.AI SDK**: For integration with CodeProject.AI Server

## Recent Improvements

### Character Ordering Stability (v2.1)

**Problem Solved**: Character ordering failures (e.g., "GJD7213" read as "JDG2371") due to using YOLO detection order instead of spatial coordinates.

**Root Cause**: YOLO models output detections in arbitrary order (not left-to-right). The previous fix incorrectly used detection order (`original_index`) as a tiebreaker in sorting, which preserved the random detection sequence instead of sorting by spatial position.

**Solution**: Removed detection order from all spatial sorts. Characters now sorted purely by bounding box coordinates (X, Y). Detection order is meaningless spatially and must never influence character arrangement.

**Impact**: 100% correct spatial ordering. Characters arranged strictly by their position in the image, regardless of YOLO detection sequence.

See [CHARACTER_ORDERING_FIX.md](history/CHARACTER_ORDERING_FIX.md) for complete technical documentation.
