# ALPR YOLO11 - AI Coding Agent Instructions

## Project Overview
This is an **Automatic License Plate Recognition (ALPR) module** for CodeProject.AI Server using YOLOv8 models with ONNX Runtime. The system performs multi-stage detection: plate detection → character detection → OCR recognition, with optional state classification and vehicle detection.

## Architecture: Multi-Stage Detection Pipeline

### Core Processing Flow (alpr/core.py)
1. **Plate Detection** → Identifies plate regions (day/night variants)
2. **4-Point Transform** → Crops & rectifies plate using corner dilation
3. **Character Detection** → Locates individual characters on plate
4. **Character Classification** → OCR recognition via char_classifier
5. **Character Organization** → Complex left-to-right, top-to-bottom ordering
6. **Optional**: State classification & vehicle detection

**Critical**: Each stage uses a **separate ONNX model** managed by the centralized `SessionManager`.

### Session Management Pattern (alpr/YOLO/session_manager.py)
- **Singleton pattern**: `get_session_manager()` returns global instance
- **DirectML fallback**: Automatically retries on CPU if DirectML fails
- **Thread-safe**: Individual locks per model + global manager lock
- **Session lifecycle**: Created once per model path, reused across requests

**When adding new models**: 
```python
# In YOLOBase subclass __init__
session_config = SessionConfig(model_path=path, use_cuda=use_cuda, use_directml=True)
self.session_id = self.session_manager.create_session(session_config)
```

### YOLO Model Hierarchy
- `YOLOBase` (alpr/YOLO/base.py): Shared ONNX inference, preprocessing, session management
- **Detection models** inherit `YOLOBase`: `PlateDetector`, `CharDetector`, `VehicleDetectorYOLO`
- **Classification models** inherit `YOLOBase`: `StateClassifier`, `CharClassifier`, `VehicleClassifierYOLO`

**All models**:
- Use `_safe_onnx_inference()` for automatic DirectML→CPU fallback
- Use `_preprocess_image()` for BGR→RGB, normalization, CHW format
- Load class names from `<model_name>.json` companion files

## Configuration System (alpr/config.py)

### Environment Variable Pattern
All settings load via `ModuleOptions.getEnvVariable(name, default)` in `load_from_env()`.

**Tunable thresholds** (expose via modulesettings.json UIElements):
- Confidence: `PLATE_DETECTOR_CONFIDENCE`, `CHAR_DETECTOR_CONFIDENCE`, etc.
- Processing: `PLATE_ASPECT_RATIO` (2.5=US, 4.75=EU), `CORNER_DILATION_PIXELS`
- Character organization: `LINE_SEPARATION_THRESHOLD`, `VERTICAL_ASPECT_RATIO`, `OVERLAP_THRESHOLD`

**Feature flags**: `ENABLE_STATE_DETECTION`, `ENABLE_VEHICLE_DETECTION`, `SAVE_DEBUG_IMAGES`

**Critical validation**: `ALPRConfig.__post_init__()` validates all paths/thresholds and builds `_model_paths` dict.

## Character Organization Logic (alpr/YOLO/char_organizer.py)

**Most complex component** - handles multi-line, vertical chars, overlaps:

1. **Multi-line detection**: Checks Y-coord gaps > `reference_height * 0.4`
2. **Mixed layout detection**: Identifies patterns like "ABC" + vertical "123MD"
3. **Vertical character detection**: `height/width > VERTICAL_ASPECT_RATIO` (default 1.5)
4. **Overlap handling**: IoU > `OVERLAP_THRESHOLD` triggers special grouping

**Processing paths** (order matters):
- Mixed layout → `_organize_mixed_layout_characters()`
- Multi-line → `_organize_multiline_simple_and_robust()`
- Single line with overlaps → `_handle_overlapping_characters()`
- Single line → `_organize_single_line_characters()`

**Debug images**: Enable `SAVE_DEBUG_IMAGES=True` to see `char_organizer_reading_order.jpg` with numbered boxes.

## CodeProject.AI Integration (alpr_adapter.py)

### ModuleRunner Pattern
- Inherits `codeproject_ai_sdk.ModuleRunner`
- Entry point: `if __name__ == "__main__": ALPRAdapter().start_loop()`
- Hardware detection in `initialise()`: Sets `use_CUDA`, `use_MPS`, `use_DirectML`
- Thread-safe processing: Uses `_processing_lock` for statistics tracking

### Request Handling
- Route: `/v1/vision/alpr` (defined in modulesettings.json)
- Parameters: `operation` (plate/vehicle/full), `min_confidence`
- Response format: `{"success": bool, "predictions": [...], "inferenceMs": int, "processMs": int}`

**Statistics tracking**: `_plates_detected`, `_histogram` (license plate frequency)

## Debug Image System

**When `SAVE_DEBUG_IMAGES=True`**, each stage saves to `debug_images_dir`:
- `input_*.jpg` → Original image
- `plate_detector_*.jpg` → Plate bounding boxes
- `plate_crop_*.jpg` → Rectified plate regions
- `char_detector_*.jpg` → Character boxes on plate
- `char_organizer_reading_order.jpg` → Numbered reading order
- `state_classifier_*.jpg` → State prediction overlay
- `final_*.jpg` → Complete annotated result

**Use case**: Debugging character organization failures, plate aspect ratio issues, confidence tuning.

## Model Requirements

**All models are ONNX format** (`.onnx` files + `.json` class mappings):
- `plate_detector.onnx` → Day/night plate detection
- `char_detector.onnx` → Locates characters on plate
- `char_classifier.onnx` → Alphanumeric OCR
- `state_classifier.onnx` → US state identification
- `vehicle_detector.onnx` + `vehicle_classifier.onnx` → Make/model (optional)

**Location**: `models/` directory (set via `ONNX_MODELS_DIR` env var).

## Development Workflow

### Running the Module
```powershell
# Install dependencies
pip install -r requirements.txt

# Set environment variables (or use modulesettings.json)
$env:SAVE_DEBUG_IMAGES = "True"
$env:ENABLE_STATE_DETECTION = "True"

# Run adapter
python alpr_adapter.py
```

### Testing Changes
1. Enable debug images to visualize pipeline stages
2. Test with diverse plate types: single-line, multi-line, vertical chars
3. Check session manager logs for DirectML fallback behavior
4. Verify thread safety with concurrent requests (adapter uses lock)

### Adding New Detection Models
1. Create class inheriting `YOLOBase` in `alpr/YOLO/`
2. Implement `detect()` or `classify()` method using `_safe_onnx_inference()`
3. Add model path to `ALPRConfig._model_paths`
4. Add to `modulesettings.json` EnvironmentVariables + UIElements

## Common Pitfalls

1. **Character ordering failures**: Check `char_organizer.py` debug output for which path was taken. Adjust `LINE_SEPARATION_THRESHOLD` for multi-line plates.

2. **DirectML errors (0x80004005)**: Session manager auto-falls back to CPU. Check logs for "Falling back to CPU execution...".

3. **Plate aspect ratio mismatches**: US plates ≈2.5, EU plates ≈4.75. Set `PLATE_ASPECT_RATIO` per region or 0 for auto-detect.

4. **Missing characters**: Lower `CHAR_DETECTOR_CONFIDENCE` (default 0.40). Check `plate_crop_*.jpg` for quality issues.

5. **Memory leaks**: Ensure `cleanup()` is called on ALPRSystem and session manager. Adapter handles this in `__del__()`.

## File Organization Conventions

- **No relative imports**: Use absolute imports from `alpr.` package root
- **Logging**: Use `codeproject_ai_sdk.LogMethod` in adapter layer, plain `print()` in YOLO classes
- **Error handling**: Exceptions in `alpr/exceptions.py`, catch in `alpr_adapter.py` and return `{"success": False, "error": "..."}`
- **Type hints**: All public methods use type annotations from `typing`

## Platform-Specific Notes

- **Windows**: DirectML GPU acceleration via `onnxruntime-directml==1.23.0`
- **Linux**: Set `USE_CUDA=True` for NVIDIA GPUs
- **macOS**: MPS detection in adapter (Apple Silicon GPU)

**Critical**: `requirements.txt` vs `requirements.windows.txt` - Windows uses DirectML package specifically.
