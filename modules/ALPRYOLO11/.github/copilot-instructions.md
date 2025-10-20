# ALPR YOLO11 - AI Coding Agent Instructions

## Project Overview
This is an **Automatic License Plate Recognition (ALPR) module** for CodeProject.AI Server powered by **YOLO11 ONNX models** running on ONNX Runtime (DirectML/CUDA/CPU). The system performs a multi-stage detection flow: plate detection → character detection → OCR recognition, with optional state classification, vehicle analytics, and rolling speed estimation.

## Architecture: Multi-Stage Detection Pipeline

### Core Processing Flow (alpr/core.py)
1. **Plate Detection** → Identifies plate regions (day/night variants)
2. **4-Point Transform** → Crops & rectifies plate using corner dilation
3. **Character Detection** → Locates individual characters on plate
4. **Character Classification** → OCR recognition via `char_classifier`
5. **Character Organization** → Resolves reading order across multi-line, vertical, overlapping layouts
6. **Speed Tracking (optional)** → `VehicleSpeedTracker` correlates detections across frames for MPH output
7. **Optional analytics**: State classification & vehicle detection

**Critical**: Each stage uses a **separate ONNX model** managed by the centralized `SessionManager`.

### Session Management Pattern (alpr/YOLO/session_manager.py)
- **Singleton pattern**: `get_session_manager(default_device_id)` returns global instance
- **DirectML fallback**: Automatically retries on CPU if DirectML fails (supports both 'DmlExecutionProvider' and 'DirectMLExecutionProvider' naming variants)
- **Thread-safe**: Individual locks per model + global manager lock
- **Session lifecycle**: Created once per model path, reused across requests
- **Device targeting**: Pass `device_id` (0-3) from configuration; session manager records whether GPU or CPU is active via `is_using_cpu_only()`
- **Frame timeout handling**: Speed tracker resets frame counter if no frames received for 500ms

**When adding new models**: 
```python
# In YOLOBase subclass __init__
session_config = SessionConfig(
	model_path=path,
	use_cuda=use_cuda,
	use_directml=True,
	device_id=device_id,  # 0-3 for multi-GPU systems
)
self.session_id = self.session_manager.create_session(session_config)
```

**Note**: `modulesettings.json` uses `ENABLE_SPEED_DETECTION` but code expects `ENABLE_SPEED_CALCULATION` - both work via environment loading.

### YOLO Model Hierarchy
- `YOLOBase` (`alpr/YOLO/base.py`): Shared ONNX inference helpers, preprocessing, session lifecycle
- **Plate & character pipeline**: `PlateDetector` and `CharacterDetector` wrap YOLO11 ONNX heads (day/night plate split + char detection/OCR)
- **State classifier**: `StateClassifier` loads YOLO11 classification weights for US states
- **Vehicle analytics (optional)**: `VehicleDetector` orchestrates `VehicleDetectorYOLO` (detect) and `VehicleClassifierYOLO` (classify) when the corresponding ONNX models are supplied

**All models**:
- Use `_safe_onnx_inference()` for automatic DirectML→CPU fallback
- Share `_preprocess_image()` for BGR→RGB conversion, normalization, CHW layout
- Load class names from `<model_name>.json` companion files

## Configuration System (alpr/config.py)

### Environment Variable Pattern
All settings load via `ModuleOptions.getEnvVariable(name, default)` in `load_from_env()`.

**Tunable thresholds** (expose via modulesettings.json UIElements):
- Confidence: `PLATE_DETECTOR_CONFIDENCE` (default 0.45), `CHAR_DETECTOR_CONFIDENCE` (default 0.40 from env, field default 0.20), `CHAR_CLASSIFIER_CONFIDENCE` (default 0.40), `STATE_CLASSIFIER_CONFIDENCE` (default 0.45), `VEHICLE_*` (default 0.45)
- Plate processing: `PLATE_ASPECT_RATIO` (field default 4.0, env default 2.5: 2.5 = US, 4.75 = EU, `0` for auto), `CORNER_DILATION_PIXELS` (default 5), `CHAR_BOX_DILATION_WIDTH/HEIGHT` (default 0, 0)
- Character organization: `LINE_SEPARATION_THRESHOLD`, `VERTICAL_ASPECT_RATIO`, `OVERLAP_THRESHOLD`, `MIN_CHARS_FOR_CLUSTERING`, `HEIGHT_FILTER_THRESHOLD`, `CLUSTERING_Y_SCALE_FACTOR`
- Speed tracking: `FRAME_RATE`, `PLATE_WIDTH_INCHES`, `PLATE_HEIGHT_INCHES`, `SPEED_TRACKING_WINDOW_FRAMES`, `SPEED_MIN_TRACKING_FRAMES`, `SPEED_IOU_THRESHOLD` (default 0.15), `SPEED_CENTROID_THRESHOLD` (default 5.0 → max normalized centroid distance in plate widths)

**Feature flags**: `ENABLE_STATE_DETECTION`, `ENABLE_VEHICLE_DETECTION`, `ENABLE_SPEED_CALCULATION` (code) / `ENABLE_SPEED_DETECTION` (modulesettings.json - **inconsistency**), `SAVE_DEBUG_IMAGES`

**Critical validation**: `ALPRConfig.__post_init__()` validates thresholds, ensures model/debug paths, auto-disables vehicle detection if optional models are missing, and builds `_model_paths` dict.

## Character Organization Logic (alpr/YOLO/char_organizer.py)

**Most complex component** - handles multi-line, vertical chars, overlaps:

1. **Multi-line detection**: Checks Y-coord gaps > `reference_height * 0.4`
2. **Mixed layout detection**: Identifies patterns like "ABC" + vertical "123MD"
3. **Vertical character detection**: `height/width > VERTICAL_ASPECT_RATIO` (config default 1.5, organizer uses 1.3)
4. **Overlap handling**: IoU > `OVERLAP_THRESHOLD` triggers special grouping

**Processing paths** (order matters - first match wins):
1. Mixed layout → `_organize_mixed_layout_characters()` (horizontal + vertical chars like "ABC123MD")
2. Multi-line → `_organize_multiline_simple_and_robust()` (Y-gap detection)
3. Single line with overlaps → `_handle_overlapping_characters()` (IoU > threshold)
4. Single line → `_organize_single_line_characters()` (default path with height filtering and clustering)

**Note**: The organizer uses aspect ratio > 1.3 for vertical detection (different from config default of 1.5)

**Default thresholds**: `LINE_SEPARATION_THRESHOLD=0.6`, `VERTICAL_ASPECT_RATIO=1.5` (changed to 1.3 in organizer), `OVERLAP_THRESHOLD=0.3`, `MIN_CHARS_FOR_CLUSTERING=6`, `HEIGHT_FILTER_THRESHOLD=0.6` (changed to 0.8 in organizer), `CLUSTERING_Y_SCALE_FACTOR=3.0`. Adjust via environment variables when regional plates deviate.

**Debug images**: Enable `SAVE_DEBUG_IMAGES=True` to see `char_organizer_reading_order.jpg` with numbered boxes.

## Speed Tracking (alpr/utils/speed_tracker.py)

- `VehicleSpeedTracker` keeps a `PlateTrack` deque per license number with up to `SPEED_TRACKING_WINDOW_FRAMES` entries (default 20)
- Matching reuses tracks when either IoU ≥ `SPEED_IOU_THRESHOLD` (0.15) **or** normalized centroid distance ≤ `SPEED_CENTROID_THRESHOLD` (default 5.0, measured in plate widths)
- Frame tracking resets if no frames received for `frame_timeout_ms` (default 500ms) to handle processing gaps
- Speeds are averaged across the last five calculations to dampen jitter; results appear as `speed_mph` + `tracking_frames` in API payloads and adapter logs
- Uses perspective-corrected distance calculation from 4-point corners when available (fallback to bbox method)
- Physical dimensions come from `PLATE_WIDTH_INCHES` / `PLATE_HEIGHT_INCHES`; adjust to your region for accurate MPH
- Set `DEBUG_SPEED_CALC=True` to trace computations in the console

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

**Statistics tracking**: `_plates_detected`, `_histogram` (license plate frequency), speed telemetry logged when `speed_mph` is returned

## Debug Image System

**When `SAVE_DEBUG_IMAGES=True`**, each stage saves to `debug_images_dir`:
- `input_*.jpg` → Original image
- `plate_detector_*.jpg` → Plate bounding boxes
- `plate_crop_*.jpg` → Rectified plate regions
- `char_detector_*.jpg` → Character boxes on plate
- `char_organizer_reading_order.jpg` → Numbered reading order
- `state_classifier_*.jpg` → State prediction overlay
- `vehicle_detector_*.jpg` / `vehicle_result_*.jpg` → Vehicle detections & classifications (when enabled)
- `final_*.jpg` → Complete annotated result

**Use case**: Debugging character organization failures, plate aspect ratio issues, confidence tuning.

## Model Requirements

**All shipping models use YOLO11 ONNX weights** (`.onnx` + companion `.json` class maps):
- `plate_detector.onnx` / `.json` → Dual head (day/night) plate detection
- `char_detector.onnx` / `.json` → Character box detection
- `char_classifier.onnx` / `.json` → OCR classifier
- `state_classifier.onnx` / `.json` → US state identification

**Optional models** (not bundled):
- `vehicle_detector.onnx` + `vehicle_classifier.onnx` → Vehicle make/model pipeline. Provide these before enabling `ENABLE_VEHICLE_DETECTION`; config auto-disables if files missing.
- Custom regional models → Drop into `models/` and expose new session IDs via `ALPRConfig` + UI bindings.

**Model root**: defaults to `models/` (override via `ONNX_MODELS_DIR`). Ensure Windows paths are normalized since validation runs before sessions initialize.

## Development Workflow

### Running the Module
On Windows, prefer `requirements.windows.txt` to ensure ONNX Runtime DirectML is installed; other platforms can use the generic `requirements.txt` variant.
```powershell
# Install dependencies
pip install -r requirements.windows.txt    # Windows (DirectML)
# pip install -r requirements.txt          # Linux / macOS

# Example environment overrides (or use modulesettings.json)
$env:SAVE_DEBUG_IMAGES = "True"
$env:ENABLE_STATE_DETECTION = "True"
$env:ENABLE_SPEED_CALCULATION = "True"  # Note: modulesettings.json uses ENABLE_SPEED_DETECTION
$env:SPEED_IOU_THRESHOLD = "0.15"
$env:SPEED_CENTROID_THRESHOLD = "5.0"  # Default is 5.0 (measured in plate widths)

# Run adapter
python alpr_adapter.py
```

### Testing Changes
1. Enable debug images to visualize pipeline stages
2. Test with diverse plate types: single-line, multi-line, vertical chars
3. Capture short video sequences to validate speed tracking stability (centroid + IoU thresholds)
4. Check session manager logs for DirectML fallback behavior
5. Verify thread safety with concurrent requests (adapter uses lock)

### Adding New Detection Models
1. Create class inheriting `YOLOBase` in `alpr/YOLO/`
2. Implement `detect()` or `classify()` method using `_safe_onnx_inference()`
3. Add model path to `ALPRConfig._model_paths`
4. Add to `modulesettings.json` EnvironmentVariables + UIElements

## Common Pitfalls

1. **Character ordering failures**: Check `char_organizer.py` debug output for which path was taken. Adjust `LINE_SEPARATION_THRESHOLD` for multi-line plates.

2. **DirectML errors (0x80004005)**: Session manager auto-falls back to CPU. Check logs for "Falling back to CPU execution...".

3. **Plate aspect ratio mismatches**: US plates ≈2.5, EU plates ≈4.75. Set `PLATE_ASPECT_RATIO` per region or 0 for auto-detect. Note: field default is 4.0, env default is 2.5.

4. **Missing characters**: Lower `CHAR_DETECTOR_CONFIDENCE` (env default 0.40, field default 0.20) and/or raise `CHAR_BOX_DILATION_*` (defaults 0, 0) to capture tight crops. Check `plate_crop_*.jpg` + `char_detector_*.jpg` for truncation.

5. **Speed stuck at `None`**: Increase `SPEED_TRACKING_WINDOW_FRAMES` or relax `SPEED_CENTROID_THRESHOLD` (default 5.0) for fast-moving vehicles; ensure sequential frames share track IDs. Note: tracks reset if no frames received for 500ms.

6. **Memory leaks**: Ensure `cleanup()` is called on ALPRSystem and session manager. Adapter handles this in `__del__()`.

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
