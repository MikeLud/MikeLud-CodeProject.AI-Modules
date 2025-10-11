"""
ONNX Session Manager for ALPR models.
Manages separate InferenceSession instances for each model with proper resource management.
"""
import os
import threading
import weakref
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


@dataclass
class SessionConfig:
    """Configuration for an ONNX inference session."""
    model_path: str
    use_cuda: bool = False
    use_directml: bool = True
    session_options: Optional[ort.SessionOptions] = None
    providers: Optional[List[str]] = None


class ONNXSessionManager:
    """
    Manages separate ONNX InferenceSession instances for different models.
    Provides session isolation, resource management, and DirectML fallback capabilities.
    """
    
    def __init__(self):
        """Initialize the session manager."""
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX Runtime is not available. Please install it with 'pip install onnxruntime' or 'pip install onnxruntime-directml'")
        
        # Dictionary to store sessions by model path
        self._sessions: Dict[str, ort.InferenceSession] = {}
        
        # Track which models have failed DirectML and fallen back to CPU
        self._directml_failed: Dict[str, bool] = {}
        
        # Session metadata (input/output names, shapes)
        self._session_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Threading locks for each session
        self._session_locks: Dict[str, threading.Lock] = {}
        
        # Global lock for session creation/deletion
        self._manager_lock = threading.Lock()
        
        # Keep track of available providers
        self._available_providers = ort.get_available_providers()
        
        print(f"ONNX Session Manager initialized. Available providers: {self._available_providers}")
    
    def create_session(self, config: SessionConfig) -> str:
        """
        Create a new ONNX inference session for a model.
        
        Args:
            config: Session configuration
            
        Returns:
            Session ID (model path) for referencing the session
            
        Raises:
            RuntimeError: If session creation fails
            FileNotFoundError: If model file doesn't exist
        """
        model_path = config.model_path
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        with self._manager_lock:
            # Check if session already exists
            if model_path in self._sessions:
                print(f"Session for {model_path} already exists, returning existing session")
                return model_path
            
            # Create session lock
            self._session_locks[model_path] = threading.Lock()
            
            # Initialize DirectML failure tracking
            self._directml_failed[model_path] = False
            
            try:
                session = self._create_onnx_session(config)
                self._sessions[model_path] = session
                
                # Store session metadata
                self._session_metadata[model_path] = {
                    'input_name': session.get_inputs()[0].name,
                    'output_names': [output.name for output in session.get_outputs()],
                    'input_shape': session.get_inputs()[0].shape,
                    'providers': session.get_providers()
                }
                
                print(f"Created ONNX session for {model_path} with providers: {session.get_providers()}")
                return model_path
                
            except Exception as e:
                # Clean up on failure
                self._session_locks.pop(model_path, None)
                self._directml_failed.pop(model_path, None)
                raise RuntimeError(f"Failed to create ONNX session for {model_path}: {e}")
    
    def _create_onnx_session(self, config: SessionConfig) -> ort.InferenceSession:
        """Create an ONNX runtime session with appropriate providers."""
        providers = []
        
        # Use custom providers if specified
        if config.providers:
            providers = config.providers.copy()
        else:
            # Auto-detect providers based on configuration
            # Priority order: DirectML -> CUDA -> CPU
            
            gpu_provider_added = False
            
            # Check for DirectML provider first (Windows GPU acceleration)
            if config.use_directml:
                if 'DmlExecutionProvider' in self._available_providers:
                    providers.append(('DmlExecutionProvider', {"device_id": 0}))
                    print(f"Using DirectML GPU acceleration for {config.model_path} with device_id: 0")
                    gpu_provider_added = True
                elif 'DirectMLExecutionProvider' in self._available_providers:
                    # Some versions use this provider name
                    providers.append(('DirectMLExecutionProvider', {"device_id": 0}))
                    print(f"Using DirectML GPU acceleration for {config.model_path} with device_id: 0")
                    gpu_provider_added = True
                else:
                    print(f"DirectML requested but not available. Available providers: {self._available_providers}")
            
            # Add CUDA provider if requested and available (Linux/CUDA systems)
            if config.use_cuda and 'CUDAExecutionProvider' in self._available_providers and not gpu_provider_added:
                providers.append(('CUDAExecutionProvider', {"device_id": 0}))
                print(f"Using CUDA GPU acceleration for {config.model_path} with device_id: 0")
                gpu_provider_added = True
            
            # Always add CPU provider as fallback
            providers.append('CPUExecutionProvider')
            
            if not gpu_provider_added:
                print(f"No GPU acceleration available for {config.model_path}, using CPU only")
        
        # Create session options
        session_options = config.session_options or ort.SessionOptions()
        if not config.session_options:
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        return ort.InferenceSession(
            config.model_path,
            providers=providers,
            sess_options=session_options
        )
    
    def get_session(self, session_id: str) -> ort.InferenceSession:
        """
        Get an existing ONNX session.
        
        Args:
            session_id: Session ID (model path)
            
        Returns:
            ONNX InferenceSession
            
        Raises:
            KeyError: If session doesn't exist
        """
        if session_id not in self._sessions:
            raise KeyError(f"No session found for {session_id}")
        
        return self._sessions[session_id]
    
    def get_session_metadata(self, session_id: str) -> Dict[str, Any]:
        """
        Get metadata for a session.
        
        Args:
            session_id: Session ID (model path)
            
        Returns:
            Dictionary with session metadata
            
        Raises:
            KeyError: If session doesn't exist
        """
        if session_id not in self._session_metadata:
            raise KeyError(f"No metadata found for session {session_id}")
        
        return self._session_metadata[session_id].copy()
    
    def run_inference(self, session_id: str, input_data: Dict[str, Any]) -> List[Any]:
        """
        Run inference on a specific session with DirectML fallback capability.
        
        Args:
            session_id: Session ID (model path)
            input_data: Input data dictionary {input_name: input_array}
            
        Returns:
            List of output arrays
            
        Raises:
            RuntimeError: If inference fails even after CPU fallback
        """
        if session_id not in self._sessions:
            raise KeyError(f"No session found for {session_id}")
        
        session = self._sessions[session_id]
        session_lock = self._session_locks[session_id]
        metadata = self._session_metadata[session_id]
        
        # Try inference with DirectML fallback
        for attempt in range(2):  # 0: original, 1: CPU fallback
            try:
                with session_lock:
                    outputs = session.run(metadata['output_names'], input_data)
                return outputs
                
            except Exception as e:
                error_str = str(e).lower()
                
                # Check if this is a DirectML-specific error
                is_directml_error = any(keyword in error_str for keyword in [
                    'dml', 'directml', 'dmlfusednode', 'direct3d', 'directx',
                    'd3d12', 'gpu device', 'dxgi', '80004005', 'dmlexecutionprovider'
                ])
                
                if is_directml_error and attempt == 0 and not self._directml_failed[session_id]:
                    print(f"DirectML error detected for {session_id}: {e}")
                    self._fallback_to_cpu(session_id)
                    continue  # Retry with CPU
                else:
                    # Either not a DirectML error, already on CPU, or final attempt
                    provider_info = "CPU" if self._directml_failed[session_id] else "DirectML/GPU"
                    raise RuntimeError(f"ONNX inference failed for {session_id} on {provider_info}. Original error: {str(e)}") from e
        
        # Should never reach here
        raise RuntimeError(f"Unexpected error in ONNX inference retry logic for {session_id}")
    
    def _fallback_to_cpu(self, session_id: str):
        """
        Recreate a session with CPU-only providers when DirectML fails.
        
        Args:
            session_id: Session ID (model path) to fallback
        """
        if self._directml_failed[session_id]:
            return  # Already using CPU
        
        print(f"Falling back to CPU execution for {session_id}...")
        self._directml_failed[session_id] = True
        
        with self._manager_lock:
            try:
                # Create new session with CPU-only provider
                session_options = ort.SessionOptions()
                session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                
                new_session = ort.InferenceSession(
                    session_id,  # session_id is the model path
                    providers=['CPUExecutionProvider'],
                    sess_options=session_options
                )
                
                # Replace the old session
                old_session = self._sessions[session_id]
                self._sessions[session_id] = new_session
                
                # Update metadata
                self._session_metadata[session_id].update({
                    'providers': new_session.get_providers()
                })
                
                print(f"CPU fallback successful for {session_id}")
                
                # Clean up old session if possible
                try:
                    del old_session
                except:
                    pass
                    
            except Exception as e:
                raise RuntimeError(f"Failed to fallback to CPU execution for {session_id}: {e}")
    
    def remove_session(self, session_id: str):
        """
        Remove a session and clean up resources.
        
        Args:
            session_id: Session ID (model path) to remove
        """
        with self._manager_lock:
            if session_id in self._sessions:
                try:
                    del self._sessions[session_id]
                except:
                    pass
                
            self._session_metadata.pop(session_id, None)
            self._session_locks.pop(session_id, None)
            self._directml_failed.pop(session_id, None)
            
            print(f"Removed session for {session_id}")
    
    def clear_all_sessions(self):
        """Remove all sessions and clean up resources."""
        with self._manager_lock:
            session_ids = list(self._sessions.keys())
            for session_id in session_ids:
                self.remove_session(session_id)
            
            print("Cleared all ONNX sessions")
    
    def get_session_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all active sessions.
        
        Returns:
            Dictionary with session information
        """
        info = {}
        for session_id in self._sessions:
            info[session_id] = {
                'metadata': self._session_metadata.get(session_id, {}),
                'directml_failed': self._directml_failed.get(session_id, False),
                'providers': self._sessions[session_id].get_providers()
            }
        return info
    
    def __del__(self):
        """Cleanup when the session manager is destroyed."""
        try:
            self.clear_all_sessions()
        except:
            pass


# Global session manager instance
_session_manager: Optional[ONNXSessionManager] = None
_manager_lock = threading.Lock()


def get_session_manager() -> ONNXSessionManager:
    """
    Get the global ONNX session manager instance (singleton pattern).
    
    Returns:
        ONNXSessionManager instance
    """
    global _session_manager
    
    if _session_manager is None:
        with _manager_lock:
            if _session_manager is None:
                _session_manager = ONNXSessionManager()
    
    return _session_manager


def cleanup_session_manager():
    """Clean up the global session manager."""
    global _session_manager
    
    if _session_manager is not None:
        with _manager_lock:
            if _session_manager is not None:
                _session_manager.clear_all_sessions()
                _session_manager = None