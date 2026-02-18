"""
Face Preprocessing Module

Improves MediaPipe Face Mesh detection by:
1. Using OpenCV Haar Cascade for initial face detection
2. Cropping and upscaling the face region
3. Running MediaPipe on the preprocessed face

This helps with videos where the face is small or far from the camera.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class FacePreprocessor:
    """
    Two-stage face detection pipeline:
    1. Haar Cascade for robust initial detection
    2. Crop + Upscale for MediaPipe compatibility
    """
    
    def __init__(self, target_face_size: int = 256, upscale_factor: float = 1.5):
        """
        Initialize the preprocessor.
        
        Args:
            target_face_size: Target size for the cropped face region (pixels)
            upscale_factor: Factor to expand the bounding box for context
        """
        self.target_face_size = target_face_size
        self.upscale_factor = upscale_factor
        
        # Load Haar Cascade for frontal face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            logger.error("Failed to load Haar Cascade classifier")
            raise RuntimeError("Haar Cascade not found")
        
        # Cache for last known face position (for tracking stability)
        self._last_face_bbox = None
        self._frames_since_detection = 0
        self._max_tracking_frames = 10  # Max frames to use cached position
    
    def detect_face_haar(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect face using Haar Cascade.
        
        Args:
            frame: BGR image (OpenCV format)
            
        Returns:
            Tuple (x, y, w, h) of the largest face, or None if no face found
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with multiple scale factors for robustness
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(faces) == 0:
            # Try more aggressive detection
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(20, 20)
            )
        
        if len(faces) == 0:
            return None
        
        # Return the largest face
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        return tuple(largest_face)
    
    def expand_bbox(self, bbox: Tuple[int, int, int, int], 
                    frame_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """
        Expand bounding box to include more context (forehead, neck).
        
        Args:
            bbox: (x, y, w, h) of face
            frame_shape: (height, width) of frame
            
        Returns:
            Expanded (x, y, w, h), clipped to frame bounds
        """
        x, y, w, h = bbox
        frame_h, frame_w = frame_shape[:2]
        
        # Expand by upscale factor
        new_w = int(w * self.upscale_factor)
        new_h = int(h * self.upscale_factor * 1.2)  # Extra height for forehead
        
        # Center the expansion
        new_x = x - (new_w - w) // 2
        new_y = y - (new_h - h) // 3  # Shift up slightly for forehead
        
        # Clip to frame bounds
        new_x = max(0, new_x)
        new_y = max(0, new_y)
        new_w = min(new_w, frame_w - new_x)
        new_h = min(new_h, frame_h - new_y)
        
        return (new_x, new_y, new_w, new_h)
    
    def crop_and_upscale(self, frame: np.ndarray, 
                         bbox: Tuple[int, int, int, int]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Crop face region and upscale to target size.
        
        Args:
            frame: Full frame (BGR)
            bbox: (x, y, w, h) of face region
            
        Returns:
            Tuple of (cropped_frame, transform_info)
            transform_info contains data needed to map coordinates back
        """
        x, y, w, h = bbox
        
        # Crop
        cropped = frame[y:y+h, x:x+w]
        
        # Calculate scale to reach target size
        scale = self.target_face_size / max(w, h)
        
        if scale > 1.0:
            # Upscale using INTER_CUBIC for quality
            new_w = int(w * scale)
            new_h = int(h * scale)
            upscaled = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        else:
            # Keep original size if already large enough
            upscaled = cropped
            scale = 1.0
        
        transform_info = {
            'offset_x': x,
            'offset_y': y,
            'scale': scale,
            'original_w': w,
            'original_h': h
        }
        
        return upscaled, transform_info
    
    def preprocess_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """
        Main preprocessing function.
        
        Args:
            frame: Full frame (BGR)
            
        Returns:
            Tuple of (processed_frame, transform_info)
            If no face is detected, returns (original_frame, None)
        """
        # Try Haar detection
        face_bbox = self.detect_face_haar(frame)
        
        if face_bbox is not None:
            # Found a face, update cache
            self._last_face_bbox = face_bbox
            self._frames_since_detection = 0
        elif self._last_face_bbox is not None and self._frames_since_detection < self._max_tracking_frames:
            # Use cached position for temporal stability
            face_bbox = self._last_face_bbox
            self._frames_since_detection += 1
        else:
            # No face found and cache expired
            self._last_face_bbox = None
            return frame, None
        
        # Expand bounding box for context
        expanded_bbox = self.expand_bbox(face_bbox, frame.shape)
        
        # Crop and upscale
        processed_frame, transform_info = self.crop_and_upscale(frame, expanded_bbox)
        
        return processed_frame, transform_info
    
    def transform_landmarks_back(self, landmarks, transform_info: Dict[str, Any]) -> None:
        """
        Transform MediaPipe landmarks from cropped coordinates back to original frame.
        
        Args:
            landmarks: MediaPipe landmark object (modified in place)
            transform_info: Transform info from preprocess_frame
            
        Note: MediaPipe landmarks are in normalized [0, 1] coordinates.
              We need to convert: cropped_normalized -> cropped_pixel -> original_pixel -> original_normalized
        """
        if transform_info is None or landmarks is None:
            return
        
        offset_x = transform_info['offset_x']
        offset_y = transform_info['offset_y']
        scale = transform_info['scale']
        original_w = transform_info['original_w']
        original_h = transform_info['original_h']
        
        # Get scaled dimensions
        scaled_w = int(original_w * scale)
        scaled_h = int(original_h * scale)
        
        # Transform each landmark
        for lm in landmarks.landmark:
            # Normalized to pixel in cropped frame
            px = lm.x * scaled_w
            py = lm.y * scaled_h
            
            # Pixel in cropped back to pixel in original
            orig_px = px / scale + offset_x
            orig_py = py / scale + offset_y
            
            # Normalize to original frame (assuming we know frame size)
            # Note: This is approximate since we don't have the original frame size here
            # The caller should handle proper normalization if needed
            lm.x = orig_px
            lm.y = orig_py
            # Z and other attributes remain unchanged (relative to face size)
