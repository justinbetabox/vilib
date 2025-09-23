#!/usr/bin/env python3
from __future__ import annotations
from typing import Optional, Dict, Any
import cv2
import numpy as np

# You already have these modules in your package:
# - colors.py defines AVAILABLE_COLORS and COLOR_RANGES (HSV ranges)
# - face.py exposes detect_faces_bgr(img_bgr) -> np.ndarray of [x,y,w,h]
from .colors import AVAILABLE_COLORS, COLOR_RANGES
from .face import detect_faces


class Vilib:
    """
    Processing-only helpers.
    """

    # ---- toggles ----
    _color_enabled: bool = False
    _color_name: Optional[str] = None
    _face_enabled: bool = False

    # ---- perf knobs (safe defaults) ----
    _COLOR_DOWNSCALE: float = 0.25   # detect at 1/4 size
    _FACE_DOWNSCALE: float  = 0.50   # detect at 1/2 size
    _COLOR_MIN_BOX: int     = 8      # ignore tiny blobs

    # Useful for UI/telemetry (optional)
    detect_obj_parameter: Dict[str, Any] = {}

    # ------------- public API -------------
    @staticmethod
    def color_detect(color: str = "red") -> None:
        """Enable color detection for a given color name."""
        c = (color or "").lower()
        if c not in AVAILABLE_COLORS:
            raise ValueError(f"Unsupported color '{color}'. Options: {AVAILABLE_COLORS}")
        Vilib._color_enabled = True
        Vilib._color_name = c

    @staticmethod
    def close_color_detection() -> None:
        """Disable color detection."""
        Vilib._color_enabled = False
        Vilib._color_name = None

    @staticmethod
    def face_detect_switch(flag: bool = False) -> None:
        """Enable/disable face detection."""
        Vilib._face_enabled = bool(flag)

    @staticmethod
    def process_frame(frame: np.ndarray) -> np.ndarray:
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            return frame

        img = frame

        if Vilib._color_enabled and Vilib._color_name:
            img = Vilib._run_color(img, Vilib._color_name)

        if Vilib._face_enabled:
            img = Vilib._run_face(img)

        return img

    # ------------- internals -------------
    @staticmethod
    def _run_color(img: np.ndarray, color_name: str) -> np.ndarray:
        h, w = img.shape[:2]
        Vilib.detect_obj_parameter.update({
            "color": color_name, "color_x": None, "color_y": None,
            "color_w": 0, "color_h": 0, "color_n": 0
        })

        # downscale for speed
        ds = Vilib._COLOR_DOWNSCALE
        sw, sh = max(1, int(w * ds)), max(1, int(h * ds))
        small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_LINEAR)

        # BGR -> HSV (because ranges are in HSV)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

        # Build mask (handle red wraparound via multiple ranges)
        ranges = COLOR_RANGES[color_name]  # list[(lo, hi)]
        mask = None
        for lo, hi in ranges:
            m = cv2.inRange(hsv, lo, hi)
            mask = m if mask is None else cv2.bitwise_or(mask, m)

        # Denoise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)

        # Contours
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        Vilib.detect_obj_parameter["color_n"] = len(cnts)

        if not cnts:
            # default center
            Vilib.detect_obj_parameter.update({
                "color_x": w // 2, "color_y": h // 2, "color_w": 0, "color_h": 0
            })
            return img

        best = None
        best_area = 0
        scale = 1.0 / ds
        min_side = Vilib._COLOR_MIN_BOX

        for c in cnts:
            x, y, cw, ch = cv2.boundingRect(c)
            # up-scale to original image
            X, Y = int(x * scale), int(y * scale)
            W, H = int(cw * scale), int(ch * scale)
            if W < min_side or H < min_side:
                continue
            area = W * H
            if area > best_area:
                best_area = area
                best = (X, Y, W, H)

        if best:
            X, Y, W, H = best
            # draw in red (BGR)
            cv2.rectangle(img, (X, Y), (X + W, Y + H), (0, 0, 255), 2)
            cv2.putText(img, color_name, (X, max(0, Y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)
            Vilib.detect_obj_parameter.update({
                "color_x": int(X + W / 2),
                "color_y": int(Y + H / 2),
                "color_w": int(W),
                "color_h": int(H),
            })

        return img

    @staticmethod
    def _run_face(img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        ds = Vilib._FACE_DOWNSCALE
        sw, sh = max(1, int(w * ds)), max(1, int(h * ds))
        small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_LINEAR)

        try:
            faces_small = detect_faces(small)  # (N,4) in small coords
        except Exception:
            return img

        if faces_small is None or len(faces_small) == 0:
            return img

        scale = 1.0 / ds
        for (x, y, fw, fh) in np.asarray(faces_small).reshape(-1, 4):
            X, Y, W, H = int(x * scale), int(y * scale), int(fw * scale), int(fh * scale)
            cv2.rectangle(img, (X, Y), (X + W, Y + H), (0, 255, 0), 2)

        return img
    