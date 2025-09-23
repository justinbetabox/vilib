import numpy as np

# HSV color ranges (0-180 for H, 0-255 for S/V) — tweak as you like.
COLOR_RANGES = {
    "red":      [ (np.array([0,   80,  40]), np.array([10,  255, 255])),
                  (np.array([170, 80,  40]), np.array([180, 255, 255])) ],
    "orange":   [ (np.array([12,  80,  80]), np.array([22,  255, 255])) ],
    "yellow":   [ (np.array([22,  80, 120]), np.array([35,  255, 255])) ],
    "green":    [ (np.array([40, 120,  80]), np.array([85,  255, 255])) ],
    "blue":     [ (np.array([90, 120,  80]), np.array([120, 255, 255])) ],
    "purple":   [ (np.array([125, 30,  60]), np.array([155, 255, 255])) ],
    "magenta":  [ (np.array([160, 30,  60]), np.array([179, 255, 255])) ],
}

AVAILABLE_COLORS = list(COLOR_RANGES.keys())