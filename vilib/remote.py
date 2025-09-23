import os
import requests

# Default API endpoint for all cars
_DEFAULT_API = "http://10.42.0.1:5001"

# Allow override via environment variable if advanced users need it
API = os.getenv("VIDEO_API", _DEFAULT_API)

def _post(cmd, **kw):
    try:
        r = requests.post(f"{API}/vision", json={"cmd": cmd, **kw}, timeout=2.0)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to reach video API at {API}: {e}")

def set_color(color):
    return _post("set_color", color=color)

def color_on(color=None):
    if color:
        set_color(color)
    return _post("color_on")

def color_off():
    return _post("color_off")

def face_on():
    return _post("face_on")

def face_off():
    return _post("face_off")