import os
import cv2
import numpy as np

def _find_haar_dir():
    env = os.environ.get("OPENCV_HAAR_DIR")
    if env and os.path.isfile(os.path.join(env, "haarcascade_frontalface_default.xml")):
        return env
    try:
        haar = getattr(cv2, "data", None)
        d = getattr(haar, "haarcascades", None) if haar else None
        if d and os.path.isfile(os.path.join(d, "haarcascade_frontalface_default.xml")):
            return d
    except Exception:
        pass
    for d in ("/usr/share/opencv4/haarcascades", "/usr/share/opencv/haarcascades"):
        if os.path.isfile(os.path.join(d, "haarcascade_frontalface_default.xml")):
            return d
    raise RuntimeError("OpenCV Haar cascades not found. Install 'opencv-data' or set OPENCV_HAAR_DIR.")

_CASCADE_PATH = os.path.join(_find_haar_dir(), "haarcascade_frontalface_default.xml")
_CASCADE = cv2.CascadeClassifier(_CASCADE_PATH)

def detect_faces(img):
    """
    Returns a numpy array of shape (N, 4) with rows [x, y, w, h].
    No drawing is done here.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = _CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Normalize to (N, 4)
    faces = np.array(faces)
    if faces.size == 0:
        return faces.reshape(0, 4)
    if faces.ndim == 3 and faces.shape[1] == 1 and faces.shape[2] >= 4:
        faces = faces[:, 0, :4]
    elif faces.ndim == 2 and faces.shape[1] >= 4:
        faces = faces[:, :4]
    else:
        faces = faces.reshape(-1, 4)
    return faces.astype(int, copy=False)