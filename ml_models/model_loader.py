# -*- coding: utf-8 -*-
# Load ML models once at application startup.
import os
import sys
import threading
import traceback
from typing import Dict, List, Optional, Tuple

from config import Config

_text_detector = None
_image_detector = None
_video_detector = None
_models_load_thread: Optional[threading.Thread] = None

# Paths actually used after init_models() (for /_debug/models and logs).
_last_loaded_text_path: Optional[str] = None
_last_loaded_tfidf_path: Optional[str] = None
_last_loaded_image_path: Optional[str] = None

_VECTORIZER_FILENAMES = frozenset(
    {
        "tfidf_vectorizer.pkl",
        "tfidf.pkl",
        "vectorizer.pkl",
    }
)
_TEXT_PREFERENCE = (
    "text_model.pkl",
    "news_classifier.pkl",
    "classifier.pkl",
    "model.pkl",
)
_TFIDF_PREFERENCE = (
    "tfidf_vectorizer.pkl",
    "tfidf.pkl",
    "vectorizer.pkl",
)


def get_loaded_model_paths() -> Dict[str, Optional[str]]:
    return {
        "text_model": _last_loaded_text_path,
        "tfidf": _last_loaded_tfidf_path,
        "image": _last_loaded_image_path,
    }


def _pick_real_file(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    if os.path.isfile(path) and not _looks_like_git_lfs_pointer(path):
        return path
    return None


def _resolve_local_text_paths() -> Tuple[Optional[str], Optional[str]]:
    """Resolve classifier + vectorizer under saved_models (and env overrides via Config)."""
    saved = os.path.join(Config.BASE_DIR, "ml_models", "saved_models")
    txt = _pick_real_file(Config.TEXT_MODEL_PATH)
    vec = _pick_real_file(Config.TFIDF_PATH)
    if txt and vec:
        return txt, vec

    def scan_preferred(names: tuple, current: Optional[str]) -> Optional[str]:
        if current:
            return current
        if not os.path.isdir(saved):
            return None
        for name in names:
            p = os.path.join(saved, name)
            hit = _pick_real_file(p)
            if hit:
                return hit
        return None

    txt = scan_preferred(_TEXT_PREFERENCE, txt)
    vec = scan_preferred(_TFIDF_PREFERENCE, vec)

    if not txt and os.path.isdir(saved):
        try:
            names = sorted(f for f in os.listdir(saved) if f.lower().endswith(".pkl"))
        except OSError:
            names = []
        others = [f for f in names if f.lower() not in {x.lower() for x in _VECTORIZER_FILENAMES}]
        if len(others) == 1:
            txt = _pick_real_file(os.path.join(saved, others[0]))

    if txt and not vec:
        vec = scan_preferred(_TFIDF_PREFERENCE, None)
    if vec and not txt:
        txt = scan_preferred(_TEXT_PREFERENCE, None)

    return txt, vec


def _unique_paths(*paths: str) -> List[str]:
    seen = set()
    out: List[str] = []
    for p in paths:
        if not p or p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def _image_tflite_candidates() -> List[str]:
    base = Config.BASE_DIR
    return _unique_paths(
        Config.IMAGE_MODEL_TFLITE_PATH,
        os.path.join(base, "ml_models", "image_model.tflite"),
        os.path.join(base, "ml", "image_model.tflite"),
    )


def _image_h5_candidates() -> List[str]:
    base = Config.BASE_DIR
    return _unique_paths(
        Config.IMAGE_MODEL_PATH,
        os.path.join(base, "ml_models", "image_model.h5"),
        os.path.join(base, "ml", "image_model.h5"),
    )


def _first_real_model_file(paths: List[str]) -> Optional[str]:
    for p in paths:
        if os.path.isfile(p) and not _looks_like_git_lfs_pointer(p):
            return p
    return None


def _resolve_image_model_path() -> Optional[str]:
    # Default: Keras .h5 first (local train_image_model.py output). Use TRUELENS_IMAGE_PREFER_TFLITE=1
    # to prefer TFLite (e.g. production without full TensorFlow). TRUELENS_IMAGE_ONLY_TFLITE blocks .h5.
    override = (os.environ.get("TRUELENS_IMAGE_MODEL_PATH") or "").strip()
    if override:
        if os.path.isfile(override) and not _looks_like_git_lfs_pointer(override):
            return override
    tflite_only = os.environ.get("TRUELENS_IMAGE_ONLY_TFLITE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if tflite_only:
        return _first_real_model_file(_image_tflite_candidates())

    prefer_tflite = os.environ.get("TRUELENS_IMAGE_PREFER_TFLITE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    prefer_h5 = os.environ.get("TRUELENS_IMAGE_PREFER_H5", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if prefer_tflite and not prefer_h5:
        p = _first_real_model_file(_image_tflite_candidates())
        if p is not None:
            return p
        return _first_real_model_file(_image_h5_candidates())

    p = _first_real_model_file(_image_h5_candidates())
    if p is not None:
        return p
    return _first_real_model_file(_image_tflite_candidates())


def _looks_like_git_lfs_pointer(path: str) -> bool:
    try:
        if os.path.getsize(path) > 512:
            return False
        with open(path, "rb") as f:
            return f.read(40).startswith(b"version https://git-lfs.github.com/spec/v1")
    except OSError:
        return False


def looks_like_git_lfs_pointer(path: str) -> bool:
    return _looks_like_git_lfs_pointer(path)


def resolve_image_model_path() -> Optional[str]:
    return _resolve_image_model_path()


def init_models() -> None:
    # Load text / image / video detectors when model files exist.
    global _text_detector, _image_detector, _video_detector
    global _last_loaded_text_path, _last_loaded_tfidf_path, _last_loaded_image_path
    from ml_models.text_detector import TextDetector
    from ml_models.video_detector import VideoDetector

    _last_loaded_text_path = None
    _last_loaded_tfidf_path = None
    _last_loaded_image_path = None

    _text_detector = None
    text_p, vec_p = _resolve_local_text_paths()
    if text_p and vec_p:
        if _looks_like_git_lfs_pointer(text_p) or _looks_like_git_lfs_pointer(vec_p):
            print(
                "TrueLens: text model or tfidf file is a Git LFS pointer (not real weights on disk). "
                f"classifier={text_p!r} vectorizer={vec_p!r}",
                file=sys.stderr,
            )
        else:
            try:
                _text_detector = TextDetector(text_p, vec_p)
                _last_loaded_text_path = text_p
                _last_loaded_tfidf_path = vec_p
                print(
                    f"TrueLens: loaded text models - classifier={text_p!r}, vectorizer={vec_p!r}",
                    file=sys.stderr,
                )
            except Exception:
                print("TrueLens: failed to load text models.", file=sys.stderr)
                traceback.print_exc()
                _text_detector = None
    else:
        print(
            "TrueLens: no local text models found. Expected .pkl files under "
            f"{os.path.join(Config.BASE_DIR, 'ml_models', 'saved_models')!r} "
            "(e.g. text_model.pkl + tfidf_vectorizer.pkl), or set TRUELENS_TEXT_MODEL_PATH / TRUELENS_TFIDF_PATH.",
            file=sys.stderr,
        )

    _image_detector = None
    img_path = _resolve_image_model_path()
    if img_path is None:
        if os.path.isfile(Config.IMAGE_MODEL_PATH) and _looks_like_git_lfs_pointer(
            Config.IMAGE_MODEL_PATH
        ):
            print(
                "TrueLens: image_model.h5 is a Git LFS pointer (large file not on disk). "
                "On Render, run bash scripts/render_git_lfs_pull.sh in build, or set "
                "PRETRAINED_MODELS_BASE_URL / IMAGE_MODEL_DOWNLOAD_URL, or add image_model.tflite.",
                file=sys.stderr,
            )
        elif os.path.isfile(Config.IMAGE_MODEL_TFLITE_PATH) and _looks_like_git_lfs_pointer(
            Config.IMAGE_MODEL_TFLITE_PATH
        ):
            print(
                "TrueLens: image_model.tflite is a Git LFS pointer; run git lfs pull in build.",
                file=sys.stderr,
            )
    else:
        try:
            from ml_models.image_detector import ImageDetector

            _image_detector = ImageDetector(img_path)
            if _image_detector.is_loaded():
                _last_loaded_image_path = img_path
                print(f"TrueLens: loaded image model from {img_path!r}", file=sys.stderr)
        except Exception:
            print("TrueLens: failed to load image model.", file=sys.stderr)
            traceback.print_exc()
            _image_detector = None

    _video_detector = None
    if _image_detector is not None and _image_detector.is_loaded():
        _video_detector = VideoDetector(_image_detector)


def start_models_loading_thread() -> None:
    global _models_load_thread
    if _models_load_thread is not None and _models_load_thread.is_alive():
        return

    def _run() -> None:
        try:
            init_models()
        except Exception:
            print("TrueLens: init_models() crashed.", file=sys.stderr)
            traceback.print_exc()

    _models_load_thread = threading.Thread(
        target=_run, daemon=True, name="truelens-ml-loader"
    )
    _models_load_thread.start()


def wait_for_models_init(timeout: Optional[float] = None) -> None:
    # Join background loader when TRUELENS_DEFER_MODEL_LOAD=1; otherwise init_models() already ran in create_app.
    global _models_load_thread
    if timeout is None:
        raw = (os.environ.get("TRUELENS_MODEL_LOAD_TIMEOUT_SEC") or "").strip()
        try:
            timeout = float(raw) if raw else 600.0
        except ValueError:
            timeout = 600.0
    t = _models_load_thread
    if t is not None and t.is_alive():
        t.join(timeout=timeout)


def get_text_detector():
    return _text_detector


def get_image_detector():
    return _image_detector


def get_video_detector():
    return _video_detector
