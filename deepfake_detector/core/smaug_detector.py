import os
import sys
from typing import Dict, Any, Optional


class SmaugDetector:
    """
    Local fallback detector using Hugging Face's abacusai/Smaug-AI-Image-Detector.
    If unavailable or fails to load, gracefully falls back without crashing.
    """
    MODEL_ID = "abacusai/Smaug-AI-Image-Detector"
    _instance = None
    _model = None
    _init_attempted = False
    _load_error = None

    def __init__(self):
        pass

    @classmethod
    def _get_model(cls):
        if cls._model is not None:
            return cls._model

        if cls._init_attempted and cls._load_error is not None:
            return None

        cls._init_attempted = True
        try:
            from huggingface_hub import snapshot_download
            try:
                model_path = snapshot_download(cls.MODEL_ID, local_files_only=True)
            except Exception:
                model_path = snapshot_download(cls.MODEL_ID, local_files_only=False)

            if model_path not in sys.path:
                sys.path.insert(0, model_path)

            try:
                from smaug_image_detect import SmaugImageDetect
                cls._model = SmaugImageDetect.from_pretrained(model_path)
                return cls._model
            except Exception as e_custom:
                # Try pipeline or AutoModel fallback if custom class import fails
                try:
                    from transformers import pipeline
                    cls._model = pipeline("image-classification", model=model_path)
                    return cls._model
                except Exception:
                    raise e_custom

        except Exception as e:
            cls._load_error = str(e)
            return None

    def detect(self, image_path: str) -> Dict[str, Any]:
        """
        Run inference on the image using abacusai/Smaug-AI-Image-Detector.
        Returns a dict with 'status' and 'score' (0.0 - 1.0) or fallback status.
        Never raises exceptions; fails gracefully.
        """
        if not os.path.exists(image_path):
            return {
                'status': 'error',
                'model': self.MODEL_ID,
                'score': None,
                'message': f"Image file not found: {image_path}"
            }

        try:
            model = self._get_model()
            if model is None:
                err_hint = self._load_error or "Model weights not cached or could not be loaded"
                return {
                    'status': 'fallback_unavailable',
                    'model': self.MODEL_ID,
                    'score': None,
                    'message': f"Local Smaug model unavailable ({err_hint}). Cleanly falling back to forensic checks."
                }

            # Inference
            if callable(model):
                res = model(image_path)
                # If returns a float or dict
                if isinstance(res, (float, int)):
                    score = float(res)
                elif isinstance(res, list) and len(res) > 0 and isinstance(res[0], dict):
                    # Pipeline output format: [{'label': 'AI', 'score': 0.95}, ...]
                    ai_items = [item['score'] for item in res if 'ai' in item.get('label', '').lower() or 'fake' in item.get('label', '').lower()]
                    score = float(ai_items[0]) if ai_items else float(res[0].get('score', 0.5))
                elif isinstance(res, dict) and 'score' in res:
                    score = float(res['score'])
                else:
                    score = float(res)

                return {
                    'status': 'success',
                    'model': self.MODEL_ID,
                    'score': round(min(1.0, max(0.0, score)), 4),
                    'message': 'Smaug AI Image Detector inference completed successfully.'
                }

        except Exception as e:
            return {
                'status': 'fallback_unavailable',
                'model': self.MODEL_ID,
                'score': None,
                'message': f"Smaug inference error ({str(e)}). Cleanly falling back to forensic checks."
            }
