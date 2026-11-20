import os
import json
from typing import Dict, Any, Optional
import requests


class SightengineDetector:
    """
    Client for Sightengine's AI Generated Image and Deepfake Detection API.
    Endpoint: https://api.sightengine.com/1.0/check.json
    Models: 'genai,deepfake'
    """
    ENDPOINT = "https://api.sightengine.com/1.0/check.json"

    def __init__(self, api_user: Optional[str] = None, api_secret: Optional[str] = None):
        self.api_user = api_user or os.environ.get('SIGHTENGINE_API_USER')
        self.api_secret = api_secret or os.environ.get('SIGHTENGINE_API_SECRET')

    def detect(
        self,
        image_path: str,
        api_user: Optional[str] = None,
        api_secret: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send image to Sightengine for AI image (genai) and face deepfake detection.
        Returns a dictionary with parsed scores or error status.
        """
        if api_user is not None and api_user.strip() == '':
            user = None
        else:
            user = api_user or self.api_user or os.environ.get('SIGHTENGINE_API_USER')

        if api_secret is not None and api_secret.strip() == '':
            secret = None
        else:
            secret = api_secret or self.api_secret or os.environ.get('SIGHTENGINE_API_SECRET')

        if not user or not secret:
            return {
                'enabled': False,
                'status': 'skipped',
                'ai_generated_score': None,
                'deepfake_score': None,
                'combined_score': None,
                'message': 'Sightengine credentials not provided (missing SIGHTENGINE_API_USER or SIGHTENGINE_API_SECRET).'
            }

        if not os.path.exists(image_path):
            return {
                'enabled': True,
                'status': 'error',
                'ai_generated_score': None,
                'deepfake_score': None,
                'combined_score': None,
                'message': f"Image file not found: {image_path}"
            }

        params = {
            'models': 'genai,deepfake',
            'api_user': user.strip(),
            'api_secret': secret.strip()
        }

        try:
            with open(image_path, 'rb') as f:
                files = {'media': f}
                response = requests.post(
                    self.ENDPOINT,
                    files=files,
                    data=params,
                    timeout=25
                )

            if response.status_code != 200:
                err_msg = f"HTTP {response.status_code}: {response.text[:200]}"
                try:
                    res_json = response.json()
                    if 'error' in res_json:
                        err_msg = res_json['error'].get('message', err_msg)
                except Exception:
                    pass

                return {
                    'enabled': True,
                    'status': 'error',
                    'ai_generated_score': None,
                    'deepfake_score': None,
                    'combined_score': None,
                    'message': f"Sightengine API Error: {err_msg}"
                }

            data = response.json()

            if data.get('status') == 'failure':
                err_msg = data.get('error', {}).get('message', 'Sightengine returned status failure')
                return {
                    'enabled': True,
                    'status': 'error',
                    'ai_generated_score': None,
                    'deepfake_score': None,
                    'combined_score': None,
                    'message': f"Sightengine API failure: {err_msg}"
                }

            # 1. AI-Generated score from type.ai_generated
            type_info = data.get('type', {})
            ai_gen_score = float(type_info.get('ai_generated', 0.0))

            # 2. Deepfake score from faces[].attributes.deepfake
            faces = data.get('faces', [])
            deepfake_scores = []
            for face in faces:
                attrs = face.get('attributes', {})
                if 'deepfake' in attrs:
                    deepfake_scores.append(float(attrs['deepfake']))

            max_deepfake_score = max(deepfake_scores) if deepfake_scores else 0.0

            # 3. Combined Sightengine score (max of generative AI and face deepfake)
            combined_score = max(ai_gen_score, max_deepfake_score)

            return {
                'enabled': True,
                'status': 'success',
                'ai_generated_score': round(ai_gen_score, 4),
                'deepfake_score': round(max_deepfake_score, 4),
                'combined_score': round(combined_score, 4),
                'face_count': len(faces),
                'per_face_deepfake': deepfake_scores,
                'message': 'Sightengine scan successful.',
                'raw': data
            }

        except Exception as e:
            return {
                'enabled': True,
                'status': 'error',
                'ai_generated_score': None,
                'deepfake_score': None,
                'combined_score': None,
                'message': f"Sightengine request exception: {str(e)}"
            }
