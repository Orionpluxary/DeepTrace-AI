import os
import json
import base64
import urllib.request
import urllib.error
from typing import Dict, Any, Optional, Tuple


try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


class LLMVisionVerifier:
    """
    Multimodal AI Vision Verifier for deepfake and image authenticity analysis.
    Supports NVIDIA NIM (Llama 3.2 Vision), Google Gemini, and OpenAI Vision models via direct REST API.
    """

    SYSTEM_PROMPT = (
        "You are an expert digital image authenticity and optical physics analyst. "
        "Your task is to analyze photograph technical fidelity, camera optics consistency, and biological realism.\n\n"
        "Evaluation Protocol:\n"
        "1. Resolution Awareness: If the image is low resolution (< 450px, e.g. messaging crop), "
        "do NOT penalize for lacking skin pores—judge overall structural anatomy and proportions.\n"
        "2. Optical Consistency: Examine bi-ocular corneal reflections (do highlights in both pupils match?), "
        "lighting direction, shadow angles, and glasses refraction.\n"
        "3. Anatomical Realism: Check ear cartilage structure, teeth inter-dental boundaries, hair contours, and jewelry.\n"
        "4. Camera Smoothing vs Digital Compositing: Distinguish standard smartphone beauty smoothing from artificial compositing or boundary seams.\n"
        "5. Multi-Subject & Collages: If the image contains multiple faces or is a side-by-side comparison, evaluate each face individually in primary_findings.\n\n"
        "Return ONLY a valid JSON object with these exact keys:\n"
        "{\n"
        '  "verdict": "Likely Genuine / Authentic Photo" | "Suspicious / Inconclusive" | "High Probability Deepfake",\n'
        '  "verdict_code": "genuine" | "suspicious" | "deepfake",\n'
        '  "authenticity_score": <number between 0 and 100>,\n'
        '  "manipulation_probability": <number between 0 and 100>,\n'
        '  "confidence": "High" | "Medium" | "Low",\n'
        '  "is_synthetic": <true or false>,\n'
        '  "primary_findings": [<list of 3-5 concise forensic bullet points>],\n'
        '  "mobile_processing_notes": "<notes on beauty smoothing, portrait bokeh, or compression, or null>"\n'
        "}"
    )

    def __init__(self, provider: Optional[str] = None, api_key: Optional[str] = None, model: Optional[str] = None):
        self.provider = provider or os.environ.get('DEFAULT_LLM_PROVIDER') or 'gemini'
        self.api_key = api_key
        self.model = model or os.environ.get('DEFAULT_LLM_MODEL')

    def _resolve_provider_and_key(self, provider: Optional[str], api_key: Optional[str]) -> Tuple[str, Optional[str]]:
        prov = (provider or self.provider or os.environ.get('DEFAULT_LLM_PROVIDER') or 'gemini').lower()
        if 'gemini' in prov or 'google' in prov:
            prov = 'gemini'
            key = api_key or self.api_key or os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
        elif 'nvidia' in prov:
            prov = 'nvidia'
            key = api_key or self.api_key or os.environ.get('NVIDIA_API_KEY')
        elif 'openai' in prov or 'gpt' in prov:
            prov = 'openai'
            key = api_key or self.api_key or os.environ.get('OPENAI_API_KEY')
        else:
            key = api_key or self.api_key or os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY') or os.environ.get('NVIDIA_API_KEY') or os.environ.get('OPENAI_API_KEY')
            if key and key.startswith('nvapi-'):
                prov = 'nvidia'
            elif key and key.startswith('sk-'):
                prov = 'openai'
            else:
                prov = 'gemini'
        return prov, key

    def verify_image(self, image_path: str, api_key: Optional[str] = None, provider: Optional[str] = None, model: Optional[str] = None) -> Dict[str, Any]:
        """
        Run multimodal vision verification on the given image.
        Returns a dict with analysis results or error information.
        """
        prov, key = self._resolve_provider_and_key(provider, api_key)

        if not key:
            return {
                'enabled': False,
                'status': 'skipped',
                'message': f'No API key provided for {prov}. Set GEMINI_API_KEY, NVIDIA_API_KEY, or OPENAI_API_KEY.'
            }

        try:
            with open(image_path, 'rb') as f:
                image_bytes = f.read()

            # Determine mime type
            ext = os.path.splitext(image_path)[1].lower()
            mime_type = 'image/jpeg'
            if ext == '.png':
                mime_type = 'image/png'
            elif ext == '.webp':
                mime_type = 'image/webp'

            if prov == 'nvidia':
                chosen_model = model or (self.model if self.model and 'nemotron' in self.model else None) or 'nvidia/nemotron-3.5-super-vl-120b-a12b'
                return self._call_nvidia(image_bytes, mime_type, key, chosen_model)
            elif prov == 'gemini':
                chosen_model = model or (self.model if self.model and 'gemini' in self.model else None) or 'gemini-3.1-flash-lite'
                return self._call_gemini(image_bytes, mime_type, key, chosen_model)
            elif prov == 'openai':
                chosen_model = model or (self.model if self.model and 'gpt' in self.model else None) or 'gpt-4o-mini'
                return self._call_openai(image_bytes, mime_type, key, chosen_model)
            else:
                return {
                    'enabled': False,
                    'status': 'error',
                    'message': f"Unsupported LLM provider: {prov}"
                }
        except Exception as e:
            return {
                'enabled': True,
                'status': 'error',
                'message': f"AI Vision verification error: {str(e)}"
            }

    def _call_gemini(self, image_bytes: bytes, mime_type: str, api_key: str, model: str) -> Dict[str, Any]:
        """Call Google Gemini Generative Language API directly."""
        b64_data = base64.b64encode(image_bytes).decode('utf-8')
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": self.SYSTEM_PROMPT},
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": b64_data
                            }
                        }
                    ]
                }
            ],
            "generationConfig": {
                "response_mime_type": "application/json",
                "temperature": 0.1
            }
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
            method='POST'
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                res_body = response.read().decode('utf-8')
                res_json = json.loads(res_body)
                
                # Extract text
                candidates = res_json.get('candidates', [])
                if not candidates:
                    return {'enabled': True, 'status': 'error', 'message': 'Gemini returned no candidates'}
                
                parts = candidates[0].get('content', {}).get('parts', [])
                raw_text = "".join(p.get('text', '') for p in parts).strip()
                parsed = self._clean_and_parse_json(raw_text)
                parsed['enabled'] = True
                parsed['status'] = 'success'
                parsed['provider'] = 'Gemini'
                parsed['model'] = model
                return parsed

        except urllib.error.HTTPError as e:
            fallback_models = ['gemini-3.1-flash-lite', 'gemini-3.8-flash', 'gemini-3.5-flash']
            if e.code in (404, 503) and model in fallback_models:
                remaining = [m for m in fallback_models if m != model]
                if remaining:
                    return self._call_gemini(image_bytes, mime_type, api_key, remaining[0])
            # Failover to NVIDIA NIM if Gemini is experiencing high demand spike
            if os.environ.get('NVIDIA_API_KEY'):
                nv_key = os.environ.get('NVIDIA_API_KEY')
                nv_res = self._call_nvidia(image_bytes, mime_type, nv_key, 'nvidia/nemotron-3.5-super-vl-120b-a12b')
                if nv_res.get('status') == 'success':
                    nv_res['mobile_processing_notes'] = f"Gemini ({model}) temporarily busy ({e.code}); seamlessly verified via NVIDIA Nemotron 120B."
                    return nv_res
            err_text = e.read().decode('utf-8', errors='ignore')
            return {
                'enabled': True,
                'status': 'error',
                'message': f"Gemini API HTTP Error {e.code}: {err_text[:200]}"
            }

    def _call_openai(self, image_bytes: bytes, mime_type: str, api_key: str, model: str) -> Dict[str, Any]:
        """Call OpenAI Chat Completions API directly."""
        b64_data = base64.b64encode(image_bytes).decode('utf-8')
        url = "https://api.openai.com/v1/chat/completions"

        data_uri = f"data:{mime_type};base64,{b64_data}"
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": self.SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze this image for authenticity and deepfake / synthetic indicators."},
                        {"type": "image_url", "image_url": {"url": data_uri}}
                    ]
                }
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode('utf-8'),
            headers={
                'Content-Type': 'application/json',
                'Authorization': f"Bearer {api_key}"
            },
            method='POST'
        )

        try:
            with urllib.request.urlopen(req, timeout=25) as response:
                res_body = response.read().decode('utf-8')
                res_json = json.loads(res_body)
                content = res_json['choices'][0]['message']['content']
                parsed = self._clean_and_parse_json(content)
                parsed['enabled'] = True
                parsed['status'] = 'success'
                parsed['provider'] = 'OpenAI'
                parsed['model'] = model
                return parsed
        except urllib.error.HTTPError as e:
            err_text = e.read().decode('utf-8', errors='ignore')
            return {
                'enabled': True,
                'status': 'error',
                'message': f"OpenAI API HTTP Error {e.code}: {err_text[:200]}"
            }

    def _call_nvidia(self, image_bytes: bytes, mime_type: str, api_key: str, model: str) -> Dict[str, Any]:
        """Call NVIDIA NIM API directly (e.g. meta/llama-3.2-11b-vision-instruct)."""
        if api_key and not api_key.startswith('nvapi-') and len(api_key) > 20:
            api_key = f"nvapi-{api_key}"
        b64_data = base64.b64encode(image_bytes).decode('utf-8')
        url = "https://integrate.api.nvidia.com/v1/chat/completions"

        data_uri = f"data:{mime_type};base64,{b64_data}"
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{self.SYSTEM_PROMPT}\n\nAnalyze this image for authenticity and deepfake / synthetic indicators."},
                        {"type": "image_url", "image_url": {"url": data_uri}}
                    ]
                }
            ],
            "max_tokens": 2048,
            "temperature": 0.1
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode('utf-8'),
            headers={
                'Content-Type': 'application/json',
                'Authorization': f"Bearer {api_key}"
            },
            method='POST'
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                res_body = response.read().decode('utf-8')
                res_json = json.loads(res_body)
                msg = res_json['choices'][0]['message']
                content = msg.get('content') or msg.get('reasoning_content') or ''
                parsed = self._clean_and_parse_json(content)
                parsed['enabled'] = True
                parsed['status'] = 'success'
                parsed['provider'] = 'NVIDIA NIM'
                parsed['model'] = model
                return parsed
        except Exception as e:
            err_text = getattr(e, 'read', lambda: b'')()
            err_msg = err_text.decode('utf-8', errors='ignore') if err_text else str(e)
            return {
                'enabled': True,
                'status': 'error',
                'message': f"NVIDIA NIM API Error: {err_msg[:200]}"
            }

    def _clean_and_parse_json(self, raw_text: str) -> Dict[str, Any]:
        text = raw_text.strip()

        # Check for model refusal text (e.g. Meta safety trigger)
        refusal_phrases = ["not going to engage", "cannot fulfill", "unable to assist", "i cannot analyze", "against my safety"]
        if any(rp in text.lower() for rp in refusal_phrases):
            return {
                'status': 'refusal',
                'enabled': True,
                'message': 'Model safety filter refusal: ' + text[:150]
            }
        
        # 1. Try stripping markdown code fences
        cleaned = text
        if '```json' in cleaned:
            start = cleaned.find('```json') + 7
            end = cleaned.find('```', start)
            cleaned = cleaned[start:end].strip() if end != -1 else cleaned[start:].strip()
        elif '```' in cleaned:
            start = cleaned.find('```') + 3
            end = cleaned.find('```', start)
            cleaned = cleaned[start:end].strip() if end != -1 else cleaned[start:].strip()

        # 2. Try parsing cleaned text directly
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return self._normalize_result(parsed)
        except Exception:
            pass

        # 3. Try finding outermost JSON object braces { ... }
        first_brace = text.find('{')
        last_brace = text.rfind('}')
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            try:
                parsed = json.loads(text[first_brace:last_brace+1])
                if isinstance(parsed, dict):
                    return self._normalize_result(parsed)
            except Exception:
                pass

        # 4. Fallback for natural language / analytical text outputs (e.g. Llama 3.2 Vision)
        import re
        m_auth = re.search(r'authenticity[_\s]*score[^\d]*(\d+(?:\.\d+)?)', text, re.IGNORECASE)
        m_manip = re.search(r'manipulation[_\s]*probability[^\d]*(\d+(?:\.\d+)?)', text, re.IGNORECASE)
        m_synth = re.search(r'is_synthetic[^\w]*(true|false)', text, re.IGNORECASE)

        if m_auth:
            auth_score = float(m_auth.group(1))
            manip_prob = float(m_manip.group(1)) if m_manip else max(1.0, 100.0 - auth_score)
            is_synthetic = (m_synth.group(1).lower() == 'true') if m_synth else (manip_prob >= 50.0)
            verdict_code = 'deepfake' if manip_prob >= 60.0 else ('suspicious' if manip_prob >= 30.0 else 'genuine')
            verdict = 'High Probability Deepfake' if verdict_code == 'deepfake' else ('Suspicious / Inconclusive' if verdict_code == 'suspicious' else 'Likely Genuine / Authentic Photo')
        elif m_manip:
            manip_prob = float(m_manip.group(1))
            auth_score = max(1.0, 100.0 - manip_prob)
            is_synthetic = (m_synth.group(1).lower() == 'true') if m_synth else (manip_prob >= 50.0)
            verdict_code = 'deepfake' if manip_prob >= 60.0 else ('suspicious' if manip_prob >= 30.0 else 'genuine')
            verdict = 'High Probability Deepfake' if verdict_code == 'deepfake' else ('Suspicious / Inconclusive' if verdict_code == 'suspicious' else 'Likely Genuine / Authentic Photo')
        else:
            text_lower = text.lower()
            clean_search_text = re.sub(r'manipulation[_\s]*probability', '', text_lower)
            is_synthetic = any(w in clean_search_text for w in ['high probability deepfake', 'ai-generated', 'synthetic media', 'face swap', 'deepfake', 'fabricated portrait'])

            if is_synthetic:
                verdict = "Suspicious / Manipulated Content Detected"
                verdict_code = "deepfake"
                auth_score = 30.0
                manip_prob = 70.0
            else:
                verdict = "Likely Genuine / Authentic Photo"
                verdict_code = "genuine"
                auth_score = 90.0
                manip_prob = 10.0

        findings = []
        for line in text.split('\n'):
            clean_line = line.strip().lstrip('*-+•# \t')
            if len(clean_line) > 18:
                findings.append(clean_line)
                if len(findings) >= 5:
                    break

        if not findings:
            findings = [text[:250]]

        return self._normalize_result({
            'verdict': verdict,
            'verdict_code': verdict_code,
            'authenticity_score': auth_score,
            'manipulation_probability': manip_prob,
            'confidence': 'High',
            'is_synthetic': is_synthetic,
            'primary_findings': findings,
            'mobile_processing_notes': None
        })

    @staticmethod
    def _normalize_result(result: Dict[str, Any]) -> Dict[str, Any]:
        """Make contradictory model fields safe for ensemble scoring."""
        normalized = dict(result)
        try:
            auth_score = min(100.0, max(0.0, float(normalized.get('authenticity_score', 50.0))))
        except (TypeError, ValueError):
            auth_score = 50.0
        try:
            manip_prob = min(100.0, max(0.0, float(normalized.get('manipulation_probability', 100.0 - auth_score))))
        except (TypeError, ValueError):
            manip_prob = 100.0 - auth_score

        verdict_code = str(normalized.get('verdict_code', '')).lower()
        is_synthetic = bool(normalized.get('is_synthetic', False))
        if is_synthetic or verdict_code == 'deepfake':
            verdict_code = 'deepfake'
            manip_prob = max(manip_prob, 60.0)
        elif verdict_code == 'suspicious':
            manip_prob = max(manip_prob, 30.0)
        elif verdict_code == 'genuine' and manip_prob >= 30.0:
            verdict_code = 'suspicious'
        elif verdict_code not in {'genuine', 'suspicious', 'deepfake'}:
            verdict_code = 'deepfake' if manip_prob >= 60.0 else ('suspicious' if manip_prob >= 30.0 else 'genuine')

        verdict = (
            'High Probability Deepfake' if verdict_code == 'deepfake'
            else 'Suspicious / Inconclusive' if verdict_code == 'suspicious'
            else 'Likely Genuine / Authentic Photo'
        )
        normalized.update({
            'verdict': verdict,
            'verdict_code': verdict_code,
            'authenticity_score': round(100.0 - manip_prob, 1),
            'manipulation_probability': round(manip_prob, 1),
            'is_synthetic': is_synthetic or verdict_code == 'deepfake',
        })
        return normalized
