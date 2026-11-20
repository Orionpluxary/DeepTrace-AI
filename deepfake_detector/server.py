import os
import sys
import json
import tempfile
from http.server import HTTPServer, SimpleHTTPRequestHandler
from email.parser import BytesParser
from email.policy import default

# Ensure workspace root is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from dotenv import load_dotenv
    env_path = os.path.join(parent_dir, '.env')
    if os.path.exists(env_path):
        load_dotenv(dotenv_path=env_path, override=True)
    else:
        load_dotenv()
except Exception:
    pass

from deepfake_detector.core.forensics import DeepfakeDetector

WEB_DIR = os.path.join(current_dir, 'web')
detector = DeepfakeDetector()


class ForensicHTTPHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def do_GET(self):
        if self.path == '/api/health':
            se_user = os.environ.get('SIGHTENGINE_API_USER')
            se_secret = os.environ.get('SIGHTENGINE_API_SECRET')
            has_se = bool(se_user and se_secret and se_user.strip() and se_secret.strip())
            self._send_json({
                'status': 'ok',
                'engine': 'DeepTrace AI v2.4.0',
                'sightengine_active': has_se,
                'primary_detector': 'Sightengine (genai,deepfake)' if has_se else 'Smaug / Forensic Ensemble'
            })
            return
        # Default static file routing
        return super().do_GET()

    def do_POST(self):
        if self.path == '/api/analyze':
            content_type = self.headers.get('Content-Type', '')
            content_length = int(self.headers.get('Content-Length', 0))

            if content_length == 0 or 'multipart/form-data' not in content_type:
                self._send_json({'error': 'Invalid request: multipart/form-data required'}, status=400)
                return

            try:
                body = self.rfile.read(content_length)
                header_bytes = f"Content-Type: {content_type}\r\n\r\n".encode('latin-1')
                msg = BytesParser(policy=default).parsebytes(header_bytes + body)

                image_bytes = None
                filename = "upload.jpg"
                llm_key = self.headers.get('X-LLM-Key', None)
                llm_provider = self.headers.get('X-LLM-Provider', None)
                llm_model = self.headers.get('X-LLM-Model', None)
                sightengine_user = self.headers.get('X-Sightengine-User', None)
                sightengine_secret = self.headers.get('X-Sightengine-Secret', None)

                for part in msg.iter_parts():
                    cd = part.get('Content-Disposition', '')
                    name = part.get_param('name', header='content-disposition')
                    if name == 'image' or part.get_filename():
                        image_bytes = part.get_payload(decode=True)
                        if part.get_filename():
                            filename = part.get_filename()
                    elif name in ('sightengine_user', 'sightengine_api_user'):
                        val = part.get_payload(decode=True).decode('utf-8', errors='ignore').strip()
                        if val:
                            sightengine_user = val
                    elif name in ('sightengine_secret', 'sightengine_api_secret'):
                        val = part.get_payload(decode=True).decode('utf-8', errors='ignore').strip()
                        if val:
                            sightengine_secret = val
                    elif name == 'llm_key':
                        val = part.get_payload(decode=True).decode('utf-8', errors='ignore').strip()
                        if val:
                            llm_key = val
                    elif name == 'llm_provider':
                        val = part.get_payload(decode=True).decode('utf-8', errors='ignore').strip()
                        if val:
                            llm_provider = val
                    elif name == 'llm_model':
                        val = part.get_payload(decode=True).decode('utf-8', errors='ignore').strip()
                        if val:
                            llm_model = val

                if not image_bytes:
                    self._send_json({'error': 'No image file found in form data'}, status=400)
                    return

                # Save to temporary file
                ext = os.path.splitext(filename)[1] or '.jpg'
                with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as temp_file:
                    temp_path = temp_file.name
                    temp_file.write(image_bytes)

                try:
                    result = detector.analyze_image(
                        image_path=temp_path,
                        llm_api_key=llm_key,
                        llm_provider=llm_provider,
                        llm_model=llm_model,
                        sightengine_user=sightengine_user,
                        sightengine_secret=sightengine_secret
                    )
                    self._send_json(result)
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

            except Exception as e:
                self._send_json({'error': f"Forensic analysis failed: {str(e)}"}, status=500)
            return

        self._send_json({'error': 'Endpoint not found'}, status=404)

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(
            data,
            default=lambda o: o.item() if hasattr(o, 'item') else (o.tolist() if hasattr(o, 'tolist') else str(o))
        ).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)


def run_server(port: int = 7860):
    global detector
    try:
        from dotenv import load_dotenv
        env_path = os.path.join(parent_dir, '.env')
        if os.path.exists(env_path):
            load_dotenv(dotenv_path=env_path, override=True)
    except Exception:
        pass

    detector = DeepfakeDetector()
    se_user = os.environ.get('SIGHTENGINE_API_USER')
    se_secret = os.environ.get('SIGHTENGINE_API_SECRET')
    has_se = bool(se_user and se_secret and se_user.strip() and se_secret.strip())

    server_address = ('127.0.0.1', port)
    httpd = HTTPServer(server_address, ForensicHTTPHandler)
    print(f"\n=======================================================")
    print(f" DeepTrace AI — Forensic Deepfake Detection Suite")
    print(f" Web UI Running at: http://127.0.0.1:{port}")
    if has_se:
        print(f" Primary AI Detector: Sightengine (models='genai,deepfake') [ACTIVE from .env]")
    else:
        print(f" Primary AI Detector: Local Smaug & Signal Forensics [Sightengine not in .env]")
    print(f"=======================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down DeepTrace server.")
        httpd.server_close()


if __name__ == '__main__':
    port = 7860
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])
    run_server(port)
