import os
import sys
import json
import argparse
import base64

# Ensure parent directory is in path
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
from deepfake_detector.server import run_server


def main():
    parser = argparse.ArgumentParser(description="DeepTrace AI — Advanced Deepfake & Image Authenticity Detector")
    parser.add_argument('--image', '-i', type=str, help="Path to image file to analyze")
    parser.add_argument('--save-maps', '-s', type=str, default=None, help="Directory to save visual diagnostic maps")
    parser.add_argument('--json', '-j', action='store_true', help="Output raw JSON results")
    parser.add_argument('--serve', action='store_true', help="Start the interactive Web UI server")
    parser.add_argument('--port', type=int, default=7860, help="Web server port (default: 7860)")

    parser.add_argument('--llm-key', type=str, default=None, help="API key for optional multimodal LLM Vision verification")
    parser.add_argument('--llm-provider', type=str, default='gemini', choices=['gemini', 'openai', 'nvidia'], help="LLM Provider: 'gemini', 'openai', or 'nvidia' (default: gemini)")
    parser.add_argument('--llm-model', type=str, default=None, help="LLM Vision model (e.g., gemini-2.0-flash, gpt-4o-mini, or nvidia/nemotron-3.5-super-vl-120b-a12b)")
    parser.add_argument('--sightengine-user', type=str, default=None, help="Sightengine API User (optional, overrides SIGHTENGINE_API_USER)")
    parser.add_argument('--sightengine-secret', type=str, default=None, help="Sightengine API Secret (optional, overrides SIGHTENGINE_API_SECRET)")

    args = parser.parse_args()

    if args.serve:
        run_server(args.port)
        return

    if not args.image:
        parser.print_help()
        print("\nTip: Run with --serve to start the interactive web app: python cli.py --serve")
        return

    image_path = os.path.abspath(args.image)
    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        sys.exit(1)

    print(f"\n[*] Scanning: {image_path}")
    detector = DeepfakeDetector()
    results = detector.analyze_image(
        image_path=image_path,
        llm_api_key=args.llm_key,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        sightengine_user=args.sightengine_user,
        sightengine_secret=args.sightengine_secret
    )

    if args.json:
        # Don't clutter terminal with base64 maps
        clean_res = dict(results)
        clean_res['visual_maps'] = {k: "base64_data_omitted" for k in results['visual_maps']}
        print(json.dumps(clean_res, indent=2))
        return

    # Formatted terminal report
    summary = results['summary']
    verdict = summary['verdict']
    auth_score = summary['authenticity_score']
    manip_prob = summary['manipulation_probability']

    print("=" * 60)
    print(" DEEPTRACE AI FORENSIC VERDICT REPORT")
    print("=" * 60)
    print(f" Verdict:                {verdict}")
    print(f" Authenticity Score:     {auth_score}%")
    print(f" Manipulation Risk:      {manip_prob}%")
    print(f" Confidence:             {summary['confidence']}")
    print(f" Faces Detected:         {summary['face_count']}")
    print(f" Resolution:             {summary['resolution']}")
    print("-" * 60)
    print(" FORENSIC METRICS:")
    m = results['metrics']
    print(f"  • ELA Discrepancy:     {m['ela']['discrepancy_ratio']} [{m['ela']['risk_level']} Risk]")
    print(f"  • Sensor Noise Ratio:  {m['noise']['noise_variance_ratio']} [{m['noise']['risk_level']} Risk]")
    if m['noise'].get('is_bokeh'):
        print(f"    (Optical Bokeh / Depth-of-Field Verified: Subject Sharp, Background Soft)")
    print(f"  • Spectral FFT Grid:   {m['frequency']['spectral_grid_anomaly']} [{m['frequency']['risk_level']} Risk]")
    print(f"  • Seam Boundary Step:  {m['boundary']['seam_gradient_step']} [{m['boundary']['risk_level']} Risk]")
    if 'dct' in m:
        print(f"  • JPEG DCT 8x8 Grid:   {m['dct']['grid_discrepancy']} [{m['dct']['risk_level']} Risk]")
    
    # Sightengine results if present
    se = results.get('sightengine', {})
    if se and se.get('status') == 'success':
        print("-" * 60)
        print(" SIGHTENGINE AI & DEEPFAKE DETECTION:")
        print(f"  • AI Generated Score:  {se.get('ai_generated_score', 0.0)*100:.1f}%")
        print(f"  • Face Deepfake Score: {se.get('deepfake_score', 0.0)*100:.1f}%")
        print(f"  • Combined AI Score:   {se.get('combined_score', 0.0)*100:.1f}%")

    # Smaug fallback results if present
    smaug = results.get('smaug', {})
    if smaug and smaug.get('status') == 'success':
        print("-" * 60)
        print(" SMAUG LOCAL AI DETECTOR (FALLBACK):")
        print(f"  • AI Probability:      {smaug.get('score', 0.0)*100:.1f}%")

    # LLM Audit results if present
    llm = results.get('llm_audit', {})
    if llm and llm.get('status') == 'success':
        print("-" * 60)
        print(f" AI MULTIMODAL VISION AUDIT ({llm.get('provider')} / {llm.get('model')}):")
        print(f"  • LLM Verdict:         {llm.get('verdict')}")
        print(f"  • LLM Authenticity:    {llm.get('authenticity_score')}%")
        if llm.get('mobile_processing_notes'):
            print(f"  • Mobile Processing:   {llm.get('mobile_processing_notes')}")

    print("-" * 60)
    print(" KEY EVIDENCE & FINDINGS:")
    for ind in results['indicators']:
        print(f"  [+] {ind}")
    print("=" * 60)


    # Save visual maps if requested
    if args.save_maps:
        os.makedirs(args.save_maps, exist_ok=True)
        for map_name, b64_str in results['visual_maps'].items():
            if b64_str.startswith('data:image'):
                header, encoded = b64_str.split(',', 1)
                img_data = base64.b64decode(encoded)
                out_path = os.path.join(args.save_maps, f"{map_name}.jpg")
                with open(out_path, 'wb') as f:
                    f.write(img_data)
                print(f"[Saved Map] {out_path}")


if __name__ == '__main__':
    main()
