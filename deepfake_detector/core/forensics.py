import base64
import os
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import cv2
from PIL import Image
from PIL.ExifTags import TAGS

from deepfake_detector.core.face_engine import FaceDetector
from deepfake_detector.core.llm_verifier import LLMVisionVerifier
from deepfake_detector.core.sightengine_detector import SightengineDetector
from deepfake_detector.core.smaug_detector import SmaugDetector


class DeepfakeDetector:
    def __init__(self):
        self.face_detector = FaceDetector()
        self.llm_verifier = LLMVisionVerifier()
        self.sightengine_detector = SightengineDetector()
        self.smaug_detector = SmaugDetector()

    def analyze_image(
        self,
        image_path: str,
        llm_api_key: Optional[str] = None,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        sightengine_user: Optional[str] = None,
        sightengine_secret: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive forensic analysis on an image file.
        Returns detailed report and base64-encoded visual maps.
        Combines Sightengine (genai, deepfake), Smaug fallback, physical signal processing,
        and optional multimodal LLM Vision verification.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")

        # Load image via OpenCV and PIL
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise ValueError(f"Could not decode image at {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        h, w = image_bgr.shape[:2]

        # 1. Face Detection
        faces = self.face_detector.detect_faces(image_bgr)
        primary_face = faces[0] if faces else None

        # 2. Error Level Analysis (ELA)
        ela_data = self._compute_ela(image_rgb, primary_face)

        # 3. Noise Residual (PRNU Inconsistency & Optical Depth of Field)
        noise_data = self._compute_noise_residual(gray, primary_face)

        # 4. Frequency Domain (2D FFT Spectrum)
        fft_data = self._compute_fft_spectrum(gray, primary_face)

        # 5. Face Boundary Seam Discontinuity
        boundary_data = self._compute_boundary_gradient(gray, primary_face)

        # 6. JPEG 8x8 Block Grid (DCT) Discontinuity Analysis
        dct_data = self._compute_dct_grid(gray, primary_face)

        # 7. Metadata & EXIF Analysis
        metadata_data = self._analyze_metadata(image_path, width=w, height=h)

        # 8. Sightengine AI Image & Face Deepfake Detection (Primary & Sole API)
        sightengine_result = self.sightengine_detector.detect(
            image_path=image_path,
            api_user=sightengine_user,
            api_secret=sightengine_secret
        )

        # 9. Fallback to Smaug ONLY if Sightengine credentials missing or API call failed
        smaug_result = None
        if sightengine_result.get('status') != 'success':
            smaug_result = self.smaug_detector.detect(image_path=image_path)

        # 10. Optional LLM Vision Audit ONLY as fallback when Sightengine is unavailable
        llm_result = None
        if sightengine_result.get('status') != 'success' and llm_provider and llm_provider != 'offline':
            llm_result = self.llm_verifier.verify_image(
                image_path=image_path,
                api_key=llm_api_key,
                provider=llm_provider,
                model=llm_model
            )

        # 11. Compute Unified Score with 3-tier verdict (100% Sightengine when active)
        ensemble = self._compute_ensemble_score(
            ela_data=ela_data,
            noise_data=noise_data,
            fft_data=fft_data,
            boundary_data=boundary_data,
            dct_data=dct_data,
            metadata_data=metadata_data,
            has_face=(primary_face is not None),
            sightengine_result=sightengine_result,
            smaug_result=smaug_result,
            llm_result=llm_result
        )

        # Multi-Face Per-Face Evaluation (aligned with Sightengine)
        per_face_results = []
        se_faces = sightengine_result.get('per_face_deepfake', []) if sightengine_result else []
        se_success = bool(sightengine_result and sightengine_result.get('status') == 'success')

        for idx, f in enumerate(faces):
            if se_success:
                if idx < len(se_faces):
                    df_val = se_faces[idx]
                    f_manip = round(max(sightengine_result.get('ai_generated_score', 0.0), df_val) * 100.0, 1)
                else:
                    f_manip = ensemble['manipulation_probability']
                f_auth = round(100.0 - f_manip, 1)
                f_vcode = 'deepfake' if f_manip >= 70.0 else ('genuine' if f_manip <= 30.0 else 'suspicious')
                f_verdict = 'Likely AI-Generated / Manipulated' if f_vcode == 'deepfake' else ('Likely Authentic' if f_vcode == 'genuine' else 'Inconclusive / Compressed Signal')
            elif len(faces) == 1:
                f_auth = ensemble['authenticity_score']
                f_manip = ensemble['manipulation_probability']
                f_verdict = ensemble['verdict']
                f_vcode = ensemble['verdict_code']
            else:
                f_ela = self._compute_ela(image_rgb, f)
                f_noise = self._compute_noise_residual(gray, f)
                f_fft = self._compute_fft_spectrum(gray, f)
                f_boundary = self._compute_boundary_gradient(gray, f)
                f_dct = self._compute_dct_grid(gray, f)
                f_score = self._compute_ensemble_score(
                    ela_data=f_ela,
                    noise_data=f_noise,
                    fft_data=f_fft,
                    boundary_data=f_boundary,
                    dct_data=f_dct,
                    metadata_data=metadata_data,
                    has_face=True,
                    llm_result=None
                )
                f_auth = f_score['authenticity_score']
                f_manip = f_score['manipulation_probability']
                f_verdict = f_score['verdict']
                f_vcode = f_score['verdict_code']

            per_face_results.append({
                'face_id': idx + 1,
                'bbox': f['bbox'],
                'detection_confidence': round(float(f.get('score', 1.0)), 2),
                'authenticity_score': f_auth,
                'manipulation_probability': f_manip,
                'verdict': f_verdict,
                'verdict_code': f_vcode
            })

        # 9. Generate Visual Maps with clear per-face color-coded tags
        annotated_image = self._draw_face_boxes(image_bgr.copy(), faces, per_face_results)
        maps = {
            'annotated': self._image_to_base64(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)),
            'ela_map': self._image_to_base64(ela_data['visual_map']),
            'noise_map': self._image_to_base64(noise_data['visual_map']),
            'fft_map': self._image_to_base64(fft_data['visual_map']),
            'boundary_map': self._image_to_base64(boundary_data['visual_map']),
            'dct_map': self._image_to_base64(dct_data['visual_map']),
        }

        indicators = list(ensemble['indicators'])
        if len(faces) > 1:
            indicators.insert(0, f"Multi-face composition: {len(faces)} individual faces detected and analyzed across the canvas.")

        return {
            'summary': {
                'verdict': ensemble['verdict'],
                'verdict_code': ensemble['verdict_code'],  # 'genuine', 'suspicious', 'deepfake'
                'authenticity_score': ensemble['authenticity_score'],
                'manipulation_probability': ensemble['manipulation_probability'],
                'confidence': ensemble['confidence'],
                'face_count': len(faces),
                'resolution': f"{w}x{h}",
                'per_face': per_face_results,
            },
            'indicators': indicators,
            'metrics': {
                'ela': {
                    'discrepancy_ratio': round(ela_data['discrepancy_ratio'], 3),
                    'global_energy': round(ela_data['global_mean'], 2),
                    'face_energy': round(ela_data.get('face_mean', 0.0), 2),
                    'risk_level': ela_data['risk_level'],
                },
                'noise': {
                    'noise_variance_ratio': round(noise_data['variance_ratio'], 3),
                    'face_noise_std': round(noise_data.get('face_std', 0.0), 2),
                    'bg_noise_std': round(noise_data.get('bg_std', 0.0), 2),
                    'is_bokeh': noise_data.get('is_bokeh', False),
                    'risk_level': noise_data['risk_level'],
                },
                'frequency': {
                    'spectral_grid_anomaly': round(fft_data['grid_anomaly_score'], 3),
                    'high_frequency_energy': round(fft_data['hf_ratio'], 3),
                    'risk_level': fft_data['risk_level'],
                },
                'boundary': {
                    'seam_gradient_step': round(float(boundary_data['gradient_step']), 3),
                    'risk_level': boundary_data['risk_level'],
                },
                'dct': {
                    'grid_discrepancy': round(float(dct_data['grid_discrepancy']), 3),
                    'risk_level': dct_data['risk_level'],
                },
                'metadata': metadata_data,
            },
            'sightengine': sightengine_result,
            'smaug': smaug_result,
            'llm_audit': llm_result,
            'visual_maps': maps
        }

    def _compute_ela(self, image_rgb: np.ndarray, primary_face: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Error Level Analysis: re-compress image to JPEG 90 and calculate pixel difference.
        Properly calibrated with absolute difference gating to prevent division-by-zero
        and false positive ratios on low-energy images.
        """
        # Re-encode to JPEG quality 90
        success, encoded = cv2.imencode('.jpg', cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not success:
            return {'discrepancy_ratio': 0.0, 'global_mean': 0.0, 'face_mean': 0.0, 'bg_mean': 0.0, 'risk_level': 'Low', 'visual_map': image_rgb}

        recompressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        recompressed_rgb = cv2.cvtColor(recompressed, cv2.COLOR_BGR2RGB)

        # Difference
        diff = cv2.absdiff(image_rgb, recompressed_rgb).astype(np.float32)
        diff_gray = cv2.cvtColor(diff.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        global_mean = float(np.mean(diff_gray))
        global_std = float(np.std(diff_gray))

        discrepancy_ratio = 0.0
        face_mean = 0.0
        bg_mean = global_mean
        abs_diff = 0.0

        if primary_face:
            x1, y1, x2, y2 = primary_face['bbox']
            h, w = diff_gray.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            face_region = diff_gray[y1:y2, x1:x2]
            if face_region.size > 0:
                face_mean = float(np.mean(face_region))
                
                # Mask out face to get background
                mask = np.ones_like(diff_gray, dtype=bool)
                mask[y1:y2, x1:x2] = False
                bg_region = diff_gray[mask]
                bg_mean = float(np.mean(bg_region)) if bg_region.size > 0 else global_mean

                abs_diff = abs(face_mean - bg_mean)
                # Gating: When the absolute difference is less than 1.5 gray levels (out of 255)
                # or when global compression energy is low (< 3.0), the error difference is negligible
                # quantization noise, not evidence of JPEG splicing.
                if abs_diff < 1.5 or (global_mean < 3.0 and abs_diff < 2.5):
                    discrepancy_ratio = 0.0
                else:
                    # Scaled ratio using robust denominator
                    discrepancy_ratio = abs_diff / max(bg_mean, 5.0)

        # Colorized Heatmap for visualization
        scaled_diff = np.clip(diff_gray * 12.0, 0, 255).astype(np.uint8)
        colorized_map = cv2.applyColorMap(scaled_diff, cv2.COLORMAP_TURBO)
        colorized_rgb = cv2.cvtColor(colorized_map, cv2.COLOR_BGR2RGB)

        # Assess risk: requires BOTH high ratio AND meaningful absolute error difference
        if discrepancy_ratio > 0.45 and abs_diff > 3.5:
            risk = 'High'
        elif discrepancy_ratio > 0.22 and abs_diff > 2.0:
            risk = 'Moderate'
        else:
            risk = 'Low'

        return {
            'discrepancy_ratio': discrepancy_ratio,
            'global_mean': global_mean,
            'global_std': global_std,
            'face_mean': face_mean,
            'bg_mean': bg_mean,
            'abs_diff': abs_diff,
            'risk_level': risk,
            'visual_map': colorized_rgb
        }

    def _compute_noise_residual(self, gray: np.ndarray, primary_face: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Sensor noise residual (PRNU proxy): high-pass filter by subtracting median blur.
        Calibrated for photographic depth-of-field (bokeh) and flat-region noise sampling
        to prevent false positives on portrait photography.
        """
        denoised = cv2.medianBlur(gray, 3)
        residual = cv2.absdiff(gray, denoised).astype(np.float32)

        global_std = float(np.std(residual))
        face_std = global_std
        bg_std = global_std
        variance_ratio = 0.0
        is_bokeh = False

        if primary_face:
            x1, y1, x2, y2 = primary_face['bbox']
            h, w = gray.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            face_res = residual[y1:y2, x1:x2]
            mask = np.ones_like(residual, dtype=bool)
            mask[y1:y2, x1:x2] = False
            
            # Check optical depth-of-field / bokeh:
            # Measure sharpness (Laplacian variance) of in-focus face vs background
            if face_res.size > 0 and np.any(mask):
                lap_face = float(cv2.Laplacian(gray[y1:y2, x1:x2], cv2.CV_64F).var())
                lap_bg = float(cv2.Laplacian(gray, cv2.CV_64F)[mask].var())

                # If background is soft/blurred while face is sharp, bokeh is present
                if lap_bg < 200.0 and lap_face > 1.8 * max(lap_bg, 1.0):
                    is_bokeh = True

                # Estimate noise on flat regions (masking out sharp edges like glasses, eyes, jawline)
                edges = cv2.Canny(gray, 35, 95)
                flat_mask = (edges == 0)

                face_flat = residual[y1:y2, x1:x2][flat_mask[y1:y2, x1:x2]]
                bg_flat = residual[mask][flat_mask[mask]]

                face_std = float(np.std(face_flat)) if face_flat.size > 30 else float(np.std(face_res))
                bg_std = float(np.std(bg_flat)) if bg_flat.size > 30 else float(np.std(residual[mask]))

                raw_diff = abs(face_std - bg_std)

                # Optical depth of field calibration:
                # In portrait mode / bokeh, the out-of-focus background has naturally lower high-frequency noise.
                # This is standard physics, not a deepfake.
                if raw_diff < 1.8:
                    variance_ratio = 0.0
                elif is_bokeh and face_std >= bg_std:
                    # In bokeh, face having higher residual than blurred background is expected.
                    # Only flag if face noise is wildly anomalous (> 14.0 std)
                    if face_std > 14.0:
                        variance_ratio = (face_std - bg_std) / max(bg_std, 1.0) * 0.4
                    else:
                        variance_ratio = 0.05  # Normal depth of field
                elif face_std < 1.5 and bg_std > 5.0:
                    # Classic GAN artifact: face is plastic/synthetic while background has real sensor noise
                    variance_ratio = (bg_std - face_std) / (bg_std + 1e-4)
                else:
                    variance_ratio = raw_diff / max(bg_std, 2.0)

        # Visual map: normalized noise residual
        norm_res = cv2.normalize(residual, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        norm_color = cv2.cvtColor(norm_res, cv2.COLOR_GRAY2RGB)

        if variance_ratio > 0.50:
            risk = 'High'
        elif variance_ratio > 0.25:
            risk = 'Moderate'
        else:
            risk = 'Low'

        return {
            'variance_ratio': variance_ratio,
            'face_std': face_std,
            'bg_std': bg_std,
            'is_bokeh': is_bokeh,
            'risk_level': risk,
            'visual_map': norm_color
        }

    def _compute_fft_spectrum(self, gray: np.ndarray, primary_face: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        2D FFT Spectral Analysis: detects periodic grid artifacts typical of GAN/Diffusion upsamplers.
        """
        h, w = gray.shape
        if primary_face:
            x1, y1, x2, y2 = primary_face['bbox']
            patch = gray[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        else:
            patch = gray

        if patch.size < 64 * 64:
            patch = gray

        # Resize patch to power of 2 for clean FFT
        patch_resized = cv2.resize(patch, (256, 256)).astype(np.float32)
        f = np.fft.fft2(patch_resized)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift) + 1e-8
        log_mag = 20 * np.log(magnitude)

        # High-frequency analysis: check for anomalous spikes in mid-high frequencies
        center = 128
        y, x = np.ogrid[:256, :256]
        dist_from_center = np.sqrt((x - center)**2 + (y - center)**2)
        
        # High frequency ring
        hf_mask = (dist_from_center > 40) & (dist_from_center < 120)
        hf_values = log_mag[hf_mask]
        
        # Outlier spike detection (GAN checkerboard artifact test)
        q75, q25 = np.percentile(hf_values, [75, 25])
        iqr = q75 - q25
        upper_bound = q75 + (2.2 * iqr)
        spike_ratio = float(np.sum(hf_values > upper_bound) / (hf_values.size + 1e-5))
        grid_anomaly_score = min(1.0, spike_ratio * 30.0)

        hf_ratio = float(np.mean(hf_values) / (np.mean(log_mag) + 1e-4))

        # Visual spectrum map (normalized 0-255)
        spectrum_norm = cv2.normalize(log_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        spectrum_color = cv2.applyColorMap(spectrum_norm, cv2.COLORMAP_VIRIDIS)
        spectrum_rgb = cv2.cvtColor(spectrum_color, cv2.COLOR_BGR2RGB)

        if grid_anomaly_score > 0.45:
            risk = 'High'
        elif grid_anomaly_score > 0.20:
            risk = 'Moderate'
        else:
            risk = 'Low'

        return {
            'grid_anomaly_score': grid_anomaly_score,
            'hf_ratio': hf_ratio,
            'risk_level': risk,
            'visual_map': spectrum_rgb
        }

    def _compute_boundary_gradient(self, gray: np.ndarray, primary_face: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Face boundary gradient discontinuity: detects blending mask seam / feathering.
        Gated by absolute gradient step to prevent division blowup on soft regions.
        """
        h, w = gray.shape
        gradient = cv2.Laplacian(gray, cv2.CV_32F)
        abs_gradient = np.abs(gradient)

        gradient_step = 0.0
        if primary_face:
            x1, y1, x2, y2 = primary_face['bbox']
            x1, y1 = max(10, x1), max(10, y1)
            x2, y2 = min(w - 10, x2), min(h - 10, y2)

            if (x2 - x1) > 30 and (y2 - y1) > 30:
                inner_mask = np.zeros_like(gray, dtype=np.uint8)
                cv2.rectangle(inner_mask, (x1 + 6, y1 + 6), (x2 - 6, y2 - 6), 255, -1)
                
                border_mask = np.zeros_like(gray, dtype=np.uint8)
                cv2.rectangle(border_mask, (x1 - 6, y1 - 6), (x2 + 6, y2 + 6), 255, -1)
                cv2.rectangle(border_mask, (x1 + 6, y1 + 6), (x2 - 6, y2 - 6), 0, -1)

                inner_grad = float(np.mean(abs_gradient[inner_mask > 0]))
                border_grad = float(np.mean(abs_gradient[border_mask > 0]))
                
                grad_diff = abs(inner_grad - border_grad)
                if grad_diff < 2.5:
                    gradient_step = 0.0
                else:
                    gradient_step = float(grad_diff / max(border_grad, 2.0))

        # Visual gradient map
        grad_norm = np.clip(abs_gradient * 3.0, 0, 255).astype(np.uint8)
        grad_color = cv2.applyColorMap(grad_norm, cv2.COLORMAP_INFERNO)
        grad_rgb = cv2.cvtColor(grad_color, cv2.COLOR_BGR2RGB)

        if gradient_step > 0.45 and abs(inner_grad - border_grad) > 5.0:
            risk = 'High'
        elif gradient_step > 0.22 and abs(inner_grad - border_grad) > 3.0:
            risk = 'Moderate'
        else:
            risk = 'Low'

        return {
            'gradient_step': float(gradient_step),
            'risk_level': risk,
            'visual_map': grad_rgb
        }

    def _compute_dct_grid(self, gray: np.ndarray, primary_face: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Block Artifact Grid (BAG) / 8x8 DCT Compression Grid Discontinuity Analysis.
        Detects if a face swap has been spliced across a JPEG 8x8 block grid,
        causing block alignment offsets or boundary power mismatches.
        """
        h, w = gray.shape
        if h < 16 or w < 16:
            return {'grid_discrepancy': 0.0, 'risk_level': 'Low', 'visual_map': np.zeros((h, w, 3), dtype=np.uint8)}

        # Compute horizontal and vertical differences across 8-pixel block boundaries
        diff_h = np.abs(gray[1:, :] - gray[:-1, :]).astype(np.float32)
        diff_v = np.abs(gray[:, 1:] - gray[:, :-1]).astype(np.float32)

        # 8x8 block grid boundary mask
        grid_mask_h = np.zeros_like(diff_h, dtype=bool)
        grid_mask_h[7::8, :] = True

        grid_mask_v = np.zeros_like(diff_v, dtype=bool)
        grid_mask_v[:, 7::8] = True

        # Energy on block grid boundaries vs internal pixels
        on_grid_energy = (np.mean(diff_h[grid_mask_h]) + np.mean(diff_v[grid_mask_v])) / 2.0
        off_grid_energy = (np.mean(diff_h[~grid_mask_h]) + np.mean(diff_v[~grid_mask_v])) / 2.0
        
        bag_ratio = float(abs(on_grid_energy - off_grid_energy) / (off_grid_energy + 1e-4))

        # Face vs Background Grid alignment comparison if face is detected
        face_bag = bag_ratio
        if primary_face:
            x1, y1, x2, y2 = primary_face['bbox']
            f_gray = gray[y1:y2, x1:x2]
            if f_gray.shape[0] >= 16 and f_gray.shape[1] >= 16:
                fd_h = np.abs(f_gray[1:, :] - f_gray[:-1, :]).astype(np.float32)
                fd_v = np.abs(f_gray[:, 1:] - f_gray[:, :-1]).astype(np.float32)
                fm_h = np.zeros_like(fd_h, dtype=bool); fm_h[7::8, :] = True
                fm_v = np.zeros_like(fd_v, dtype=bool); fm_v[:, 7::8] = True
                f_on = (np.mean(fd_h[fm_h]) + np.mean(fd_v[fm_v])) / 2.0
                f_off = (np.mean(fd_h[~fm_h]) + np.mean(fd_v[~fm_v])) / 2.0
                face_bag = float(abs(f_on - f_off) / (f_off + 1e-4))

        grid_discrepancy = float(abs(face_bag - bag_ratio))
        is_low_res = (w < 450 or h < 450)
        high_thresh = 0.50 if is_low_res else 0.38
        mod_thresh = 0.28 if is_low_res else 0.20
        risk_level = 'High' if grid_discrepancy > high_thresh else ('Moderate' if grid_discrepancy > mod_thresh else 'Low')

        # Generate thermal visual map
        grid_vis = np.zeros((h, w), dtype=np.float32)
        grid_vis[1:, :] += diff_h * 1.2
        grid_vis[:, 1:] += diff_v * 1.2
        norm_vis = cv2.normalize(grid_vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        color_map = cv2.applyColorMap(norm_vis, cv2.COLORMAP_INFERNO)
        if primary_face:
            x1, y1, x2, y2 = primary_face['bbox']
            cv2.rectangle(color_map, (x1, y1), (x2, y2), (0, 255, 255), 1)

        return {
            'grid_discrepancy': round(grid_discrepancy, 3),
            'risk_level': risk_level,
            'visual_map': cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)
        }

    def _analyze_metadata(self, image_path: str, width: int = 0, height: int = 0) -> Dict[str, Any]:
        """
        Inspect EXIF metadata, camera hardware presence, and AI signatures.
        """
        info = {
            'has_exif': False,
            'camera_make': None,
            'camera_model': None,
            'software': None,
            'ai_marker_detected': False,
            'is_low_res': (width > 0 and height > 0 and (width < 450 or height < 450)),
            'details': []
        }

        try:
            with Image.open(image_path) as img:
                exif = img.getexif()
                if exif:
                    info['has_exif'] = True
                    for tag_id, val in exif.items():
                        tag = TAGS.get(tag_id, str(tag_id))
                        val_str = str(val).strip()
                        if tag == 'Make':
                            info['camera_make'] = val_str
                        elif tag == 'Model':
                            info['camera_model'] = val_str
                        elif tag == 'Software':
                            info['software'] = val_str
                        
                        # Check text fields for AI generator keywords
                        val_lower = val_str.lower()
                        for marker in ['midjourney', 'stable diffusion', 'comfyui', 'dall-e', 'flux', 'firefly', 'novelai', 'facefusion', 'roop']:
                            if marker in val_lower:
                                info['ai_marker_detected'] = True
                                info['details'].append(f"AI signature '{marker}' found in metadata tag '{tag}'")

                # Check raw PNG / JPEG text comments
                if hasattr(img, 'text') and isinstance(img.text, dict):
                    for k, v in img.text.items():
                        v_lower = str(v).lower()
                        for marker in ['parameters', 'prompt', 'steps:', 'sampler:', 'model_hash', 'sd_model']:
                            if marker in v_lower:
                                info['ai_marker_detected'] = True
                                info['details'].append(f"Generative AI workflow key '{k}' detected in image headers")

        except Exception as e:
            info['details'].append(f"Metadata read error: {str(e)}")

        return info

    def _compute_ensemble_score(
        self,
        ela_data: Dict[str, Any],
        noise_data: Dict[str, Any],
        fft_data: Dict[str, Any],
        boundary_data: Dict[str, Any],
        metadata_data: Dict[str, Any],
        has_face: bool,
        dct_data: Optional[Dict[str, Any]] = None,
        sightengine_result: Optional[Dict[str, Any]] = None,
        smaug_result: Optional[Dict[str, Any]] = None,
        llm_result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Weighted ensemble combining Sightengine (genai, deepfake), local Smaug fallback,
        and physical signal processing (ELA, PRNU, 2D FFT, DCT) into a 3-tier verdict:
          - >= 0.70: "Likely AI-Generated / Manipulated" (Red)
          - <= 0.30: "Likely Authentic" (Green)
          - 0.31 - 0.69: "Inconclusive / Compressed Signal" (Amber)
        """
        score_manipulation = 0.0
        indicators = []

        # 1. ELA Score (Weight 25%)
        ela_ratio = ela_data['discrepancy_ratio']
        if ela_data['risk_level'] == 'High':
            score_manipulation += 25.0
            indicators.append("Severe JPEG compression discrepancy detected between face and background (typical of face swaps).")
        elif ela_data['risk_level'] == 'Moderate':
            score_manipulation += 10.0
            indicators.append("Moderate compression level variance across image regions.")
        else:
            indicators.append("Compression error levels are uniform across regions (consistent with single-pass capture).")

        # 2. Noise Residual PRNU (Weight 20%)
        if noise_data['risk_level'] == 'High':
            score_manipulation += 20.0
            indicators.append("Significant sensor noise inconsistency: subject area exhibits artificial smoothing or mismatched noise pattern.")
        elif noise_data['risk_level'] == 'Moderate':
            score_manipulation += 8.0
            indicators.append("Mild noise pattern variance between central subject and surroundings.")
        else:
            if noise_data.get('is_bokeh', False):
                indicators.append("Consistent optical depth-of-field: subject is sharp with natural background bokeh blur.")
            else:
                indicators.append("Camera sensor noise distribution is consistent across the frame.")

        # 3. FFT Frequency Anomaly (Weight 20%)
        if fft_data['risk_level'] == 'High':
            score_manipulation += 20.0
            indicators.append("Frequency domain shows periodic grid/checkerboard upsampling peaks (characteristic of GANs/Diffusion).")
        elif fft_data['risk_level'] == 'Moderate':
            score_manipulation += 8.0
            indicators.append("Subtle high-frequency spectral deviations observed.")
        else:
            indicators.append("Natural 1/f spectral power falloff without synthetic periodic lattice spikes.")

        # 4. Boundary Seam Discontinuity (Weight 15%)
        if has_face:
            if boundary_data['risk_level'] == 'High':
                score_manipulation += 15.0
                indicators.append("Sharp gradient discontinuity at facial contour indicating blending mask seams.")
            elif boundary_data['risk_level'] == 'Moderate':
                score_manipulation += 6.0
                indicators.append("Minor gradient transition variance near face boundary.")
            else:
                indicators.append("Facial boundary contours show smooth, natural optical gradient transitions.")

        # 5. JPEG 8x8 Block Grid (DCT) Discontinuity (Weight 10%)
        if dct_data and dct_data.get('risk_level') == 'High':
            score_manipulation += 12.0
            indicators.append("Significant 8x8 JPEG block grid misalignment detected between face and surrounding image (compositing footprint).")
        elif dct_data and dct_data.get('risk_level') == 'Moderate':
            score_manipulation += 5.0
            indicators.append("Subtle block boundary grid variance observed.")
        else:
            indicators.append("JPEG 8x8 block artifact grid is continuous across the canvas.")

        # 6. Metadata Factors (Weight 10%)
        if metadata_data['ai_marker_detected']:
            score_manipulation += 35.0
            indicators.append("Definitive AI generation markers identified in image metadata.")
        elif metadata_data['camera_make'] and metadata_data['camera_model']:
            score_manipulation = max(0.0, score_manipulation - 15.0)
            indicators.append(f"Authentic camera hardware tags verified: {metadata_data['camera_make']} {metadata_data['camera_model']}.")
        elif metadata_data.get('is_low_res', False):
            indicators.append("Low resolution image detected (consistent with messaging apps like WhatsApp/Telegram or web cropping).")
        elif not metadata_data['has_exif']:
            indicators.append("EXIF metadata not present (standard for web & social media images).")

        # Base physical forensic manipulation score in [0.0, 1.0]
        forensic_norm = min(1.0, max(0.0, score_manipulation / 100.0))

        # 7. Primary Integration: Sightengine AI & Deepfake Detection (Sightengine Only API)
        if sightengine_result and sightengine_result.get('status') == 'success':
            se_ai = float(sightengine_result.get('ai_generated_score', 0.0))
            se_df = float(sightengine_result.get('deepfake_score', 0.0))
            se_combined = float(sightengine_result.get('combined_score', max(se_ai, se_df)))

            indicators.append(
                f"[Sightengine Detection] Evaluated via Sightengine API (models=genai,deepfake). Generative AI: {se_ai*100:.1f}%, Face Deepfake: {se_df*100:.1f}%."
            )

            # Pure Sightengine API determination (100% authoritative)
            final_score = se_combined

        # 8. Fallback to Smaug Local Detector (abacusai/Smaug-AI-Image-Detector)
        elif smaug_result and smaug_result.get('status') == 'success':
            smaug_val = float(smaug_result.get('score', 0.5))
            indicators.append(
                f"[Smaug Local AI Detector] Fallback active. AI generation probability: {smaug_val*100:.1f}%."
            )
            combined_score = (smaug_val * 0.70) + (forensic_norm * 0.30)
            if smaug_val >= 0.70:
                final_score = max(combined_score, smaug_val)
            elif smaug_val <= 0.20 and forensic_norm <= 0.35:
                final_score = min(combined_score, 0.30)
            else:
                final_score = combined_score

        # 9. Clean Fallback to Physical Forensics if neither is available
        else:
            if sightengine_result and sightengine_result.get('status') == 'error':
                indicators.append(f"[Sightengine Note] {sightengine_result.get('message', 'Sightengine API error')}. Cleanly fell back to physical forensics.")
            elif sightengine_result and sightengine_result.get('status') == 'skipped':
                indicators.append("[Sightengine Note] Credentials not provided; evaluated via physical forensic checks.")

            if smaug_result and smaug_result.get('message'):
                indicators.append(f"[Smaug Note] {smaug_result.get('message')}")

            final_score = forensic_norm

        # 10. Optional Multimodal LLM Vision Audit (Corroboration)
        if llm_result and llm_result.get('status') == 'success':
            for finding in llm_result.get('primary_findings', []):
                indicators.append(f"[AI Vision Audit] {finding}")
            if llm_result.get('mobile_processing_notes'):
                indicators.append(f"[Mobile Filter Notes] {llm_result['mobile_processing_notes']}")

            # If Sightengine was not active, LLM vision refines heuristic scores
            if not (sightengine_result and sightengine_result.get('status') == 'success'):
                llm_manip = float(llm_result.get('manipulation_probability', 50.0)) / 100.0
                if llm_result.get('verdict_code') == 'deepfake':
                    final_score = max(final_score, llm_manip)
                elif llm_result.get('verdict_code') == 'genuine' and final_score <= 0.40:
                    final_score = (final_score * 0.30) + (llm_manip * 0.70)

        # 11. Clamp final score and map to exact 3-tier verdicts:
        #  - >= 0.70: "Likely AI-Generated / Manipulated" (Red)
        #  - <= 0.30: "Likely Authentic" (Green)
        #  - 0.31 - 0.69: "Inconclusive / Compressed Signal" (Amber)
        final_score = round(min(0.99, max(0.01, final_score)), 3)

        if final_score >= 0.70:
            verdict = "Likely AI-Generated / Manipulated"
            verdict_code = "deepfake"
            confidence = "High" if final_score >= 0.85 else "Medium"
        elif final_score <= 0.30:
            verdict = "Likely Authentic"
            verdict_code = "genuine"
            confidence = "High" if final_score <= 0.15 else "Medium"
        else:
            verdict = "Inconclusive / Compressed Signal"
            verdict_code = "suspicious"
            confidence = "Medium"

        manipulation_probability = round(final_score * 100.0, 1)
        authenticity_score = round(100.0 - manipulation_probability, 1)

        return {
            'verdict': verdict,
            'verdict_code': verdict_code,
            'authenticity_score': authenticity_score,
            'manipulation_probability': manipulation_probability,
            'final_score': final_score,
            'confidence': confidence,
            'indicators': indicators
        }

    def _draw_face_boxes(
        self,
        image_bgr: np.ndarray,
        faces: List[Dict[str, Any]],
        per_face_data: Optional[List[Dict[str, Any]]] = None
    ) -> np.ndarray:
        for idx, f in enumerate(faces):
            x1, y1, x2, y2 = f['bbox']
            
            if per_face_data and idx < len(per_face_data):
                pfd = per_face_data[idx]
                auth_score = pfd['authenticity_score']
                vcode = pfd['verdict_code']
                if vcode == 'deepfake':
                    color = (40, 40, 240)  # Red
                    tag = f"Face #{idx+1}: Manipulated ({int(auth_score)}%)"
                elif vcode == 'suspicious':
                    color = (0, 165, 255)  # Orange
                    tag = f"Face #{idx+1}: Suspicious ({int(auth_score)}%)"
                else:
                    color = (0, 230, 100)  # Green
                    tag = f"Face #{idx+1}: Authentic ({int(auth_score)}%)"
            else:
                color = (0, 230, 100)
                tag = f"Face #{idx+1}"

            cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.52
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(tag, font, font_scale, thickness)
            
            top_y = max(24, y1 - 6)
            cv2.rectangle(image_bgr, (x1, top_y - text_h - 6), (x1 + text_w + 10, top_y + baseline + 2), (15, 23, 42), -1)
            cv2.rectangle(image_bgr, (x1, top_y - text_h - 6), (x1 + text_w + 10, top_y + baseline + 2), color, 1)
            cv2.putText(image_bgr, tag, (x1 + 5, top_y), font, font_scale, color, thickness, cv2.LINE_AA)

        return image_bgr

    def _image_to_base64(self, img_rgb: np.ndarray) -> str:
        """Convert RGB numpy array to base64 JPEG string."""
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        success, buffer = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if success:
            b64 = base64.b64encode(buffer).decode('utf-8')
            return f"data:image/jpeg;base64,{b64}"
        return ""
