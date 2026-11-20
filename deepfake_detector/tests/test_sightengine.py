import unittest
from unittest.mock import patch, MagicMock
from deepfake_detector.core.sightengine_detector import SightengineDetector
from deepfake_detector.core.forensics import DeepfakeDetector


class TestSightengineIntegration(unittest.TestCase):
    def test_missing_credentials(self):
        detector = SightengineDetector(api_user="", api_secret="")
        res = detector.detect("fake.jpg", api_user="", api_secret="")
        self.assertEqual(res['status'], 'skipped')
        self.assertFalse(res['enabled'])

    def test_mock_sightengine_response(self):
        detector = SightengineDetector('test_user', 'test_secret')
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            'status': 'success',
            'type': {'ai_generated': 0.85},
            'faces': [
                {'attributes': {'deepfake': 0.12}},
                {'attributes': {'deepfake': 0.94}}
            ]
        }
        with patch('requests.post', return_value=mock_resp):
            res = detector.detect(
                r'C:\Users\subha\.gemini\antigravity\brain\bb7b2f2f-1904-4458-ab78-0ccf2427a752\.user_uploaded\media_1795105488906_5489271f.jpg'
            )
            self.assertEqual(res['status'], 'success')
            self.assertEqual(res['ai_generated_score'], 0.85)
            self.assertEqual(res['deepfake_score'], 0.94)
            self.assertEqual(res['combined_score'], 0.94)

    def test_3tier_verdicts(self):
        fd = DeepfakeDetector()
        
        # Test Red tier (>= 0.70)
        s_red = fd._compute_ensemble_score(
            ela_data={'risk_level': 'High', 'discrepancy_ratio': 0.6},
            noise_data={'risk_level': 'High', 'noise_variance_ratio': 0.5},
            fft_data={'risk_level': 'High', 'spectral_grid_anomaly': 0.6},
            boundary_data={'risk_level': 'High', 'gradient_step': 0.5},
            metadata_data={'has_exif': False, 'camera_make': None, 'camera_model': None, 'ai_marker_detected': True},
            has_face=True,
            dct_data={'risk_level': 'High', 'block_variance': 0.5},
            sightengine_result={'status': 'success', 'combined_score': 0.92, 'ai_generated_score': 0.92, 'deepfake_score': 0.80}
        )
        self.assertEqual(s_red['verdict'], 'Likely AI-Generated / Manipulated')
        self.assertGreaterEqual(s_red['final_score'], 0.70)

        # Test Green tier (<= 0.30)
        s_green = fd._compute_ensemble_score(
            ela_data={'risk_level': 'Low', 'discrepancy_ratio': 0.05},
            noise_data={'risk_level': 'Low', 'noise_variance_ratio': 0.05},
            fft_data={'risk_level': 'Low', 'spectral_grid_anomaly': 0.05},
            boundary_data={'risk_level': 'Low', 'gradient_step': 0.05},
            metadata_data={'has_exif': True, 'camera_make': 'Apple', 'camera_model': 'iPhone', 'ai_marker_detected': False},
            has_face=True,
            dct_data={'risk_level': 'Low', 'block_variance': 0.05},
            sightengine_result={'status': 'success', 'combined_score': 0.05, 'ai_generated_score': 0.05, 'deepfake_score': 0.02}
        )
        self.assertEqual(s_green['verdict'], 'Likely Authentic')
        self.assertLessEqual(s_green['final_score'], 0.30)

        # Test Amber tier (0.31 - 0.69)
        s_amber = fd._compute_ensemble_score(
            ela_data={'risk_level': 'Moderate', 'discrepancy_ratio': 0.25},
            noise_data={'risk_level': 'Moderate', 'noise_variance_ratio': 0.25},
            fft_data={'risk_level': 'Moderate', 'spectral_grid_anomaly': 0.25},
            boundary_data={'risk_level': 'Moderate', 'gradient_step': 0.25},
            metadata_data={'has_exif': False, 'camera_make': None, 'camera_model': None, 'ai_marker_detected': False},
            has_face=True,
            dct_data={'risk_level': 'Moderate', 'block_variance': 0.25},
            sightengine_result={'status': 'success', 'combined_score': 0.50, 'ai_generated_score': 0.50, 'deepfake_score': 0.45}
        )
        self.assertEqual(s_amber['verdict'], 'Inconclusive / Compressed Signal')
        self.assertTrue(0.31 <= s_amber['final_score'] <= 0.69)


if __name__ == '__main__':
    unittest.main()
