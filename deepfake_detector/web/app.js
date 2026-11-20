// ==========================================================================
// DeepTrace AI — Client Application Logic
// ==========================================================================

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const dropZone = document.getElementById('drop-zone');
    const dropZoneContent = document.getElementById('drop-zone-content');
    const fileInput = document.getElementById('file-input');
    const browseBtn = document.getElementById('browse-btn');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const previewFilename = document.getElementById('preview-filename');
    const removeBtn = document.getElementById('remove-btn');
    const scanBtn = document.getElementById('scan-btn');
    const scanBtnText = document.getElementById('scan-btn-text');

    const scanProgressBox = document.getElementById('scan-progress-box');
    const progressBarFill = document.getElementById('progress-bar-fill');
    const progressStatusText = document.getElementById('progress-status-text');
    const progressPercentage = document.getElementById('progress-percentage');

    const emptyState = document.getElementById('empty-state');
    const resultsView = document.getElementById('results-view');

    // Results Elements
    const gaugeProgress = document.getElementById('gauge-progress');
    const scoreValue = document.getElementById('score-value');
    const verdictBadge = document.getElementById('verdict-badge');
    const confidenceBadge = document.getElementById('confidence-badge');
    const verdictTitle = document.getElementById('verdict-title');
    const verdictDesc = document.getElementById('verdict-desc');
    const pillFaces = document.getElementById('pill-faces');
    const pillManipProb = document.getElementById('pill-manip-prob');
    const pillResolution = document.getElementById('pill-resolution');

    // Visual Explorer Elements
    const mapTabs = document.querySelectorAll('.tab-btn');
    const activeMapImg = document.getElementById('active-map-img');
    const mapLegend = document.getElementById('map-legend');

    // Metrics Elements
    const valElaRatio = document.getElementById('val-ela-ratio');
    const badgeEla = document.getElementById('badge-ela');
    const barEla = document.getElementById('bar-ela');

    const valNoiseRatio = document.getElementById('val-noise-ratio');
    const badgeNoise = document.getElementById('badge-noise');
    const barNoise = document.getElementById('bar-noise');

    const valFftScore = document.getElementById('val-fft-score');
    const badgeFft = document.getElementById('badge-fft');
    const barFft = document.getElementById('bar-fft');

    const valBoundaryStep = document.getElementById('val-boundary-step');
    const badgeBoundary = document.getElementById('badge-boundary');
    const barBoundary = document.getElementById('bar-boundary');

    const valDctDiscrepancy = document.getElementById('val-dct-discrepancy');
    const badgeDct = document.getElementById('badge-dct');
    const barDct = document.getElementById('bar-dct');

    const findingsList = document.getElementById('findings-list');
    const exportJsonBtn = document.getElementById('export-json-btn');

    // AI Vision Results Elements (shown in findings if analysis is present)
    const aiResultsCard = document.getElementById('ai-results-card');
    const aiModelLabel = document.getElementById('ai-model-label');
    const aiVerdictTag = document.getElementById('ai-verdict-tag');
    const aiAuditContent = document.getElementById('ai-audit-content');

    // AI Provider & Key Settings UI Elements
    const uiLlmProvider = document.getElementById('ui-llm-provider');
    const uiApiKeyContainer = document.getElementById('ui-api-key-container');
    const uiLlmKey = document.getElementById('ui-llm-key');
    const aiActiveIndicator = document.getElementById('ai-active-indicator');
    const uiKeyHint = document.getElementById('ui-key-hint');

    if (uiLlmProvider) {
        uiLlmProvider.addEventListener('change', () => {
            const val = uiLlmProvider.value;
            if (val === 'offline') {
                if (uiApiKeyContainer) uiApiKeyContainer.style.display = 'none';
                if (aiActiveIndicator) {
                    aiActiveIndicator.textContent = 'Disabled';
                    aiActiveIndicator.style.color = '#94a3b8';
                    aiActiveIndicator.style.background = 'rgba(148, 163, 184, 0.15)';
                }
            } else if (val === 'nvidia') {
                if (uiApiKeyContainer) uiApiKeyContainer.style.display = 'none';
                if (aiActiveIndicator) {
                    aiActiveIndicator.textContent = 'NVIDIA Active';
                    aiActiveIndicator.style.color = '#10b981';
                    aiActiveIndicator.style.background = 'rgba(16, 185, 129, 0.15)';
                }
            } else {
                if (uiApiKeyContainer) uiApiKeyContainer.style.display = 'block';
                if (aiActiveIndicator) {
                    aiActiveIndicator.textContent = val.toUpperCase() + ' Mode';
                    aiActiveIndicator.style.color = '#38bdf8';
                    aiActiveIndicator.style.background = 'rgba(56, 189, 248, 0.15)';
                }
                if (val === 'gemini') {
                    if (uiLlmKey) uiLlmKey.placeholder = 'Enter Google Gemini API Key (from Google AI Studio)';
                    if (uiKeyHint) uiKeyHint.textContent = 'Free at aistudio.google.com — or leave blank if in .env';
                } else if (val === 'openai') {
                    if (uiLlmKey) uiLlmKey.placeholder = 'Enter OpenAI API Key (sk-...)';
                    if (uiKeyHint) uiKeyHint.textContent = 'Leave blank to use OPENAI_API_KEY from .env';
                }
            }
        });
    }

    // Check API Health and Detector Status
    const seStatusBadge = document.getElementById('sightengine-status-badge');
    fetch('/api/health')
        .then(res => res.json())
        .then(data => {
            if (seStatusBadge) {
                if (data.sightengine_active) {
                    seStatusBadge.textContent = 'Active (.env Loaded)';
                    seStatusBadge.style.color = '#10b981';
                    seStatusBadge.style.background = 'rgba(16, 185, 129, 0.15)';
                } else {
                    seStatusBadge.textContent = 'Fallback Mode (Smaug / Forensics)';
                    seStatusBadge.style.color = '#38bdf8';
                    seStatusBadge.style.background = 'rgba(56, 189, 248, 0.15)';
                }
            }
        })
        .catch(err => console.warn('Health check error:', err));

    // App State
    let currentFile = null;
    let currentAnalysis = null;

    const MAP_LEGENDS = {
        'annotated': '<strong>Face Detection:</strong> Shows localized bounding boxes and detector confidence.',
        'ela_map': '<strong>Error Level Analysis (ELA):</strong> Multi-colored thermal map. High-intensity areas (red/white) show spliced or altered regions with differing compression.',
        'noise_map': '<strong>Sensor Noise (PRNU):</strong> Camera sensor grain. Smooth patches on a noisy image indicate synthetic smoothing or AI regeneration.',
        'fft_map': '<strong>2D FFT Spectrum:</strong> Frequency domain representation. Star/grid spikes away from center reveal synthetic upsampling artifacts.',
        'boundary_map': '<strong>Boundary Seam:</strong> Edge gradients along facial contours. Highlights blending seams from face swap feathering.',
        'dct_map': '<strong>JPEG Block Grid (DCT):</strong> 8x8 compression blocking. Detects grid displacement or misalignment caused by splicing face swaps into host images.'
    };

    // --------------------------------------------------------------------------
    // File Selection & Drag-and-Drop
    // --------------------------------------------------------------------------
    browseBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    dropZone.addEventListener('click', () => {
        if (!currentFile) fileInput.click();
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('dragover');
        });
    });

    dropZone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files && files.length > 0) {
            handleFileSelect(files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        clearSelectedFile();
    });

    function handleFileSelect(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select a valid image file (JPEG, PNG, WEBP).');
            return;
        }

        currentFile = file;
        previewFilename.textContent = file.name;

        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            dropZoneContent.style.display = 'none';
            previewContainer.style.display = 'flex';
            scanBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    function clearSelectedFile() {
        currentFile = null;
        fileInput.value = '';
        imagePreview.src = '';
        dropZoneContent.style.display = 'flex';
        previewContainer.style.display = 'none';
        scanBtn.disabled = true;
        scanProgressBox.style.display = 'none';
    }

    // --------------------------------------------------------------------------
    // Scan Execution
    // --------------------------------------------------------------------------
    scanBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        scanBtn.disabled = true;
        scanBtnText.textContent = 'Analyzing Forensic Signatures...';
        scanProgressBox.style.display = 'flex';
        simulateProgressStages();

        const formData = new FormData();
        formData.append('image', currentFile);

        // Sightengine credentials override from UI
        const seUser = document.getElementById('ui-sightengine-user');
        const seSecret = document.getElementById('ui-sightengine-secret');
        if (seUser && seUser.value.trim()) {
            formData.append('sightengine_user', seUser.value.trim());
        }
        if (seSecret && seSecret.value.trim()) {
            formData.append('sightengine_secret', seSecret.value.trim());
        }

        if (uiLlmProvider && uiLlmProvider.value !== 'offline') {
            formData.append('llm_provider', uiLlmProvider.value);
            if (uiLlmKey && uiLlmKey.value.trim()) {
                formData.append('llm_key', uiLlmKey.value.trim());
            }
        }

        try {
            const response = await fetch('/api/analyze', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.error || 'Forensic analysis failed');
            }

            const data = await response.json();
            currentAnalysis = data;

            // Complete progress bar
            setScanProgress(100, 'Analysis Complete');
            setTimeout(() => {
                scanProgressBox.style.display = 'none';
                scanBtn.disabled = false;
                scanBtnText.textContent = 'Run Deep Forensic Scan';
                renderResults(data);
            }, 600);

        } catch (err) {
            alert('Error running scan: ' + err.message);
            scanBtn.disabled = false;
            scanBtnText.textContent = 'Run Deep Forensic Scan';
            scanProgressBox.style.display = 'none';
        }
    });

    function simulateProgressStages() {
        const stages = [
            { pct: 20, text: 'Connecting to Sightengine AI Detection API...', tag: 'stage-ai' },
            { pct: 50, text: 'Querying models: genai, deepfake...', tag: 'stage-ai' },
            { pct: 75, text: 'Generating diagnostic visual maps (ELA, Noise, FFT)...', tag: 'stage-ela' },
            { pct: 95, text: 'Finalizing 3-Tier Sightengine verdict...', tag: 'stage-seam' }
        ];

        stages.forEach((stage, idx) => {
            setTimeout(() => {
                if (scanBtn.disabled && currentFile) {
                    setScanProgress(stage.pct, stage.text);
                    document.querySelectorAll('.stage-tag').forEach(t => t.classList.remove('active'));
                    const el = document.getElementById(stage.tag);
                    if (el) el.classList.add('active');
                }
            }, (idx + 1) * 350);
        });
    }

    function setScanProgress(pct, text) {
        progressBarFill.style.width = `${pct}%`;
        progressPercentage.textContent = `${pct}%`;
        progressStatusText.textContent = text;
    }

    // --------------------------------------------------------------------------
    // Results Rendering
    // --------------------------------------------------------------------------
    function renderResults(data) {
        emptyState.style.display = 'none';
        resultsView.style.display = 'flex';

        const { summary, indicators, metrics, visual_maps } = data;

        // 1. Authenticity Score & Gauge
        const authScore = summary.authenticity_score;
        scoreValue.textContent = `${Math.round(authScore)}%`;

        // Circumference is 2 * PI * 42 ≈ 263.89
        const circumference = 264;
        const offset = circumference - (authScore / 100) * circumference;
        gaugeProgress.style.strokeDashoffset = offset;

        // Color gauge and header based on 3-tier verdict
        let strokeColor = '#10b981'; // genuine (<= 0.30)
        verdictBadge.className = 'verdict-badge';

        if (summary.verdict_code === 'deepfake') {
            strokeColor = '#f43f5e'; // Red
            verdictBadge.classList.add('badge-deepfake');
            verdictBadge.textContent = 'LIKELY AI-GENERATED / MANIPULATED';
            verdictTitle.textContent = 'Likely AI-Generated / Manipulated';
            verdictDesc.textContent = 'High probability (>= 70%) of generative AI synthesis, face swapping, or digital manipulation detected.';
        } else if (summary.verdict_code === 'suspicious') {
            strokeColor = '#f59e0b'; // Amber
            verdictBadge.classList.add('badge-suspicious');
            verdictBadge.textContent = 'INCONCLUSIVE / COMPRESSED SIGNAL';
            verdictTitle.textContent = 'Inconclusive / Compressed Signal';
            verdictDesc.textContent = 'Intermediate signal range (31% - 69%). Heavy compression or ambiguous forensic cues detected. Manual review recommended.';
        } else {
            strokeColor = '#10b981'; // Green
            verdictBadge.classList.add('badge-genuine');
            verdictBadge.textContent = 'LIKELY AUTHENTIC';
            verdictTitle.textContent = 'Likely Authentic Photo';
            verdictDesc.textContent = 'Low manipulation risk (<= 30%). Uniform sensor noise grain, authentic facial anatomy, and consistent compression.';
        }
        gaugeProgress.style.stroke = strokeColor;
        confidenceBadge.textContent = `${summary.confidence} Confidence`;

        pillFaces.textContent = `Faces: ${summary.face_count}`;
        pillManipProb.textContent = `Deepfake Risk: ${summary.manipulation_probability}%`;
        pillResolution.textContent = summary.resolution;

        // 2. Setup Visual Explorer Maps
        setActiveVisualMap('annotated', visual_maps);
        mapTabs.forEach(btn => {
            btn.classList.remove('active');
            if (btn.dataset.map === 'annotated') btn.classList.add('active');
        });

        // 3. Populate Metric Cards
        // ELA
        valElaRatio.textContent = metrics.ela.discrepancy_ratio;
        updateRiskBadge(badgeEla, barEla, metrics.ela.risk_level, metrics.ela.discrepancy_ratio, 0.45);

        // Noise
        valNoiseRatio.textContent = metrics.noise.noise_variance_ratio;
        updateRiskBadge(badgeNoise, barNoise, metrics.noise.risk_level, metrics.noise.noise_variance_ratio, 0.40);

        // FFT
        valFftScore.textContent = metrics.frequency.spectral_grid_anomaly;
        updateRiskBadge(badgeFft, barFft, metrics.frequency.risk_level, metrics.frequency.spectral_grid_anomaly, 0.45);

        // Boundary Seam
        valBoundaryStep.textContent = metrics.boundary.seam_gradient_step;
        updateRiskBadge(badgeBoundary, barBoundary, metrics.boundary.risk_level, metrics.boundary.seam_gradient_step, 0.45);

        // JPEG Block Grid (DCT)
        if (metrics.dct && valDctDiscrepancy) {
            valDctDiscrepancy.textContent = metrics.dct.grid_discrepancy;
            updateRiskBadge(badgeDct, barDct, metrics.dct.risk_level, metrics.dct.grid_discrepancy, 0.40);
        }

        // 4. Evidence Findings List
        findingsList.innerHTML = '';
        indicators.forEach(item => {
            const li = document.createElement('li');
            li.textContent = item;
            findingsList.appendChild(li);
        });

        // 4b. Render Sightengine Card if active
        const seCard = document.getElementById('sightengine-results-card');
        const seScoreBadge = document.getElementById('sightengine-score-badge');
        const seGenaiVal = document.getElementById('se-genai-val');
        const seDeepfakeVal = document.getElementById('se-deepfake-val');
        const seCombinedVal = document.getElementById('se-combined-val');
        const seTitle = document.getElementById('sightengine-verdict-title');

        if (data.sightengine && data.sightengine.status === 'success' && seCard) {
            seCard.style.display = 'block';
            const se = data.sightengine;
            const genaiPct = Math.round((se.ai_generated_score || 0.0) * 100);
            const dfPct = Math.round((se.deepfake_score || 0.0) * 100);
            const combPct = Math.round((se.combined_score || 0.0) * 100);

            if (seGenaiVal) seGenaiVal.textContent = `${genaiPct}%`;
            if (seDeepfakeVal) seDeepfakeVal.textContent = `${dfPct}%`;
            if (seCombinedVal) seCombinedVal.textContent = `${combPct}%`;

            if (seScoreBadge) {
                seScoreBadge.className = 'verdict-tag';
                if (combPct >= 70) {
                    seScoreBadge.classList.add('tag-deepfake');
                    seScoreBadge.textContent = `${combPct}% Manipulated`;
                } else if (combPct >= 31) {
                    seScoreBadge.classList.add('tag-suspicious');
                    seScoreBadge.textContent = `${combPct}% Inconclusive`;
                } else {
                    seScoreBadge.classList.add('tag-genuine');
                    seScoreBadge.textContent = `${combPct}% Authentic`;
                }
            }
        } else if (seCard) {
            seCard.style.display = 'none';
        }

        // 5. Render AI Vision Audit Card if available
        if (data.llm_audit && data.llm_audit.status === 'success' && aiResultsCard) {
            aiResultsCard.style.display = 'flex';
            const llm = data.llm_audit;
            if (aiModelLabel) {
                aiModelLabel.textContent = `${llm.provider} (${llm.model}) Semantic Audit`;
            }
            if (aiVerdictTag) {
                aiVerdictTag.className = 'verdict-tag';
                if (llm.verdict_code === 'deepfake') {
                    aiVerdictTag.classList.add('tag-deepfake');
                    aiVerdictTag.textContent = 'AI Generated / Deepfake';
                } else if (llm.verdict_code === 'suspicious') {
                    aiVerdictTag.classList.add('tag-suspicious');
                    aiVerdictTag.textContent = 'Suspicious / Inconclusive';
                } else {
                    aiVerdictTag.classList.add('tag-genuine');
                    aiVerdictTag.textContent = 'No Confirmed Manipulation';
                }
            }

            if (aiAuditContent) {
                aiAuditContent.innerHTML = '';
                (llm.primary_findings || []).forEach(pt => {
                    const div = document.createElement('div');
                    div.className = 'ai-audit-point';
                    div.textContent = pt;
                    aiAuditContent.appendChild(div);
                });

                if (llm.mobile_processing_notes) {
                    const noteBox = document.createElement('div');
                    noteBox.className = 'ai-note-box';
                    noteBox.innerHTML = `<strong>Mobile Camera / Filter Notes:</strong> ${llm.mobile_processing_notes}`;
                    aiAuditContent.appendChild(noteBox);
                }
            }
        } else if (aiResultsCard) {
            aiResultsCard.style.display = 'none';
        }

        // Scroll smoothly to results
        resultsView.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    function updateRiskBadge(badgeEl, barEl, riskLevel, val, maxVal) {
        badgeEl.textContent = `${riskLevel} Risk`;
        badgeEl.className = 'risk-badge';
        barEl.className = 'metric-bar-fill';

        const fillPct = Math.min(100, Math.max(10, Math.round((val / maxVal) * 100)));
        barEl.style.width = `${fillPct}%`;

        if (riskLevel === 'High') {
            badgeEl.classList.add('risk-high');
            barEl.classList.add('fill-high');
        } else if (riskLevel === 'Moderate') {
            badgeEl.classList.add('risk-moderate');
            barEl.classList.add('fill-moderate');
        } else {
            badgeEl.classList.add('risk-low');
            barEl.classList.add('fill-low');
        }
    }

    // --------------------------------------------------------------------------
    // Map Tab Switching
    // --------------------------------------------------------------------------
    mapTabs.forEach(btn => {
        btn.addEventListener('click', () => {
            if (!currentAnalysis) return;
            mapTabs.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            setActiveVisualMap(btn.dataset.map, currentAnalysis.visual_maps);
        });
    });

    function setActiveVisualMap(mapKey, maps) {
        if (maps && maps[mapKey]) {
            activeMapImg.src = maps[mapKey];
            mapLegend.innerHTML = MAP_LEGENDS[mapKey] || '';
        }
    }

    // --------------------------------------------------------------------------
    // Export Report
    // --------------------------------------------------------------------------
    exportJsonBtn.addEventListener('click', () => {
        if (!currentAnalysis) return;
        
        // Export data excluding heavy base64 maps for clean report
        const exportData = {
            filename: currentFile ? currentFile.name : 'image.jpg',
            timestamp: new Date().toISOString(),
            summary: currentAnalysis.summary,
            indicators: currentAnalysis.indicators,
            metrics: currentAnalysis.metrics
        };

        const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `deeptrace_report_${Date.now()}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    });
});
