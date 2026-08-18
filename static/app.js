document.addEventListener('DOMContentLoaded', () => {
    // Current state
    let activeTab = 'overview';
    let activeModel = 'roberta';
    let aspectChart = null;

    // Elements
    const navButtons = document.querySelectorAll('.nav-btn');
    const tabPanels = document.querySelectorAll('.tab-panel');
    const modelSelect = document.getElementById('backend-model-select');
    const refreshBiBtn = document.getElementById('btn-refresh-bi');
    const exportBtn = document.getElementById('btn-export-report');
    const exportFormatSelect = document.getElementById('export-format-select');

    // Top Bar Headers
    const pageTitle = document.getElementById('page-title');
    const pageSubtitle = document.getElementById('page-subtitle');

    // Overview Elements
    const kpiTotal = document.getElementById('kpi-total-reviews');
    const kpiPosPct = document.getElementById('kpi-positive-pct');
    const kpiNegPct = document.getElementById('kpi-negative-pct');
    const kpiPosFill = document.getElementById('kpi-positive-fill');
    const kpiNegFill = document.getElementById('kpi-negative-fill');
    const kpiSatisfaction = document.getElementById('kpi-satisfaction');
    const complaintBadge = document.getElementById('complaint-badge');
    const aspectsTableBody = document.querySelector('#aspects-table tbody');
    const recsList = document.getElementById('ai-recommendations-list');
    const posKeywords = document.getElementById('positive-keywords');
    const negKeywords = document.getElementById('negative-keywords');

    // Inference Elements
    const reviewInput = document.getElementById('review-input');
    const submitInferenceBtn = document.getElementById('btn-submit-inference');
    const clearInferenceBtn = document.getElementById('btn-clear-inference');
    const resultsPanel = document.getElementById('inference-results-panel');
    const initialMsg = resultsPanel.querySelector('.initial-msg');
    const resultsView = resultsPanel.querySelector('.results-view');
    const batchResultsDiv = document.getElementById('batch-results');
    const pillSentiment = document.getElementById('result-sentiment-pill');
    const confidenceScore = document.getElementById('result-score-val');
    const gaugeFill = document.getElementById('result-gauge-fill');
    const aspectBadges = document.getElementById('result-aspect-badges');
    const textPreview = document.getElementById('result-text-preview');
    const resultModelName = document.getElementById('result-model-name');

    // Agent Elements
    const agentInput = document.getElementById('agent-input');
    const askAgentBtn = document.getElementById('btn-ask-agent');
    const agentInitial = document.getElementById('agent-initial');
    const agentResults = document.getElementById('agent-results');
    const agentToolName = document.getElementById('agent-tool-name');
    const agentModelUsed = document.getElementById('agent-model-used');
    const agentAnswerText = document.getElementById('agent-answer-text');
    const agentSupportingData = document.getElementById('agent-supporting-data');
    const agentRecsList = document.getElementById('agent-recs-list');

    // Upload Elements
    const uploadZone = document.getElementById('upload-zone');
    const csvFileInput = document.getElementById('csv-file-input');
    const uploadStatus = document.getElementById('upload-status');

    // Model Performance Elements
    const metricsGrid = document.getElementById('models-metrics-cards');
    const metricsNote = document.getElementById('metrics-note');

    // BI cached data
    let cachedBiData = null;

    // ----------------- TAB NAVIGATION -----------------
    navButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.getAttribute('data-tab');

            navButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            tabPanels.forEach(p => p.classList.remove('active'));
            document.getElementById(`tab-${targetTab}`).classList.add('active');

            activeTab = targetTab;
            updateHeaderTitles();

            if (activeTab === 'overview') {
                loadBiReport();
            } else if (activeTab === 'models') {
                loadModelMetrics();
            } else if (activeTab === 'insights') {
                loadInsights();
            }
        });
    });

    function updateHeaderTitles() {
        const headerConfig = {
            'overview': {
                title: 'Sentiment & Business Intelligence Dashboard',
                sub: 'Actionable customer insights from fine-tuned transformer models',
                showRefresh: true, showExport: true
            },
            'agent': {
                title: 'AI Agent — Natural Language Query Engine',
                sub: 'Ask business questions and get data-driven answers from the AI Agent',
                showRefresh: false, showExport: false
            },
            'inference': {
                title: 'Interactive Review Sentiment Analysis',
                sub: 'Test single reviews or upload CSV files for bulk analysis',
                showRefresh: false, showExport: false
            },
            'models': {
                title: 'Model Evaluation & Comparison',
                sub: 'Fine-tuned model benchmarks on Amazon Reviews test split',
                showRefresh: false, showExport: false
            },
            'insights': {
                title: 'Business Intelligence Insights',
                sub: 'Detailed aspect breakdowns and data-driven recommendations',
                showRefresh: true, showExport: true
            }
        };

        const cfg = headerConfig[activeTab] || headerConfig['overview'];
        pageTitle.innerText = cfg.title;
        pageSubtitle.innerText = cfg.sub;
        refreshBiBtn.classList.toggle('hidden', !cfg.showRefresh);
        exportBtn.classList.toggle('hidden', !cfg.showExport);
    }

    // ----------------- MODEL SELECTION -----------------
    modelSelect.addEventListener('change', (e) => {
        activeModel = e.target.value;
        if (activeTab === 'overview' || activeTab === 'insights') {
            loadBiReport();
        }
    });

    refreshBiBtn.addEventListener('click', () => {
        loadBiReport();
    });

    // ----------------- EXPORT -----------------
    exportFormatSelect.addEventListener('change', (e) => {
        exportBtn.dataset.format = e.target.value;
    });
    exportBtn.dataset.format = 'json'; // default
    exportBtn.addEventListener('click', async () => {
        // Prompt user for format
        const format = (exportBtn.dataset.format || 'json');
        try {
            const response = await fetch(`/api/export?format=${format}&model=${activeModel}`);
            if (!response.ok) throw new Error('Export failed');
            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `bi_report.${format}`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        } catch (e) {
            console.error(e);
            alert('Export failed. Please ensure the server is running and BI data is available.');
        }
    });

    // ----------------- BI REPORT -----------------
    async function loadBiReport() {
        try {
            recsList.innerHTML = `<div class="loading-state"><i class="fa-solid fa-spinner fa-spin"></i> Analyzing reviews with ${activeModel}...</div>`;
            aspectsTableBody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: var(--text-muted)"><i class="fa-solid fa-spinner fa-spin"></i> Running sentiment analysis...</td></tr>`;

            const response = await fetch(`/api/bi-report?model=${activeModel}&limit=400`);
            if (!response.ok) {
                const err = await response.json().catch(() => ({}));
                throw new Error(err.detail || 'Failed to load BI report');
            }

            const data = await response.json();
            cachedBiData = data;
            renderBiReport(data);
        } catch (e) {
            console.error(e);
            recsList.innerHTML = `<div class="rec-item" style="border-left-color: var(--text-danger)">
                <p>${e.message || 'Failed to load BI report. Ensure models are trained and the server is running.'}</p>
            </div>`;
            aspectsTableBody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: var(--text-danger)">${e.message}</td></tr>`;
        }
    }

    function renderBiReport(data) {
        // KPI values
        kpiTotal.innerText = data.summary.total_processed.toLocaleString();
        kpiPosPct.innerText = `${data.summary.positive_ratio}%`;
        kpiNegPct.innerText = `${data.summary.negative_ratio}%`;
        kpiPosFill.style.width = `${data.summary.positive_ratio}%`;
        kpiNegFill.style.width = `${data.summary.negative_ratio}%`;
        kpiSatisfaction.innerText = data.summary.overall_satisfaction || '-';
        complaintBadge.innerText = `Top Complaint: ${data.summary.most_frequent_complaint || 'None'}`;

        // Populate new KPIs
        const kpiBestModel = document.getElementById('kpi-best-model');
        const kpiMostNegAspect = document.getElementById('kpi-most-negative-aspect');
        const kpiMostFreqComplaint = document.getElementById('kpi-most-frequent-complaint');

        if (kpiBestModel) kpiBestModel.innerText = data.summary.best_model || (data.model_used || activeModel).toUpperCase();
        if (kpiMostNegAspect) kpiMostNegAspect.innerText = data.summary.most_negative_aspect || '-';
        if (kpiMostFreqComplaint) kpiMostFreqComplaint.innerText = data.summary.most_frequent_complaint || '-';

        // Executive Insight (uses dynamic executive_insight object from backend)
        const execInsightDoc = document.getElementById('executive-insight-content');
        if (execInsightDoc) {
            const hasData = data.summary.total_processed > 0;
            const ei = data.executive_insight || {};
            if (hasData) {
                execInsightDoc.innerHTML = `
                    <p style="margin: 0 0 10px 0;">Based on a real-time BI analysis of <strong>${data.summary.total_processed}</strong> preprocessed customer reviews using <strong>${(data.model_used || activeModel).toUpperCase()}</strong>:</p>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 5px;">
                        <span style="background: rgba(255,255,255,0.02); padding: 8px 12px; border-radius: 4px; border-left: 3px solid var(--text-success);">
                            <strong>Overall Sentiment:</strong> ${ei.overall_sentiment || (data.summary.positive_ratio + '% Positive')}
                        </span>
                        <span style="background: rgba(255,255,255,0.02); padding: 8px 12px; border-radius: 4px; border-left: 3px solid var(--text-danger);">
                            <strong>Major Complaint:</strong> ${ei.major_complaint || data.summary.most_frequent_complaint}
                        </span>
                        <span style="background: rgba(255,255,255,0.02); padding: 8px 12px; border-radius: 4px; border-left: 3px solid var(--primary);">
                            <strong>Strongest Positive:</strong> ${ei.strongest_positive_aspect || 'N/A'}
                        </span>
                        <span style="background: rgba(255,255,255,0.02); padding: 8px 12px; border-radius: 4px; border-left: 3px solid var(--text-warning);">
                            <strong>Action:</strong> ${ei.recommended_action || 'N/A'}
                        </span>
                    </div>
                `;
            } else {
                execInsightDoc.innerHTML = `<p style="margin: 0; color: var(--text-muted)">No review data has been processed yet. Train models and load datasets to receive business recommendations.</p>`;
            }
        }

        // Business Impact list
        const impactSatisfaction = document.getElementById('impact-satisfaction');
        const impactNegPct = document.getElementById('impact-neg-pct');
        const impactComplaint = document.getElementById('impact-complaint');
        const impactStrongest = document.getElementById('impact-strongest');
        const impactPriority = document.getElementById('impact-priority');

        if (impactSatisfaction) impactSatisfaction.innerText = data.summary.overall_satisfaction || 'N/A';
        if (impactNegPct) impactNegPct.innerText = `${data.summary.negative_ratio}%`;
        if (impactComplaint) impactComplaint.innerText = data.summary.most_frequent_complaint || 'None';

        // Find aspect with highest positive %
        let strongest = 'None';
        if (data.aspect_analysis && data.aspect_analysis.length > 0) {
            let maxPos = -1;
            data.aspect_analysis.forEach(a => {
                if (a.positive_pct > maxPos) {
                    maxPos = a.positive_pct;
                    strongest = a.aspect;
                }
            });
        }
        if (impactStrongest) impactStrongest.innerText = strongest;
        if (impactPriority) {
            impactPriority.innerText = data.summary.most_negative_aspect || 'None';
            impactPriority.style.color = 'var(--text-danger)';
        }

        // Color satisfaction
        const sat = (data.summary.overall_satisfaction || '').toLowerCase();
        if (sat === 'excellent' || sat === 'good') {
            kpiSatisfaction.className = 'kpi-val text-success';
        } else if (sat === 'fair') {
            kpiSatisfaction.className = 'kpi-val text-warning';
        } else {
            kpiSatisfaction.className = 'kpi-val text-danger';
        }

        // Aspect table (6 columns)
        aspectsTableBody.innerHTML = '';
        data.aspect_analysis.forEach(item => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>
                    <strong>${item.aspect}</strong><br>
                    <span style="font-size: 11px; color: var(--text-muted)">${item.description}</span>
                </td>
                <td>${item.total_mentions}</td>
                <td>${item.positive_count}</td>
                <td><span class="text-success">${item.positive_pct}%</span></td>
                <td>${item.negative_count}</td>
                <td><span class="text-danger">${item.negative_pct}%</span></td>
            `;
            aspectsTableBody.appendChild(tr);
        });

        // Aspect Chart
        if (aspectChart) aspectChart.destroy();
        const chartCanvas = document.getElementById('chart-aspects');
        if (typeof Chart !== 'undefined' && chartCanvas) {
            const ctx = chartCanvas.getContext('2d');
            const labels = data.aspect_analysis.map(item => item.aspect);
            const posScores = data.aspect_analysis.map(item => item.positive_pct);
            const negScores = data.aspect_analysis.map(item => item.negative_pct);

            aspectChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [
                        { label: 'Positive %', data: posScores, backgroundColor: '#10b981', borderRadius: 4 },
                        { label: 'Negative %', data: negScores, backgroundColor: '#ef4444', borderRadius: 4 }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    indexAxis: 'y',
                    scales: {
                        x: { stacked: true, max: 100, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } },
                        y: { stacked: true, grid: { display: false }, ticks: { color: '#94a3b8' } }
                    },
                    plugins: {
                        legend: { position: 'top', labels: { color: '#94a3b8', font: { family: 'Inter' } } }
                    }
                }
            });
        }

        // AI Recommendations (removed ** headings)
        recsList.innerHTML = '';
        data.agent_recommendations.forEach(rec => {
            const div = document.createElement('div');
            div.className = 'rec-item';

            // Clean markdown bold tags if any slipped in
            const cleanRec = rec.replace(/\*\*/g, '');

            if (cleanRec.includes('⚠️') || cleanRec.includes('💲') || cleanRec.includes('📦') || cleanRec.includes('⚙️')) {
                div.style.borderLeftColor = 'var(--text-warning)';
            } else {
                div.style.borderLeftColor = 'var(--text-success)';
            }
            div.innerHTML = `<p>${cleanRec}</p>`;
            recsList.appendChild(div);
        });

        // Keywords
        posKeywords.innerHTML = '';
        data.keywords.positive.forEach(item => {
            const span = document.createElement('span');
            span.className = 'sentiment-tag';
            span.innerText = `${item.word} (${item.count})`;
            posKeywords.appendChild(span);
        });

        negKeywords.innerHTML = '';
        data.keywords.negative.forEach(item => {
            const span = document.createElement('span');
            span.className = 'sentiment-tag';
            span.innerText = `${item.word} (${item.count})`;
            negKeywords.appendChild(span);
        });
    }

    // ----------------- SINGLE REVIEW INFERENCE -----------------
    submitInferenceBtn.addEventListener('click', async () => {
        const text = reviewInput.value.trim();
        if (!text) return;

        submitInferenceBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Analyzing...`;
        submitInferenceBtn.disabled = true;
        batchResultsDiv.classList.add('hidden');

        try {
            const response = await fetch('/api/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text, model: activeModel })
            });

            if (!response.ok) {
                const err = await response.json().catch(() => ({}));
                throw new Error(err.detail || 'Inference failed');
            }

            const data = await response.json();
            displayInferenceResult(data);
        } catch (e) {
            console.error(e);
            alert(e.message || 'Error running inference. Ensure the model is trained and the server is running.');
        } finally {
            submitInferenceBtn.innerHTML = `<i class="fa-solid fa-magnifying-glass-chart"></i> Analyze Review`;
            submitInferenceBtn.disabled = false;
        }
    });

    clearInferenceBtn.addEventListener('click', () => {
        reviewInput.value = '';
        initialMsg.classList.remove('hidden');
        resultsView.classList.add('hidden');
        batchResultsDiv.classList.add('hidden');
    });

    function displayInferenceResult(data) {
        initialMsg.classList.add('hidden');
        resultsView.classList.remove('hidden');
        batchResultsDiv.classList.add('hidden');

        pillSentiment.innerText = data.sentiment;
        pillSentiment.className = `sentiment-pill ${data.sentiment === 'POSITIVE' ? 'positive' : 'negative'}`;

        const confPct = data.confidence || Math.round(data.score * 100);
        confidenceScore.innerText = `${confPct}%`;
        gaugeFill.style.width = `${confPct}%`;
        gaugeFill.style.backgroundColor = data.sentiment === 'POSITIVE' ? 'var(--text-success)' : 'var(--text-danger)';

        resultModelName.innerText = (data.model || activeModel).toUpperCase();

        // Aspects (Keyword-based aspect detection vs Transformer-based sentiment classification)
        aspectBadges.innerHTML = '';
        if (data.aspects && data.aspects.length > 0) {
            data.aspects.forEach(a => {
                const badge = document.createElement('span');
                badge.className = `aspect-badge ${a.sentiment === 'POSITIVE' ? 'aspect-pos' : 'aspect-neg'}`;
                // Labeling it clearly as keyword match
                badge.innerHTML = `<i class="fa-solid fa-tag" style="margin-right: 4px;"></i> ${a.aspect} (${a.sentiment === 'POSITIVE' ? 'Pos' : 'Neg'} keyword match)`;
                aspectBadges.appendChild(badge);
            });
        } else {
            const badge = document.createElement('span');
            badge.className = 'aspect-badge';
            badge.innerText = 'General Sentiment (No aspect keywords matched)';
            aspectBadges.appendChild(badge);
        }

        textPreview.innerText = `"${data.text}"`;
    }

    // ----------------- CSV UPLOAD -----------------
    uploadZone.addEventListener('click', () => csvFileInput.click());
    uploadZone.addEventListener('dragover', (e) => { e.preventDefault(); uploadZone.classList.add('drag-over'); });
    uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });

    csvFileInput.addEventListener('change', () => {
        if (csvFileInput.files.length > 0) {
            handleFileUpload(csvFileInput.files[0]);
        }
    });

    async function handleFileUpload(file) {
        if (!file.name.endsWith('.csv')) {
            alert('Please upload a CSV file.');
            return;
        }

        // Validate text/review column presence and emptiness
        const reader = new FileReader();
        reader.onload = async function (e) {
            const content = e.target.result;
            const lines = content.split('\n');
            if (lines.length <= 1) {
                alert('CSV validation error: File is empty or lacks rows.');
                return;
            }
            const headers = lines[0].toLowerCase().split(',');
            const hasReviewCol = headers.some(h => h.trim().includes('text') || h.trim().includes('review'));
            if (!hasReviewCol) {
                alert('CSV validation error: File must contain a "text" or "review" column.');
                return;
            }

            // Proceed with upload
            uploadStatus.classList.remove('hidden');
            uploadStatus.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Uploading and analyzing ${file.name}...`;

            const formData = new FormData();
            formData.append('file', file);

            try {
                const response = await fetch(`/api/batch-upload?model=${activeModel}`, {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    const err = await response.json().catch(() => ({}));
                    throw new Error(err.detail || 'Upload failed');
                }

                const data = await response.json();
                uploadStatus.innerHTML = `<span class="text-success"><i class="fa-solid fa-check"></i> Processed ${data.results.length} reviews</span>`;
                displayBatchResults(data);
            } catch (err) {
                console.error(err);
                uploadStatus.innerHTML = `<span class="text-danger"><i class="fa-solid fa-xmark"></i> ${err.message}</span>`;
            }
        };
        reader.readAsText(file.slice(0, 1024)); // Read beginning of file for header validation
    }

    function displayBatchResults(data) {
        initialMsg.classList.add('hidden');
        resultsView.classList.add('hidden');
        batchResultsDiv.classList.remove('hidden');

        const total = data.results.length;
        const failed = data.results.filter(r => r.sentiment === 'ERROR').length;
        const successful = total - failed;
        const s = data.summary;

        // Custom validation check if any rows failed
        let validationNote = '';
        if (failed > 0) {
            validationNote = `<div class="text-danger" style="margin-top: 10px; font-size: 13px;"><i class="fa-solid fa-triangle-exclamation"></i> Warning: ${failed} prediction(s) failed or contained malformed data. Do not skip failures.</div>`;
        }

        const batchSummary = document.getElementById('batch-summary');
        batchSummary.innerHTML = `
            <div class="batch-kpis" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(110px, 1fr)); gap: 10px;">
                <div class="batch-kpi" style="padding: 10px; background: rgba(255,255,255,0.02); text-align: center; border-radius: 6px;">
                    <span class="batch-kpi-val" style="display: block; font-size: 18px; font-weight: 600;">${total}</span>
                    <span class="batch-kpi-label" style="font-size: 11px; color: var(--text-muted);">Attempted Rows</span>
                </div>
                <div class="batch-kpi" style="padding: 10px; background: rgba(255,255,255,0.02); text-align: center; border-radius: 6px;">
                    <span class="batch-kpi-val text-success" style="display: block; font-size: 18px; font-weight: 600;">${successful}</span>
                    <span class="batch-kpi-label" style="font-size: 11px; color: var(--text-muted);">Successful</span>
                </div>
                <div class="batch-kpi" style="padding: 10px; background: rgba(255,255,255,0.02); text-align: center; border-radius: 6px;">
                    <span class="batch-kpi-val text-danger" style="display: block; font-size: 18px; font-weight: 600;">${failed}</span>
                    <span class="batch-kpi-label" style="font-size: 11px; color: var(--text-muted);">Failed</span>
                </div>
                <div class="batch-kpi" style="padding: 10px; background: rgba(255,255,255,0.02); text-align: center; border-radius: 6px;">
                    <span class="batch-kpi-val text-success" style="display: block; font-size: 17px; font-weight: 600;">${s.positive_count} (${s.positive_ratio}%)</span>
                    <span class="batch-kpi-label" style="font-size: 11px; color: var(--text-muted);">Positive</span>
                </div>
                <div class="batch-kpi" style="padding: 10px; background: rgba(255,255,255,0.02); text-align: center; border-radius: 6px;">
                    <span class="batch-kpi-val text-danger" style="display: block; font-size: 17px; font-weight: 600;">${s.negative_count} (${s.negative_ratio}%)</span>
                    <span class="batch-kpi-label" style="font-size: 11px; color: var(--text-muted);">Negative</span>
                </div>
            </div>
            ${validationNote}
        `;

        const container = document.getElementById('batch-table-container');
        if (data.results && data.results.length > 0) {
            let rows = data.results.slice(0, 20).map((r, i) => `
                <tr>
                    <td>${i + 1}</td>
                    <td>${r.text.substring(0, 80)}${r.text.length > 80 ? '...' : ''}</td>
                    <td><span class="${r.sentiment === 'POSITIVE' ? 'text-success' : (r.sentiment === 'NEGATIVE' ? 'text-danger' : 'text-warning')}">${r.sentiment}</span></td>
                    <td>${(r.score * 100).toFixed(1)}%</td>
                </tr>
            `).join('');

            container.innerHTML = `
                <table class="data-table" style="margin-top: 16px;">
                    <thead><tr><th>#</th><th>Review</th><th>Sentiment</th><th>Confidence</th></tr></thead>
                    <tbody>${rows}</tbody>
                </table>
                ${data.results.length > 20 ? `<p style="color: var(--text-muted); font-size: 12px; margin-top: 8px;">Showing first 20 of ${data.results.length} results</p>` : ''}
            `;
        }
    }

    // ----------------- AI AGENT -----------------

    // Example chip clicks
    document.querySelectorAll('.example-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            agentInput.value = chip.getAttribute('data-q');
        });
    });

    askAgentBtn.addEventListener('click', async () => {
        const question = agentInput.value.trim();
        if (!question) return;

        askAgentBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Processing...`;
        askAgentBtn.disabled = true;
        agentInitial.classList.add('hidden');
        agentResults.classList.remove('hidden');
        agentAnswerText.innerHTML = `<div class="loading-state"><i class="fa-solid fa-spinner fa-spin"></i> Agent is analyzing your question...</div>`;
        agentSupportingData.innerHTML = '';
        agentRecsList.innerHTML = '';

        try {
            const response = await fetch('/api/agent', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question: question, model: activeModel })
            });

            if (!response.ok) {
                const err = await response.json().catch(() => ({}));
                throw new Error(err.detail || 'Agent request failed');
            }

            const data = await response.json();
            displayAgentResult(data);
        } catch (e) {
            console.error(e);
            agentAnswerText.innerHTML = `<p class="text-danger">${e.message || 'Agent failed. Ensure models are trained and server is running.'}</p>`;
        } finally {
            askAgentBtn.innerHTML = `<i class="fa-solid fa-paper-plane"></i> Ask Agent`;
            askAgentBtn.disabled = false;
        }
    });

    function displayAgentResult(data) {
        const stepQ = document.getElementById('agent-step-q');
        const stepTask = document.getElementById('agent-step-task');
        const stepTool = document.getElementById('agent-step-tool');
        const stepModel = document.getElementById('agent-step-model');
        const stepAns = document.getElementById('agent-step-ans');
        const stepSupport = document.getElementById('agent-step-support');
        const stepRecs = document.getElementById('agent-step-recs');

        const userQ = document.getElementById('agent-input').value.trim();
        if (stepQ) stepQ.innerText = userQ || 'Custom Query';
        if (stepTask) stepTask.innerText = data.task || 'Answering Query';
        if (stepTool) stepTool.innerText = data.tool_used || 'General Chat';
        if (stepModel) stepModel.innerText = (data.model_used || activeModel).toUpperCase();

        const cleanAns = (data.answer || 'No answer available.').replace(/\*\*/g, '');
        if (stepAns) stepAns.innerText = cleanAns;

        if (stepSupport) {
            const sd = data.supporting_data || {};
            if (Object.keys(sd).length > 0) {
                stepSupport.innerText = JSON.stringify(sd, null, 2);
            } else {
                stepSupport.innerText = 'No structured metrics returned.';
            }
        }

        if (stepRecs) {
            const recs = data.recommendations || [];
            if (recs.length > 0) {
                stepRecs.innerHTML = recs.map(r => {
                    const cleanR = r.replace(/\*\*/g, '');
                    return `<div class="rec-item" style="margin-left: 0; border-left: 3px solid var(--primary); padding-left: 10px; margin-bottom: 6px;"><p>${cleanR}</p></div>`;
                }).join('');
            } else {
                stepRecs.innerHTML = '<span style="color: var(--text-muted)">No contextual recommendations.</span>';
            }
        }
    }

    // ----------------- MODEL METRICS -----------------
    async function loadModelMetrics() {
        try {
            metricsGrid.innerHTML = `<div class="loading-state"><i class="fa-solid fa-spinner fa-spin"></i> Loading metrics...</div>`;
            const response = await fetch('/api/metrics');
            if (!response.ok) {
                const err = await response.json().catch(() => ({}));
                throw new Error(err.detail || 'Failed to load metrics');
            }

            const data = await response.json();
            renderModelMetrics(data);
        } catch (e) {
            console.error(e);
            metricsGrid.innerHTML = `<div style="color: var(--text-danger);">${e.message}</div>`;
        }
    }

    function renderModelMetrics(data) {
        metricsGrid.innerHTML = '';

        // Show meta note
        if (data._meta) {
            if (metricsNote) metricsNote.innerText = data._meta.note || '';
        }

        const displayTitles = {
            "distilbert": "DistilBERT",
            "roberta": "RoBERTa",
            "deberta": "DeBERTa-v3"
        };

        const compTableBody = document.querySelector('#model-comparison-table tbody');
        const compMetaInfo = document.getElementById('model-meta-info-container');

        if (compTableBody) {
            compTableBody.innerHTML = '';
            const modelsList = ['distilbert', 'roberta', 'deberta'];
            modelsList.forEach(mKey => {
                const tr = document.createElement('tr');
                const mData = data[mKey];
                if (mData && mData.accuracy !== undefined) {
                    const isBest = (mKey === (data._meta && data._meta.best_model));
                    const statusBadge = isBest ? '<span class="h-status success">Active / Best Model</span>' : '<span class="h-status text-muted" style="background: rgba(255,255,255,0.05); color: var(--text-muted);">Available</span>';

                    tr.innerHTML = `
                        <td><strong style="text-transform: capitalize">${mKey}</strong></td>
                        <td>${(mData.accuracy * 100).toFixed(1)}%</td>
                        <td>${(mData.precision * 100).toFixed(1)}%</td>
                        <td>${(mData.recall * 100).toFixed(1)}%</td>
                        <td><strong>${(mData.f1 * 100).toFixed(1)}%</strong></td>
                        <td>${statusBadge}</td>
                    `;
                } else {
                    tr.innerHTML = `
                        <td><strong style="text-transform: capitalize">${mKey}</strong></td>
                        <td colspan="4" style="text-align: center; color: var(--text-danger);">Not Trained</td>
                        <td><span class="h-status danger">Offline</span></td>
                    `;
                }
                compTableBody.appendChild(tr);
            });
        }

        if (compMetaInfo) {
            const latestModelUpdateTime = data.roberta?.timestamp || data.distilbert?.timestamp || data.deberta?.timestamp || 'N/A';
            compMetaInfo.innerHTML = `
                <div style="display: flex; flex-wrap: wrap; gap: 20px; border-top: 1px solid rgba(255,255,255,0.04); padding-top: 12px; margin-top: 8px;">
                    <span><i class="fa-solid fa-circle-check" style="color: var(--text-success); margin-right: 4px;"></i> Recommended Model: <strong style="text-transform: capitalize; color: var(--primary);">${(data._meta && data._meta.best_model) || 'roberta'}</strong> (Highest F1-Score)</span>
                    <span><i class="fa-solid fa-clock" style="margin-right: 4px;"></i> Latest Evaluation: <strong>${latestModelUpdateTime}</strong></span>
                    <span><i class="fa-solid fa-database" style="margin-right: 4px;"></i> Metrics Source: <code>models/metrics.json</code></span>
                </div>
            `;
        }

        for (const [key, item] of Object.entries(data)) {
            if (key === '_meta' || key === 'best_model') continue;

            const card = document.createElement('div');
            card.className = `model-spec-card ${key === activeModel ? 'highlighted' : ''}`;

            const title = displayTitles[key] || key;
            const isBest = (key === (data._meta && data._meta.best_model));
            const isAvailable = (item.accuracy !== undefined);
            const accVal = isAvailable ? (item.accuracy * 100).toFixed(1) : '0.0';
            const f1Val = isAvailable ? (item.f1 * 100).toFixed(1) : '0.0';
            const precVal = isAvailable ? (item.precision * 100).toFixed(1) : '0.0';
            const recVal = isAvailable ? (item.recall * 100).toFixed(1) : '0.0';

            card.innerHTML = `
                <div class="model-card-header">
                    <h4>${title}</h4>
                    <div class="model-badges">
                        ${isBest ? '<span class="best-badge">🏆 Best</span>' : ''}
                        ${isAvailable ? '<span class="avail-badge available">✓ Trained</span>' : '<span class="avail-badge unavailable">✗ Not Trained</span>'}
                    </div>
                </div>
                <div class="metric-row">
                    <span class="label">Accuracy</span>
                    <span class="val text-success">${accVal}%</span>
                </div>
                <div class="metric-row">
                    <span class="label">F1-Score</span>
                    <span class="val">${f1Val}%</span>
                </div>
                <div class="metric-row">
                    <span class="label">Precision</span>
                    <span class="val">${precVal}%</span>
                </div>
                <div class="metric-row">
                    <span class="label">Recall</span>
                    <span class="val">${recVal}%</span>
                </div>
                ${item.test_samples ? `<div class="metric-row" style="margin-top: 14px; border-top: 1px solid rgba(255,255,255,0.04); padding-top: 10px;">
                    <span class="label">Test Samples</span>
                    <span class="val">${item.test_samples}</span>
                </div>` : ''}
                ${item.base_checkpoint ? `<div class="metric-row">
                    <span class="label">Base Model</span>
                    <span class="val" style="font-size: 12px;">${item.base_checkpoint}</span>
                </div>` : ''}
                ${item.timestamp ? `<div class="metric-row">
                    <span class="label">Trained At</span>
                    <span class="val" style="font-size: 11px;">${item.timestamp}</span>
                </div>` : ''}
            `;
            metricsGrid.appendChild(card);
        }
    }

    // ----------------- BUSINESS INSIGHTS TAB -----------------
    async function loadInsights() {
        if (cachedBiData) {
            renderInsights(cachedBiData);
        } else {
            try {
                const response = await fetch(`/api/bi-report?model=${activeModel}&limit=400`);
                if (!response.ok) {
                    const err = await response.json().catch(() => ({}));
                    throw new Error(err.detail || 'Failed to load');
                }
                const data = await response.json();
                cachedBiData = data;
                renderInsights(data);
            } catch (e) {
                console.error(e);
            }
        }
    }

    function renderInsights(data) {
        // KPIs
        document.getElementById('bi-total').innerText = data.summary.total_processed.toLocaleString();
        document.getElementById('bi-pos-count').innerText = data.summary.positive_count.toLocaleString();
        document.getElementById('bi-neg-count').innerText = data.summary.negative_count.toLocaleString();
        document.getElementById('bi-pos-pct').innerText = `(${data.summary.positive_ratio}%)`;
        document.getElementById('bi-neg-pct').innerText = `(${data.summary.negative_ratio}%)`;
        document.getElementById('bi-model').innerText = (data.model_used || activeModel).toUpperCase();

        // Overall satisfaction
        const biSat = document.getElementById('bi-satisfaction');
        if (biSat) {
            const satVal = data.summary.overall_satisfaction || '-';
            biSat.innerText = satVal;
            const satLower = satVal.toLowerCase();
            if (satLower === 'excellent' || satLower === 'good') {
                biSat.className = 'kpi-val text-success';
            } else if (satLower === 'fair') {
                biSat.className = 'kpi-val text-warning';
            } else {
                biSat.className = 'kpi-val text-danger';
            }
        }

        // Positive aspects (sorted by positive pct)
        const posAspects = [...data.aspect_analysis].sort((a, b) => b.positive_pct - a.positive_pct);
        const posContainer = document.getElementById('bi-positive-aspects');
        posContainer.innerHTML = '';
        posAspects.forEach(a => {
            posContainer.innerHTML += `
                <div class="insight-row">
                    <span class="insight-name">${a.aspect}</span>
                    <div class="insight-bar-container">
                        <div class="insight-bar bg-success" style="width: ${a.positive_pct}%"></div>
                    </div>
                    <span class="insight-val text-success">${a.positive_pct}%</span>
                </div>
            `;
        });

        // Negative aspects (sorted by negative pct)
        const negAspects = [...data.aspect_analysis].sort((a, b) => b.negative_pct - a.negative_pct);
        const negContainer = document.getElementById('bi-negative-aspects');
        negContainer.innerHTML = '';
        negAspects.forEach(a => {
            negContainer.innerHTML += `
                <div class="insight-row">
                    <span class="insight-name">${a.aspect}</span>
                    <div class="insight-bar-container">
                        <div class="insight-bar bg-danger" style="width: ${a.negative_pct}%"></div>
                    </div>
                    <span class="insight-val text-danger">${a.negative_pct}%</span>
                </div>
            `;
        });

        // Recommendations
        const recsContainer = document.getElementById('bi-recommendations');
        recsContainer.innerHTML = '';
        data.agent_recommendations.forEach(r => {
            const div = document.createElement('div');
            div.className = 'rec-item';
            if (r.includes('⚠️') || r.includes('💲') || r.includes('📦') || r.includes('⚙️')) {
                div.style.borderLeftColor = 'var(--text-warning)';
            } else {
                div.style.borderLeftColor = 'var(--text-success)';
            }
            div.innerHTML = `<p>${r}</p>`;
            recsContainer.appendChild(div);
        });
    }

    // Initial loads
    loadBiReport();
    loadModelMetrics();
});
