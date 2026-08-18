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

        // Color satisfaction
        const sat = (data.summary.overall_satisfaction || '').toLowerCase();
        if (sat === 'excellent' || sat === 'good') {
            kpiSatisfaction.className = 'kpi-val text-success';
        } else if (sat === 'fair') {
            kpiSatisfaction.className = 'kpi-val text-warning';
        } else {
            kpiSatisfaction.className = 'kpi-val text-danger';
        }

        // Aspect table
        aspectsTableBody.innerHTML = '';
        data.aspect_analysis.forEach(item => {
            const tr = document.createElement('tr');
            let statusBadge = '<span class="h-status success">Healthy</span>';
            if (item.negative_pct > 50) {
                statusBadge = '<span class="h-status danger">Critical</span>';
            } else if (item.negative_pct > 35) {
                statusBadge = '<span class="h-status warning">Needs Review</span>';
            }

            tr.innerHTML = `
                <td>
                    <strong>${item.aspect}</strong><br>
                    <span style="font-size: 11px; color: var(--text-muted)">${item.description}</span>
                </td>
                <td>${item.total_mentions}</td>
                <td><span class="text-success">${item.positive_pct}%</span></td>
                <td><span class="text-danger">${item.negative_pct}%</span></td>
                <td>${statusBadge}</td>
            `;
            aspectsTableBody.appendChild(tr);
        });

        // Aspect Chart
        if (aspectChart) aspectChart.destroy();
        const chartCanvas = document.getElementById('chart-aspects');
        if (typeof Chart !== 'undefined') {
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

        // AI Recommendations
        recsList.innerHTML = '';
        data.agent_recommendations.forEach(rec => {
            const div = document.createElement('div');
            div.className = 'rec-item';
            if (rec.includes('⚠️') || rec.includes('💲') || rec.includes('📦') || rec.includes('⚙️')) {
                div.style.borderLeftColor = 'var(--text-warning)';
            } else {
                div.style.borderLeftColor = 'var(--text-success)';
            }
            div.innerHTML = `<p>${rec}</p>`;
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

        // Aspects from server
        aspectBadges.innerHTML = '';
        if (data.aspects && data.aspects.length > 0) {
            data.aspects.forEach(a => {
                const badge = document.createElement('span');
                badge.className = `aspect-badge ${a.sentiment === 'POSITIVE' ? 'aspect-pos' : 'aspect-neg'}`;
                badge.innerText = `${a.aspect} → ${a.sentiment}`;
                aspectBadges.appendChild(badge);
            });
        } else {
            const badge = document.createElement('span');
            badge.className = 'aspect-badge';
            badge.innerText = 'General Sentiment';
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
            uploadStatus.innerHTML = `<span class="text-success"><i class="fa-solid fa-check"></i> Analyzed ${data.summary.total_processed} reviews</span>`;
            displayBatchResults(data);
        } catch (e) {
            console.error(e);
            uploadStatus.innerHTML = `<span class="text-danger"><i class="fa-solid fa-xmark"></i> ${e.message}</span>`;
        }
    }

    function displayBatchResults(data) {
        initialMsg.classList.add('hidden');
        resultsView.classList.add('hidden');
        batchResultsDiv.classList.remove('hidden');

        const s = data.summary;
        const batchSummary = document.getElementById('batch-summary');
        batchSummary.innerHTML = `
            <div class="batch-kpis">
                <div class="batch-kpi"><span class="batch-kpi-val">${s.total_processed}</span><span class="batch-kpi-label">Total</span></div>
                <div class="batch-kpi"><span class="batch-kpi-val text-success">${s.positive_count}</span><span class="batch-kpi-label">Positive (${s.positive_ratio}%)</span></div>
                <div class="batch-kpi"><span class="batch-kpi-val text-danger">${s.negative_count}</span><span class="batch-kpi-label">Negative (${s.negative_ratio}%)</span></div>
            </div>
        `;

        const container = document.getElementById('batch-table-container');
        if (data.results && data.results.length > 0) {
            let rows = data.results.slice(0, 20).map((r, i) => `
                <tr>
                    <td>${i + 1}</td>
                    <td>${r.text.substring(0, 80)}${r.text.length > 80 ? '...' : ''}</td>
                    <td><span class="${r.sentiment === 'POSITIVE' ? 'text-success' : 'text-danger'}">${r.sentiment}</span></td>
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
        agentToolName.innerText = `🔧 ${data.task || 'Analysis'}`;
        agentModelUsed.innerText = `Model: ${(data.model_used || activeModel).toUpperCase()}`;

        // Format answer with line breaks
        const answerHtml = (data.answer || 'No answer available.').replace(/\n/g, '<br>');
        agentAnswerText.innerHTML = `<div class="agent-answer-content">${answerHtml}</div>`;

        // Supporting data
        const sd = data.supporting_data || {};
        if (Object.keys(sd).length > 0) {
            let items = '';
            for (const [key, val] of Object.entries(sd)) {
                if (typeof val === 'object') continue; // Skip nested objects
                items += `<div class="support-item"><span class="support-label">${key.replace(/_/g, ' ')}</span><span class="support-val">${val}</span></div>`;
            }
            if (items) {
                agentSupportingData.innerHTML = `<h4>Supporting Data</h4><div class="support-grid">${items}</div>`;
            }
        }

        // Recommendations
        const recs = data.recommendations || [];
        if (recs.length > 0) {
            agentRecsList.innerHTML = '<h4>Recommended Actions</h4>';
            recs.forEach(r => {
                const div = document.createElement('div');
                div.className = 'rec-item';
                div.innerHTML = `<p>${r}</p>`;
                agentRecsList.appendChild(div);
            });
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
            metricsNote.innerText = data._meta.note || '';
        }

        const displayTitles = {
            "distilbert": "DistilBERT",
            "roberta": "RoBERTa",
            "deberta": "DeBERTa-v3"
        };

        for (const [key, item] of Object.entries(data)) {
            if (key === '_meta') continue;

            const card = document.createElement('div');
            card.className = `model-spec-card ${key === activeModel ? 'highlighted' : ''}`;

            const title = displayTitles[key] || key;
            const isBest = item.is_best;
            const isAvailable = item.available;
            const accVal = (item.accuracy * 100).toFixed(1);
            const f1Val = (item.f1 * 100).toFixed(1);
            const precVal = (item.precision * 100).toFixed(1);
            const recVal = (item.recall * 100).toFixed(1);

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
                ${item.trained_at ? `<div class="metric-row">
                    <span class="label">Trained At</span>
                    <span class="val" style="font-size: 11px;">${new Date(item.trained_at).toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' })}</span>
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
