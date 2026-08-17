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

    // Top Bar Headers
    const pageTitle = document.getElementById('page-title');
    const pageSubtitle = document.getElementById('page-subtitle');

    // Overview Elements
    const kpiTotal = document.getElementById('kpi-total-reviews');
    const kpiPosPct = document.getElementById('kpi-positive-pct');
    const kpiNegPct = document.getElementById('kpi-negative-pct');
    const kpiPosFill = document.getElementById('kpi-positive-fill');
    const kpiNegFill = document.getElementById('kpi-negative-fill');
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
    const pillSentiment = document.getElementById('result-sentiment-pill');
    const confidenceScore = document.getElementById('result-score-val');
    const gaugeFill = document.getElementById('result-gauge-fill');
    const aspectBadges = document.getElementById('result-aspect-badges');
    const textPreview = document.getElementById('result-text-preview');

    // Model Performance Elements
    const metricsGrid = document.getElementById('models-metrics-cards');

    // ----------------- TAB NAVIGATION -----------------
    navButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.getAttribute('data-tab');

            // Toggle buttons
            navButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            // Toggle panels
            tabPanels.forEach(p => p.classList.remove('active'));
            document.getElementById(`tab-${targetTab}`).classList.add('active');

            activeTab = targetTab;
            updateHeaderTitles();

            // Load data when entering tab
            if (activeTab === 'overview') {
                loadBiReport();
            } else if (activeTab === 'models') {
                loadModelMetrics();
            }
        });
    });

    function updateHeaderTitles() {
        if (activeTab === 'overview') {
            pageTitle.innerText = "Sentiment & Business Intelligence Dashboard";
            pageSubtitle.innerText = "Actionable customer insights generated using state-of-the-art LLMs";
            refreshBiBtn.classList.remove('hidden');
        } else if (activeTab === 'inference') {
            pageTitle.innerText = "Interactive Review Sentiment Inference";
            pageSubtitle.innerText = "Test custom statements directly using the deployed models";
            refreshBiBtn.classList.add('hidden');
        } else if (activeTab === 'models') {
            pageTitle.innerText = "Model Evaluation & Verification Specs";
            pageSubtitle.innerText = "Explore accuracy, F1 score and performance benchmarks of fine-tuned engines";
            refreshBiBtn.classList.add('hidden');
        }
    }

    // ----------------- API: UPDATE ACTIVE MODEL -----------------
    modelSelect.addEventListener('change', (e) => {
        activeModel = e.target.value;
        if (activeTab === 'overview') {
            loadBiReport();
        }
    });

    refreshBiBtn.addEventListener('click', () => {
        loadBiReport();
    });

    // ----------------- API CALLS & RENDERING -----------------

    // Get Sentiment/BI Insights
    async function loadBiReport() {
        try {
            recsList.innerHTML = `<div class="loading-state"><i class="fa-solid fa-spinner fa-spin"></i> Generating recommendations...</div>`;
            aspectsTableBody.innerHTML = `<tr><td colspan="4" style="text-align: center; color: var(--text-muted)"><i class="fa-solid fa-spinner fa-spin"></i> Running batch analyses...</td></tr>`;

            const response = await fetch(`/api/bi-report?model=${activeModel}&limit=400`);
            if (!response.ok) throw new Error("Failed to load BI report");

            const data = await response.json();
            renderBiReport(data);
        } catch (e) {
            console.error(e);
            recsList.innerHTML = `<div class="rec-item" style="border-left-color: var(--text-danger)">
                <p>Failed to load BI report. Run dataset split script first or ensure the FastAPI backend is running.</p>
            </div>`;
        }
    }

    function renderBiReport(data) {
        // KPI values
        kpiTotal.innerText = data.summary.total_processed.toLocaleString();
        kpiPosPct.innerText = `${data.summary.positive_ratio}%`;
        kpiNegPct.innerText = `${data.summary.negative_ratio}%`;

        kpiPosFill.style.width = `${data.summary.positive_ratio}%`;
        kpiNegFill.style.width = `${data.summary.negative_ratio}%`;

        // Load Aspect Breakdown Table
        aspectsTableBody.innerHTML = '';
        data.aspect_analysis.forEach(item => {
            const tr = document.createElement('tr');

            // Determine status badge
            let statusBadge = '<span class="h-status success">Healthy</span>';
            if (item.negative_pct > 35) {
                statusBadge = '<span class="h-status warning">Needs Review</span>';
            }

            tr.innerHTML = `
                <td>
                    <strong>${item.aspect}</strong><br>
                    <span style="font-size: 11px; color: var(--text-muted)">${item.description}</span>
                </td>
                <td>${item.total_mentions}</td>
                <td>
                    <div style="display: flex; gap: 8px; font-size: 12px; font-weight: 600;">
                        <span class="text-success">${item.positive_pct}% Pos</span>
                        <span class="text-danger">${item.negative_pct}% Neg</span>
                    </div>
                </td>
                <td>${statusBadge}</td>
            `;
            aspectsTableBody.appendChild(tr);
        });

        // Aspect Sentiment Chart (Horizontal bar)
        if (aspectChart) {
            aspectChart.destroy();
        }

        const chartCanvas = document.getElementById('chart-aspects');
        if (typeof Chart === 'undefined') {
            console.warn("Chart.js CDN could not be loaded. Displaying text fallback table.");
            const parent = chartCanvas.parentNode;
            if (parent && !parent.querySelector('.chart-fallback-notice')) {
                const notice = document.createElement('div');
                notice.className = 'chart-fallback-notice';
                notice.style.color = 'var(--text-warning)';
                notice.style.fontSize = '13px';
                notice.style.textAlign = 'center';
                notice.style.padding = '20px';
                notice.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i> Chart.js is offline. Use the breakdown table below for detail scores.';
                chartCanvas.style.display = 'none';
                parent.appendChild(notice);
            }
        } else {
            const ctx = chartCanvas.getContext('2d');
            const labels = data.aspect_analysis.map(item => item.aspect);
            const posScores = data.aspect_analysis.map(item => item.positive_pct);
            const negScores = data.aspect_analysis.map(item => item.negative_pct);

            aspectChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [
                        {
                            label: 'Positive Sentiment %',
                            data: posScores,
                            backgroundColor: '#10b981',
                            borderRadius: 4
                        },
                        {
                            label: 'Negative Sentiment %',
                            data: negScores,
                            backgroundColor: '#ef4444',
                            borderRadius: 4
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    indexAxis: 'y',
                    scales: {
                        x: {
                            stacked: true,
                            max: 100,
                            grid: { color: 'rgba(255, 255, 255, 0.05)' },
                            ticks: { color: '#94a3b8' }
                        },
                        y: {
                            stacked: true,
                            grid: { display: false },
                            ticks: { color: '#94a3b8' }
                        }
                    },
                    plugins: {
                        legend: {
                            position: 'top',
                            labels: { color: '#94a3b8', font: { family: 'Inter' } }
                        }
                    }
                }
            });
        }

        // AI Recommendations List
        recsList.innerHTML = '';
        data.agent_recommendations.forEach(rec => {
            const div = document.createElement('div');
            div.className = 'rec-item';

            // Highlight warnings
            if (rec.includes('⚠️') || rec.includes('💲') || rec.includes('📦')) {
                div.style.borderLeftColor = 'var(--text-warning)';
            } else {
                div.style.borderLeftColor = 'var(--text-success)';
            }

            div.innerHTML = `<p>${rec}</p>`;
            recsList.appendChild(div);
        });

        // Word Cloud Tags
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

    // Single Review Sentiment Inference
    submitInferenceBtn.addEventListener('click', async () => {
        const text = reviewInput.value.strip ? reviewInput.value.strip() : reviewInput.value.trim();
        if (!text) return;

        submitInferenceBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Analyzing...`;
        submitInferenceBtn.disabled = true;

        try {
            const response = await fetch('/api/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text, model: activeModel })
            });

            if (!response.ok) throw new Error("Inference failed");

            const data = await response.json();
            displayInferenceResult(data);
        } catch (e) {
            console.error(e);
            alert("Error running inference. Ensure backend server is responsive.");
        } finally {
            submitInferenceBtn.innerHTML = `<i class="fa-solid fa-magnifying-glass-chart"></i> Analyze Review`;
            submitInferenceBtn.disabled = false;
        }
    });

    clearInferenceBtn.addEventListener('click', () => {
        reviewInput.value = '';
        initialMsg.classList.remove('hidden');
        resultsView.classList.add('hidden');
    });

    function displayInferenceResult(data) {
        initialMsg.classList.add('hidden');
        resultsView.classList.remove('hidden');

        // Setup badge classes
        pillSentiment.innerText = data.sentiment;
        if (data.sentiment === 'POSITIVE') {
            pillSentiment.className = 'sentiment-pill positive';
        } else {
            pillSentiment.className = 'sentiment-pill negative';
        }

        const confPct = Math.round(data.score * 100);
        confidenceScore.innerHTML = `Confidence: <strong>${confPct}%</strong>`;
        gaugeFill.style.width = `${confPct}%`;

        // Match local aspects for visual tag highlights
        aspectBadges.innerHTML = '';
        const txtLower = data.text.toLowerCase();

        let aspectsMatched = 0;
        const aspectKeywordMap = {
            "Quality": ["quality", "material", "durable", "defect", "died", "broke", "plastic"],
            "Pricing": ["price", "cost", "expensive", "cheap", "value", "money"],
            "Support": ["service", "support", "shipping", "delivery", "arrived", "return", "refund"],
            "Design": ["easy", "setup", "size", "fit", "comfortable", "install", "design"]
        };

        for (const [key, list] of Object.entries(aspectKeywordMap)) {
            if (list.some(w => txtLower.includes(w))) {
                const badge = document.createElement('span');
                badge.className = 'aspect-badge';
                badge.innerText = key;
                aspectBadges.appendChild(badge);
                aspectsMatched++;
            }
        }

        if (aspectsMatched === 0) {
            const badge = document.createElement('span');
            badge.className = 'aspect-badge';
            badge.innerText = "General Sentiment";
            aspectBadges.appendChild(badge);
        }

        // Preview text
        textPreview.innerText = `"${data.text}"`;
    }

    // Model metrics loading
    async function loadModelMetrics() {
        try {
            metricsGrid.innerHTML = `<div class="loading-state"><i class="fa-solid fa-spinner fa-spin"></i> Reading metrics metadata...</div>`;
            const response = await fetch('/api/metrics');
            if (!response.ok) throw new Error("Failed to load metrics");

            const data = await response.json();
            renderModelMetrics(data);
        } catch (e) {
            console.error(e);
            metricsGrid.innerHTML = `<div style="color: var(--text-danger);">Failed to load metrics specifications.</div>`;
        }
    }

    function renderModelMetrics(data) {
        metricsGrid.innerHTML = '';

        const displayTitles = {
            "distilbert": "DistilBERT uncased",
            "roberta": "RoBERTa base (siebert)",
            "deberta": "DeBERTa-v3 base"
        };

        for (const [key, item] of Object.entries(data)) {
            const card = document.createElement('div');
            card.className = `model-spec-card ${key === activeModel ? 'highlighted' : ''}`;

            const title = displayTitles[key] || key;
            const accVal = (item.accuracy * 100).toFixed(1);
            const f1Val = (item.f1 * 100).toFixed(1);
            const recVal = (item.recall * 100).toFixed(1);

            card.innerHTML = `
                <h4>${title}</h4>
                <div class="metric-row">
                    <span class="label">Accuracy Accuracy</span>
                    <span class="val text-success">${accVal}%</span>
                </div>
                <div class="metric-row">
                    <span class="label">F1-Score</span>
                    <span class="val">${f1Val}%</span>
                </div>
                <div class="metric-row">
                    <span class="label">Recall Rate</span>
                    <span class="val">${recVal}%</span>
                </div>
                <div class="metric-row" style="margin-top: 14px; border-top: 1px solid rgba(255,255,255,0.04); padding-top: 10px;">
                    <span class="label">Model Weights</span>
                    <span class="val">${item.parameters}</span>
                </div>
                <div class="metric-row">
                    <span class="label">CPU Inference Latency</span>
                    <span class="val" style="color: var(--text-warning);">${item.eval_time_sec}ms / batch</span>
                </div>
            `;
            metricsGrid.appendChild(card);
        }
    }

    // Initial overview load
    loadBiReport();
});
