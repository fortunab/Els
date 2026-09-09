// Privacy-Preserving & Explainable Multi-Disease AI Framework Dashboard JS

// Paper Datasets & Quantitative Benchmarks
const DATASETS = {
    table1: [
        { domain: "Colorectal polyps", accuracy: 0.9324, precision: 0.9142, sensitivity: 0.9287, f1: 0.9214, auc: 0.9518 },
        { domain: "Cervical cytology", accuracy: 0.9106, precision: 0.8931, sensitivity: 0.9075, f1: 0.9002, auc: 0.9247 },
        { domain: "Alzheimer’s disease", accuracy: 0.8893, precision: 0.8728, sensitivity: 0.8916, f1: 0.8821, auc: 0.9035 },
        { domain: "Diabetic retinopathy", accuracy: 0.9217, precision: 0.9054, sensitivity: 0.9189, f1: 0.9121, auc: 0.9384 },
        { domain: "Skin lesions", accuracy: 0.9012, precision: 0.8876, sensitivity: 0.8993, f1: 0.8934, auc: 0.9142 }
    ],
    table2: [
        { method: "Centralized Training", accuracy: 0.9387, f1: 0.9274, auc: 0.9562, rounds: "–" },
        { method: "FedAvg", accuracy: 0.9218, f1: 0.9109, auc: 0.9386, rounds: 50 },
        { method: "FedProx", accuracy: 0.9284, f1: 0.9176, auc: 0.9449, rounds: 50 }
    ],
    table3: [
        { hosp: "Hospital A", accuracy: 0.9241, f1: 0.9136, auc: 0.9418, ece: 0.0284 },
        { hosp: "Hospital B", accuracy: 0.9187, f1: 0.9072, auc: 0.9365, ece: 0.0317 },
        { hosp: "Hospital C", accuracy: 0.9075, f1: 0.8963, auc: 0.9248, ece: 0.0362 },
        { hosp: "Hospital D", accuracy: 0.9152, f1: 0.9048, auc: 0.9327, ece: 0.0334 },
        { hosp: "Hospital E", accuracy: 0.9214, f1: 0.9107, auc: 0.9396, ece: 0.0291 }
    ],
    table4: [
        { domain: "Colorectal polyps", locAcc: 0.9318, iou: 0.8642, agreement: 0.9027 },
        { domain: "Cervical cytology", locAcc: 0.9145, iou: 0.8426, agreement: 0.8874 },
        { domain: "Alzheimer’s disease", locAcc: 0.8961, iou: 0.8175, agreement: 0.8738 },
        { domain: "Diabetic retinopathy", locAcc: 0.9273, iou: 0.8519, agreement: 0.8962 },
        { domain: "Skin lesions", locAcc: 0.9084, iou: 0.8361, agreement: 0.8829 }
    ],
    table5: [
        { domain: "Colorectal polyps", ece: 0.0217, brier: 0.0734, entropy: 0.1842 },
        { domain: "Cervical cytology", ece: 0.0276, brier: 0.0815, entropy: 0.1967 },
        { domain: "Alzheimer’s disease", ece: 0.0341, brier: 0.0928, entropy: 0.2149 },
        { domain: "Diabetic retinopathy", ece: 0.0234, brier: 0.0761, entropy: 0.1885 },
        { domain: "Skin lesions", ece: 0.0298, brier: 0.0854, entropy: 0.2013 }
    ],
    table6: [
        { arch: "Baseline CNN", accuracy: 0.9015, params: 34.72, flops: 9.84, latency: 42.16 },
        { arch: "Vision transformer", accuracy: 0.9178, params: 48.36, flops: 12.91, latency: 57.43 },
        { arch: "Hybrid CNN-ViT", accuracy: 0.9286, params: 39.14, flops: 10.27, latency: 46.82 },
        { arch: "NAS-optimized model", accuracy: 0.9342, params: 28.63, flops: 7.92, latency: 34.57 }
    ],
    table7: [
        { method: "ResNet-50 [54]", acc: 0.8927, f1: 0.8816, auc: 0.9048, ece: 0.0412, prAuc: 0.8914, froc: 0.8647, mcc: 0.7816 },
        { method: "EfficientNet-B4 [37]", acc: 0.9073, f1: 0.8951, auc: 0.9186, ece: 0.0368, prAuc: 0.9078, froc: 0.8819, mcc: 0.8127 },
        { method: "Vision transformer", acc: 0.9178, f1: 0.9064, auc: 0.9317, ece: 0.0335, prAuc: 0.9216, froc: 0.8964, mcc: 0.8368 },
        { method: "FedAvg-CNN", acc: 0.9218, f1: 0.9109, auc: 0.9386, ece: 0.0304, prAuc: 0.9297, froc: 0.9043, mcc: 0.8512 },
        { method: "Hybrid CNN-ViT", acc: 0.9286, f1: 0.9175, auc: 0.9448, ece: 0.0272, prAuc: 0.9378, froc: 0.9135, mcc: 0.8729 },
        { method: "Architecture MH", acc: 0.9342, f1: 0.9238, auc: 0.9516, ece: 0.0217, prAuc: 0.9442, froc: 0.9126, mcc: 0.8873 }
    ],
    table10: [
        { framework: "MedViT MTL [56]", arch: "Transformer Multi-Task", fed: "No", expl: "Partial", acc: 0.9018, auc: 0.9187, ece: 0.0415 },
        { framework: "Hybrid-MedNet [15]", arch: "CNN-Based Multi-Disease", fed: "No", expl: "No", acc: 0.8946, auc: 0.9102, ece: 0.0461 },
        { framework: "FedHealth [21]", arch: "Federated CNN", fed: "Yes", expl: "No", acc: 0.9127, auc: 0.9278, ece: 0.0368 },
        { framework: "MedFuse-Transformer [30]", arch: "Hybrid CNN-ViT", fed: "No", expl: "Partial", acc: 0.9185, auc: 0.9346, ece: 0.0324 },
        { framework: "Federated MedViT [28]", arch: "Federated Transformer", fed: "Yes", expl: "Partial", acc: 0.9241, auc: 0.9417, ece: 0.0283 },
        { framework: "Architecture MH", arch: "Hybrid Federated NAS", fed: "Yes", expl: "Yes", acc: 0.9342, auc: 0.9516, ece: 0.0217 }
    ],
    table11: [
        { config: "Full MH framework", acc: 0.9342, f1: 0.9238, auc: 0.9516, ece: 0.0217 },
        { config: "W/o federated learning", acc: 0.9181, f1: 0.9074, auc: 0.9365, ece: 0.0326 },
        { config: "W/o preprocessing", acc: 0.9097, f1: 0.8983, auc: 0.9271, ece: 0.0369 },
        { config: "W/o NAS optimization", acc: 0.9214, f1: 0.9112, auc: 0.9398, ece: 0.0297 },
        { config: "W/o explainability module", acc: 0.9258, f1: 0.9149, auc: 0.9431, ece: 0.0284 },
        { config: "W/o calibration mechanism", acc: 0.9286, f1: 0.9172, auc: 0.9453, ece: 0.0418 }
    ],
    fig1: {
        categories: ["Alzheimer", "Respiratory", "Pneumonia", "Dementia+MCI", "Brain tumor", "Cardiovascular", "Eye&Glaucoma", "Skin cancer", "Retinopathy", "Parkinson", "Breast cancer"],
        methods: ["CNN", "SVM", "ResNet", "Attn", "MLP", "RF", "Tr/ViT", "DenseNet", "DT", "KNN", "LSTM", "AlexNet"],
        matrix: [
            [61, 52, 51, 46, 38, 34, 49, 30, 26, 17, 16],
            [29, 16, 9, 33, 25, 28, 14, 10, 8, 16, 8],
            [17, 24, 28, 16, 11, 13, 21, 13, 11, 1, 6],
            [21, 17, 11, 16, 15, 7, 14, 10, 7, 4, 4],
            [16, 8, 11, 12, 10, 15, 9, 6, 6, 5, 7],
            [14, 4, 1, 11, 11, 16, 5, 2, 3, 4, 1],
            [6, 6, 7, 4, 8, 1, 9, 10, 5, 3, 4],
            [7, 8, 9, 6, 7, 5, 11, 9, 6, 0, 3],
            [6, 3, 1, 6, 5, 11, 1, 1, 0, 3, 3],
            [8, 4, 1, 5, 4, 8, 2, 0, 2, 5, 3],
            [11, 3, 0, 8, 4, 4, 1, 2, 1, 8, 1],
            [2, 1, 3, 4, 4, 3, 1, 5, 0, 0, 1]
        ]
    }
};

let fig5Chart = null;
let trainLiveChart = null;
let sotaRadarChart = null;
let isTrainingActive = false;

document.addEventListener("DOMContentLoaded", () => {
    populateTables();
    initFig1Chart();
    initFig5Chart();
    initAblationChart();
    initTrainLiveChart();
    initSotaRadarChart();
    renderMRIMatrix();
});

function switchTab(tabId) {
    document.querySelectorAll(".tab-content").forEach(el => el.classList.remove("active"));
    document.querySelectorAll(".nav-btn").forEach(el => el.classList.remove("active"));
    
    document.getElementById(`view-${tabId}`).classList.add("active");
    document.getElementById(`tab-${tabId}`).classList.add("active");
}

function switchSotaSubTab(subId) {
    document.querySelectorAll(".subtab-content").forEach(el => el.classList.remove("active"));
    document.querySelectorAll(".sub-tab-btn").forEach(el => el.classList.remove("active"));
    
    document.getElementById(`subtab-${subId}`).classList.add("active");
    event.target.classList.add("active");
}

function populateTables() {
    // Table 1
    const t1Body = document.querySelector("#table1-dom tbody");
    if (t1Body) {
        t1Body.innerHTML = DATASETS.table1.map(r => `
            <tr>
                <td><strong>${r.domain}</strong></td>
                <td>${r.accuracy.toFixed(4)}</td>
                <td>${r.precision.toFixed(4)}</td>
                <td>${r.sensitivity.toFixed(4)}</td>
                <td>${r.f1.toFixed(4)}</td>
                <td><strong>${r.auc.toFixed(4)}</strong></td>
            </tr>
        `).join("");
    }

    // Table 2
    const t2Body = document.querySelector("#table2-dom tbody");
    if (t2Body) {
        t2Body.innerHTML = DATASETS.table2.map(r => `
            <tr class="${r.method === 'FedProx' ? 'highlight-row' : ''}">
                <td><strong>${r.method}</strong></td>
                <td>${r.accuracy.toFixed(4)}</td>
                <td>${r.f1.toFixed(4)}</td>
                <td>${r.auc.toFixed(4)}</td>
                <td>${r.rounds}</td>
            </tr>
        `).join("");
    }

    // Table 3
    const t3Body = document.querySelector("#table3-dom tbody");
    if (t3Body) {
        t3Body.innerHTML = DATASETS.table3.map(r => `
            <tr>
                <td><strong>${r.hosp}</strong></td>
                <td>${r.accuracy.toFixed(4)}</td>
                <td>${r.f1.toFixed(4)}</td>
                <td>${r.auc.toFixed(4)}</td>
                <td>${r.ece.toFixed(4)}</td>
            </tr>
        `).join("");
    }

    // Table 4
    const t4Body = document.querySelector("#table4-dom tbody");
    if (t4Body) {
        t4Body.innerHTML = DATASETS.table4.map(r => `
            <tr>
                <td><strong>${r.domain}</strong></td>
                <td>${r.locAcc.toFixed(4)}</td>
                <td>${r.iou.toFixed(4)}</td>
                <td><strong>${r.agreement.toFixed(4)}</strong></td>
            </tr>
        `).join("");
    }

    // Table 5
    const t5Body = document.querySelector("#table5-dom tbody");
    if (t5Body) {
        t5Body.innerHTML = DATASETS.table5.map(r => `
            <tr>
                <td><strong>${r.domain}</strong></td>
                <td>${r.ece.toFixed(4)}</td>
                <td>${r.brier.toFixed(4)}</td>
                <td>${r.entropy.toFixed(4)}</td>
            </tr>
        `).join("");
    }

    // Table 6 Copy
    const t6Copy = document.querySelector("#table6-dom-copy tbody");
    if (t6Copy) {
        t6Copy.innerHTML = DATASETS.table6.map(r => `
            <tr class="${r.arch.includes('NAS') ? 'highlight-row' : ''}">
                <td><strong>${r.arch}</strong></td>
                <td>${r.accuracy.toFixed(4)}</td>
                <td>${r.params.toFixed(2)}M</td>
                <td>${r.flops.toFixed(2)}G</td>
                <td><strong>${r.latency.toFixed(2)} ms</strong></td>
            </tr>
        `).join("");
    }

    // Table 7
    const t7Body = document.querySelector("#table7-dom tbody");
    if (t7Body) {
        t7Body.innerHTML = DATASETS.table7.map(r => `
            <tr class="${r.method.includes('MH') ? 'highlight-row' : ''}">
                <td><strong>${r.method}</strong></td>
                <td>${r.acc.toFixed(4)}</td>
                <td>${r.f1.toFixed(4)}</td>
                <td>${r.auc.toFixed(4)}</td>
                <td>${r.ece.toFixed(4)}</td>
                <td>${r.prAuc.toFixed(4)}</td>
                <td>${r.froc.toFixed(4)}</td>
                <td>${r.mcc.toFixed(4)}</td>
            </tr>
        `).join("");
    }

    // Table 10
    const t10Body = document.querySelector("#table10-dom tbody");
    if (t10Body) {
        t10Body.innerHTML = DATASETS.table10.map(r => `
            <tr class="${r.framework.includes('MH') ? 'highlight-row' : ''}">
                <td><strong>${r.framework}</strong></td>
                <td>${r.arch}</td>
                <td>${r.fed}</td>
                <td>${r.expl}</td>
                <td>${r.acc.toFixed(4)}</td>
                <td>${r.auc.toFixed(4)}</td>
                <td>${r.ece.toFixed(4)}</td>
            </tr>
        `).join("");
    }

    // Table 11
    const t11Body = document.querySelector("#table11-dom tbody");
    if (t11Body) {
        t11Body.innerHTML = DATASETS.table11.map(r => `
            <tr class="${r.config.includes('Full') ? 'highlight-row' : ''}">
                <td><strong>${r.config}</strong></td>
                <td>${r.acc.toFixed(4)}</td>
                <td>${r.f1.toFixed(4)}</td>
                <td>${r.auc.toFixed(4)}</td>
                <td>${r.ece.toFixed(4)}</td>
            </tr>
        `).join("");
    }
}

function initFig1Chart() {
    const el = document.getElementById("fig1HeatmapCanvas");
    if (!el) return;
    const ctx = el.getContext("2d");
    const topMethods = ["CNN", "SVM", "ResNet", "Attn", "MLP", "RF"];
    const datasets = topMethods.map((method, idx) => {
        const rowIdx = DATASETS.fig1.methods.indexOf(method);
        const colors = ["#3b82f6", "#8b5cf6", "#06b6d4", "#10b981", "#f59e0b", "#ef4444"];
        return {
            label: method,
            data: DATASETS.fig1.matrix[rowIdx],
            backgroundColor: colors[idx]
        };
    });

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: DATASETS.fig1.categories,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: '#9ca3af', font: { size: 10 } } }
            },
            scales: {
                x: { ticks: { color: '#9ca3af', font: { size: 9 } } },
                y: { ticks: { color: '#9ca3af' }, title: { display: true, text: 'Publication Count', color: '#9ca3af' } }
            }
        }
    });
}

function initFig5Chart() {
    const el = document.getElementById("fig5ConvergenceCanvas");
    if (!el) return;
    const ctx = el.getContext("2d");
    const rounds = Array.from({ length: 51 }, (_, i) => i);
    
    const fedAvgData = rounds.map(r => 0.76 + (0.9218 - 0.76) * (1 - Math.exp(-0.075 * r)));
    const fedProxData = rounds.map(r => 0.77 + (0.9284 - 0.77) * (1 - Math.exp(-0.082 * r)));
    const centData = rounds.map(r => 0.77 + (0.9387 - 0.77) * (1 - Math.exp(-0.090 * r)));

    fig5Chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: rounds,
            datasets: [
                { label: 'FedAvg', data: fedAvgData, borderColor: '#3b82f6', backgroundColor: 'rgba(59, 130, 246, 0.1)', tension: 0.3, pointRadius: 2 },
                { label: 'FedProx', data: fedProxData, borderColor: '#ef4444', backgroundColor: 'rgba(239, 68, 68, 0.1)', tension: 0.3, pointRadius: 2 },
                { label: 'Centralized', data: centData, borderColor: '#8b5cf6', backgroundColor: 'rgba(139, 92, 246, 0.1)', tension: 0.3, pointRadius: 2 }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: '#f3f4f6' } }
            },
            scales: {
                x: { title: { display: true, text: 'Communication Rounds', color: '#9ca3af' }, ticks: { color: '#9ca3af' } },
                y: { min: 0.7, max: 0.96, title: { display: true, text: 'Validation Accuracy', color: '#9ca3af' }, ticks: { color: '#9ca3af' } }
            }
        }
    });
}

function initTrainLiveChart() {
    const el = document.getElementById("trainLiveChartCanvas");
    if (!el) return;
    const ctx = el.getContext("2d");
    trainLiveChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                { label: 'Training Loss', data: [], borderColor: '#ef4444', backgroundColor: 'rgba(239, 68, 68, 0.1)', yAxisID: 'yLoss', tension: 0.3 },
                { label: 'Val Accuracy', data: [], borderColor: '#34d399', backgroundColor: 'rgba(52, 211, 153, 0.1)', yAxisID: 'yAcc', tension: 0.3 }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { title: { display: true, text: 'Communication Round', color: '#9ca3af' }, ticks: { color: '#9ca3af' } },
                yLoss: { type: 'linear', position: 'left', min: 0.0, max: 1.0, title: { display: true, text: 'Loss', color: '#ef4444' }, ticks: { color: '#ef4444' } },
                yAcc: { type: 'linear', position: 'right', min: 0.6, max: 1.0, title: { display: true, text: 'Accuracy', color: '#34d399' }, ticks: { color: '#34d399' } }
            }
        }
    });
}

function initSotaRadarChart() {
    const el = document.getElementById("sotaRadarCanvas");
    if (!el) return;
    const ctx = el.getContext("2d");

    sotaRadarChart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['Accuracy', 'F1-Score', 'AUC', 'PR-AUC', 'FROC', 'MCC'],
            datasets: [
                {
                    label: 'Architecture MH (Proposed)',
                    data: [0.9342, 0.9238, 0.9516, 0.9442, 0.9126, 0.8873],
                    borderColor: '#34d399',
                    backgroundColor: 'rgba(52, 211, 153, 0.25)',
                    borderWidth: 2
                },
                {
                    label: 'ResNet-50 Baseline',
                    data: [0.8927, 0.8816, 0.9048, 0.8914, 0.8647, 0.7816],
                    borderColor: '#ef4444',
                    backgroundColor: 'rgba(239, 68, 68, 0.15)',
                    borderWidth: 1.5
                },
                {
                    label: 'Vision Transformer (ViT)',
                    data: [0.9178, 0.9064, 0.9317, 0.9216, 0.8964, 0.8368],
                    borderColor: '#c084fc',
                    backgroundColor: 'rgba(192, 132, 252, 0.15)',
                    borderWidth: 1.5
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    pointLabels: { color: '#9ca3af', font: { size: 10 } },
                    min: 0.7,
                    max: 1.0,
                    ticks: { color: '#9ca3af', backdropColor: 'transparent' }
                }
            },
            plugins: { legend: { labels: { color: '#f3f4f6' } } }
        }
    });
}

function initAblationChart() {
    const el = document.getElementById("ablationBarCanvas");
    if (!el) return;
    const ctx = el.getContext("2d");
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: DATASETS.table11.map(r => r.config.replace('W/o ', '-')),
            datasets: [
                { label: 'Accuracy', data: DATASETS.table11.map(r => r.acc), backgroundColor: '#34d399' },
                { label: 'AUC', data: DATASETS.table11.map(r => r.auc), backgroundColor: '#60a5fa' }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { labels: { color: '#9ca3af' } } },
            scales: { x: { ticks: { color: '#9ca3af', font: { size: 9 } } }, y: { min: 0.88, max: 0.96, ticks: { color: '#9ca3af' } } }
        }
    });
}

function renderMRIMatrix() {
    const grid = document.getElementById("mri-matrix-grid");
    if (!grid) return;
    const stages = ["Non demented", "Moderate", "Very mild", "Mild"];
    const columns = ["Original MRI", "Grad-CAM", "Grad-CAM++", "Score-CAM"];

    let html = "";
    stages.forEach(stage => {
        columns.forEach(col => {
            const canvasId = `mri-c-${stage.replace(/\s+/g, '')}-${col.replace(/[\s\-\+]+/g, '')}`;
            html += `
                <div class="mri-cell">
                    <canvas id="${canvasId}" width="80" height="80"></canvas>
                    <span>${stage} - ${col}</span>
                </div>
            `;
        });
    });
    grid.innerHTML = html;

    setTimeout(() => {
        stages.forEach(stage => {
            columns.forEach(col => {
                const canvasId = `mri-c-${stage.replace(/\s+/g, '')}-${col.replace(/[\s\-\+]+/g, '')}`;
                const canvas = document.getElementById(canvasId);
                if (!canvas) return;
                const ctx = canvas.getContext("2d");
                
                ctx.fillStyle = "#050814";
                ctx.fillRect(0, 0, 80, 80);
                
                ctx.beginPath();
                ctx.ellipse(40, 40, 30, 35, 0, 0, 2 * Math.PI);
                ctx.fillStyle = "#222";
                ctx.fill();
                
                ctx.beginPath();
                ctx.ellipse(33, 40, 5, 12, 0, 0, 2 * Math.PI);
                ctx.ellipse(47, 40, 5, 12, 0, 0, 2 * Math.PI);
                ctx.fillStyle = "#111";
                ctx.fill();

                if (col !== "Original MRI") {
                    const grad = ctx.createRadialGradient(40, 40, 2, 40, 40, 28);
                    if (col === "Grad-CAM") {
                        grad.addColorStop(0, "rgba(239, 68, 68, 0.85)");
                        grad.addColorStop(0.5, "rgba(245, 158, 11, 0.6)");
                        grad.addColorStop(1, "rgba(59, 130, 246, 0.0)");
                    } else if (col === "Grad-CAM++") {
                        grad.addColorStop(0, "rgba(236, 72, 153, 0.9)");
                        grad.addColorStop(0.4, "rgba(168, 85, 247, 0.7)");
                        grad.addColorStop(1, "rgba(14, 165, 233, 0.0)");
                    } else {
                        grad.addColorStop(0, "rgba(16, 185, 129, 0.85)");
                        grad.addColorStop(0.5, "rgba(6, 182, 212, 0.5)");
                        grad.addColorStop(1, "rgba(99, 102, 241, 0.0)");
                    }
                    ctx.beginPath();
                    ctx.ellipse(40, 40, 26, 30, 0, 0, 2 * Math.PI);
                    ctx.fillStyle = grad;
                    ctx.fill();
                }
            });
        });
    }, 100);
}

function startInteractiveTraining() {
    if (isTrainingActive) return;
    isTrainingActive = true;

    const modelName = document.getElementById("train-model-select").value;
    const domainName = document.getElementById("train-domain-select").value;
    const totalRounds = parseInt(document.getElementById("train-rounds-slider").value);
    const epochsPerRound = parseInt(document.getElementById("train-epochs-slider").value);
    
    const badge = document.getElementById("training-status-badge");
    const btn = document.getElementById("btn-start-training");
    const consoleLog = document.getElementById("training-console-log");
    const progressFill = document.getElementById("training-progress-fill");

    badge.innerText = "Training...";
    badge.className = "badge badge-warning";
    btn.disabled = true;

    // Target accuracies based on model type
    const modelTargets = {
        "Architecture MH": 0.9342,
        "ResNet-50": 0.8927,
        "EfficientNet-B4": 0.9073,
        "Vision transformer": 0.9178,
        "FedAvg-CNN": 0.9218,
        "Hybrid CNN-ViT": 0.9286
    };
    const targetAcc = modelTargets[modelName] || 0.9200;

    trainLiveChart.data.labels = [];
    trainLiveChart.data.datasets[0].data = [];
    trainLiveChart.data.datasets[1].data = [];
    trainLiveChart.update();

    consoleLog.innerHTML = `<div class="log-entry active">> Initiating training session for [${modelName}] on domain [${domainName}]...</div>`;

    let currentRound = 1;
    let currAcc = 0.68;
    let currLoss = 0.88;

    const interval = setInterval(() => {
        if (currentRound > totalRounds) {
            clearInterval(interval);
            isTrainingActive = false;
            badge.innerText = "Completed";
            badge.className = "badge badge-success";
            btn.disabled = false;
            progressFill.style.width = "100%";
            consoleLog.innerHTML += `<div class="log-entry success">✔ Training finished! Final Val Accuracy: ${currAcc.toFixed(4)} | Loss: ${currLoss.toFixed(4)}</div>`;
            return;
        }

        currLoss = currLoss * 0.84 + Math.random() * 0.02;
        currAcc = currAcc + (targetAcc - currAcc) * (1 - Math.exp(-0.25 * currentRound));
        currAcc = Math.min(currAcc, targetAcc);

        trainLiveChart.data.labels.push(`R${currentRound}`);
        trainLiveChart.data.datasets[0].data.push(currLoss.toFixed(4));
        trainLiveChart.data.datasets[1].data.push(currAcc.toFixed(4));
        trainLiveChart.update();

        const pct = Math.round((currentRound / totalRounds) * 100);
        progressFill.style.width = `${pct}%`;

        consoleLog.innerHTML += `<div class="log-entry">> Round ${currentRound}/${totalRounds} (${epochsPerRound} local epochs) | Loss: ${currLoss.toFixed(4)} | Val Acc: ${currAcc.toFixed(4)}</div>`;
        consoleLog.scrollTop = consoleLog.scrollHeight;

        currentRound++;
    }, 400);
}

function updateFLRoundSlider(val) {
    document.getElementById("round-val-lbl").innerText = val;
}

function runFLSimulation() {
    const rounds = parseInt(document.getElementById("fl-round-slider").value);
    const strategy = document.getElementById("fl-strategy-select").value;
    
    if (fig5Chart) {
        fig5Chart.options.plugins.title = {
            display: true,
            text: `FL Active Strategy: ${strategy} (${rounds} Rounds Executed)`,
            color: '#34d399'
        };
        fig5Chart.update();
    }
}

function downloadJSONResults() {
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(DATASETS, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute("href", dataStr);
    downloadAnchor.setAttribute("download", "multidisease_ai_paper_results.json");
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
}
