# Puskesmas Poncol Clinical Decision Support System (CDSS)

[![License: MIT](https://img.shields.io/badge/License-MIT-teal.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![ONNX Runtime Web](https://img.shields.io/badge/ONNX%20Runtime-WebAssembly%20SIMD-green.svg)](https://onnxruntime.ai/)

Non-autoregressive **System-1** decision support system for rapid ICD-10 primary care diagnostic coding. Benchmarked across **34,768 clinical encounters** at Puskesmas Poncol (Semarang, Indonesia) under an out-of-city population shift.

Evaluates an ultra-compact **13.5M parameter bidirectional encoder (ELECTRA-Small)** with domain vocabulary expansion, achieving **67.78% Top-1** and **90.69% Top-3** diagnostic accuracy at **3.2 ms** (MPS) / **14.1 ms** (CPU) latency.

---

## 🚀 1-Click Static Demo (Zero-Server / WebAssembly)

The web demonstration runs **100% client-side in the browser** using WebAssembly SIMD (`onnxruntime-web`) and a **13.2 MB quantized INT8 ONNX model**. Zero Flask or server-side Python required.

```bash
# 1. Clone repository
git clone https://github.com/solie/pusk_cdss.git
cd pusk_cdss

# 2. Launch static server
./demo/run_demo.sh
```
Open **`http://127.0.0.1:8085/`** in your browser.

- **Dedicated Vitals & Demographics**: Systolic/diastolic blood pressure, age, sex, weight, height.
- **Bilingual Interface**: Toggle between Bahasa Indonesia (`ID`) and English (`EN`).
- **8 Preloaded Clinical Presets**:
  - Presets 1–6: High-frequency & rare case detection wins (ISPA, Hypertension, ARV, Gastritis, Dental Necrosis, Conjunctivitis).
  - Presets 7–8: Clinically realistic dilemma & error cases (Undifferentiated acute fever base-rate dilemma, pediatric mumps rare disease miss).
- **Physician Final Judgment Panel**: Human-in-the-loop review and diagnostic confirmation.

---

## 📊 Benchmark Summary (Out-of-City Holdout, $n = 3,758$)

Evaluation on 3,758 unseen encounters under severe demographic and epidemiological population distribution shift (`kdesa = XXX`):

| Model Architecture | Parameters | Precision / Runtime | Top-1 | Top-2 | Top-3 | Macro-F₁ | ECE (15 bins) | Latency |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Zero-Shot TinyDecide | 13.8M | Float32 (CPU) | 1.09% | 2.45% | 32.65% | 0.0020 | 0.0635 | 117.9 ms |
| Zero-Shot Gemma 3 270M | 268M | Float32 (MPS) | 2.32% | 4.04% | 5.69% | 0.0271 | 0.0258 | 304.9 ms |
| Zero-Shot openJev | 151M | Float32 (MPS) | 5.46% | 6.47% | 8.49% | 0.0178 | 0.0173 | 26.6 ms |
| Gemma 3 270M (Extended) | 268M | Cross-encoder (+100 tok) | 49.77% | 72.12% | 81.97% | 0.3541 | **0.0300** | 363.7 ms |
| ModernBERT System-1 | 149M | Marker head (+100 tok) | 63.07% | 78.79% | 86.67% | 0.4747 | 0.0358 | 35.7 ms |
| TF-IDF + Linear SVM | 1.4 MB | Sparse n-grams + vitals | 65.38% | 80.02% | 87.39% | **0.6952** | 0.0581 | **<2.0 ms** |
| **ELECTRA-Small System-1** | **13.5M** | **Marker head (+100 tok, Peak Ep 13)** | **67.78%** | **84.25%** | **90.69%** | 0.5086 | 0.0418 | **3.2 ms** |

*Random guess baseline across 29 classes: 3.45%. On multi-threaded commodity CPU, ELECTRA-Small INT8 executes in 14.1 ms.*

---

## 📁 Repository Structure

```
pusk_cdss/
├── data/
│   └── poncol_dummy.csv          # Synthetic dataset (1,000 visits matching real schema)
├── demo/
│   ├── index.html                # Tailwind + Lucide client-side SPA
│   ├── ort.min.js                # ONNX Runtime Web library
│   ├── presets.json              # 8 clinical test cases
│   ├── run_demo.sh               # 1-click launcher
│   └── models/electra/
│       ├── model_int8.onnx       # 13.2 MB quantized ELECTRA decision model
│       ├── vocab.json            # WordPiece dictionary (30,622 tokens)
│       └── criteria.json         # 29 ICD-10 diagnostic criteria descriptions
├── figures/
│   ├── figure1_patient_attrition.png   # Study flowchart & cohort partition
│   └── figure2_empirical_panels.png    # 3-panel validation (Scaling, Calibration, Epochs)
├── scripts/
│   ├── poncol_data.py            # Data cleaning, feature extraction, ICD-10 label mapping
│   ├── electra_system1.py        # Vanilla PyTorch ELECTRA-Small training & evaluation
│   ├── ablation_electra.py       # Clinical context & physiological feature ablation
│   ├── export_electra_onnx.py    # ONNX export and dynamic INT8 quantization
│   ├── baseline_tfidf_svm.py     # Classical TF-IDF + Linear SVM pipeline
│   ├── modernbert_system1.py     # Native ModernBERT System-1 architecture
│   ├── gemma_system1.py          # Native Gemma 3 System-1 architecture
│   └── generate_dummy_data.py    # Synthetic dataset generator
├── clinical_tokens.json          # 100 Indonesian/Javanese clinical tokens
├── requirements.txt              # Minimal Python dependencies
└── README.md
```

---

## 🛠️ Python Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 1. Generating Dummy Data
To regenerate the synthetic clinical dataset matching the 16-column Puskesmas Poncol EHR schema:
```bash
python scripts/generate_dummy_data.py
```
Output: `data/dummy_poncol.csv` (1,000 synthetic patient visits).

### 2. Training ELECTRA-Small System-1
```bash
python scripts/electra_system1.py --epochs 4 --lr 3e-4
```

### 3. Running Feature Ablation Study
```bash
python scripts/ablation_electra.py
```
Evaluates Full Note vs. Anamnesis Only (no vitals) vs. Vitals Only (no text) to observe diagnostic degradation and collapse to `Z00`.

### 4. Exporting to ONNX & INT8 Quantization
```bash
python scripts/export_electra_onnx.py --model-dir runs/electra --out-dir models/electra_system1
```

---

## ⚖️ Data Protection & Privacy

Due to patient confidentiality safeguards and statutory regulations governing medical records under Indonesian health data protection laws (Law No. 27/2022 on Personal Data Protection), raw de-identified patient consultation records cannot be shared publicly.

The provided `data/poncol_dummy.csv` is completely synthetic and generated using statistical sampling across the 29 target ICD-10 classes to enable full code execution and pipeline verification without compromising patient privacy.

---

## 📖 Citation

```bibtex
@article{poncol_cdss_2026,
  title={System-1 Decision Architectures for Primary Care Diagnostic Coding: Comparing Lexical Classifiers, Cross-Encoders, and Bidirectional Language Models under Population Shift},
  author={[Authors]},
  journal={arXiv preprint},
  year={2026}
}
```
