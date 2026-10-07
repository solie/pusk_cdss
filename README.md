# Puskesmas Clinical Decision Support System (CDSS)

[![License: MIT](https://img.shields.io/badge/License-MIT-teal.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-INT8%20Quantization-green.svg)](https://onnxruntime.ai/)

Non-autoregressive **System-1** clinical decision support system for rapid ICD-10 primary care diagnostic coding. Benchmarked across **34,768 clinical consultations** at Puskesmas Poncol (Semarang, Indonesia) under an out-of-city population shift.

Evaluates an ultra-compact **13.5M parameter bidirectional encoder (ELECTRA-Small)** with domain vocabulary expansion, achieving **67.78% Top-1** and **90.69% Top-3** diagnostic accuracy at **3.2 ms** (MPS GPU) / **14.1 ms** (CPU INT8) latency.

---

## 🌐 Live Web Demonstration

A live, zero-server client-side web application running the quantized 13.2 MB INT8 ONNX decision engine via WebAssembly is accessible at:

**[https://cdss.data-lab.xyz/](https://cdss.data-lab.xyz/)**

- **Dedicated Clinical Vitals**: Systolic/diastolic blood pressure, age, sex, weight, height.
- **Bilingual Interface**: Seamless toggle between Bahasa Indonesia (`ID`) and English (`EN`).
- **8 Preloaded Clinical Presets**:
  - Presets 1–5: High-frequency primary care wins (ISPA, Hypertension, ARV, Gastritis, Dental Necrosis).
  - Preset 6: Low-prevalence detection win (Acute Conjunctivitis `H10`, 0.58% prevalence, 93.0% confidence).
  - Presets 7–8: Clinically realistic dilemma and failure cases (Undifferentiated acute fever base-rate dilemma, pediatric mumps rare disease miss).
- **Physician Final Judgment Panel**: Human-in-the-loop review and official EHR confirmation.

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
│   └── dummy_pusk.csv          # Synthetic dataset (1,000 visits matching 16-col schema)
├── figures/
│   ├── figure1_patient_attrition.png   # Study flowchart & population shift partition
│   └── figure2_empirical_panels.png    # 3-panel validation (Scaling, Calibration, Epochs)
├── scripts/
│   ├── pusk_data.py              # Data cleaning, feature extraction, ICD-10 label mapping
│   ├── metrics.py                # Evaluation utilities (ECE, Brier, temperature scaling)
│   ├── baseline_tfidf_svm.py     # Classical TF-IDF + Linear SVM 5-fold pipeline
│   ├── electra_system1.py        # Vanilla PyTorch ELECTRA-Small training & evaluation
│   ├── ablation_electra.py       # Cross-modal clinical context feature ablation
│   ├── export_electra_onnx.py    # ONNX export and dynamic INT8 quantization
│   ├── modernbert_system1.py     # Native ModernBERT System-1 architecture
│   ├── gemma_system1.py          # Native Gemma 3 System-1 architecture
│   ├── generate_dummy_data.py    # Synthetic dataset generator
│   └── plot_empirical_panels.py  # Publication figures plotting script
├── clinical_tokens.json          # 100 Indonesian/Javanese clinical tokens
├── requirements.txt              # Minimal Python dependencies
└── README.md
```

---

## 🛠️ Step-by-Step Instructions: Start to Finish

### Step 1: Environment Setup
Clone the repository and install dependencies in a fresh virtual environment:
```bash
# Clone repository
git clone https://github.com/solie/pusk_cdss.git
cd pusk_cdss

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

### Step 2: Data Preparation & Synthetic Data Generation
The repository includes a synthetic clinical dataset (`data/dummy_pusk.csv`) matching the exact 16-column schema and statistical distribution of the Puskesmas EHR.

To verify or regenerate the synthetic dataset:
```bash
python scripts/generate_dummy_data.py
```
To verify the dataset parser, label coverage, and patient-level Stratified Group K-Fold split:
```bash
python scripts/pusk_data.py --csv data/dummy_pusk.csv
```

---

### Step 3: Run the Classical Linear SVM Baseline
Evaluate the 5-fold cross-validation TF-IDF + Linear SVM pipeline over patient-grouped folds:
```bash
python scripts/baseline_tfidf_svm.py --csv data/dummy_pusk.csv --out runs/baseline
```
Outputs:
- `runs/baseline/metrics.json`: Accuracy, macro-F₁, and cross-entropy log-loss.
- `runs/baseline/predictions.csv`: Out-of-fold predicted labels, scores, and probabilities.

---

### Step 4: Train the ELECTRA-Small System-1 Model
Train the native 13.5M parameter bidirectional marker-pointer decision engine. The script automatically expands the tokenizer with `clinical_tokens.json`, optimizes with AdamW and cosine decay, and evaluates calibration:
```bash
# Train from scratch (default: 4 epochs)
python scripts/electra_system1.py --epochs 4 --lr 3e-4

# Or resume training from an existing checkpoint
python scripts/electra_system1.py --resume --epochs 4 --lr 1.5e-4
```
Outputs:
- `runs/electra/best_model.pt`: Checkpoint with highest validation Top-1 accuracy.
- `runs/electra/training_history.json`: Epoch-by-epoch loss, validation accuracy, and calibration logs.
- `runs/electra/xxx_holdout_metrics.json`: Final evaluated metrics on the holdout cohort.

---

### Step 5: Run the Cross-Modal Feature Ablation Study
Evaluate the impact of physiological vitals versus clinical narratives on the holdout cohort:
```bash
python scripts/ablation_electra.py
```
Compares three input configurations:
1. **Full Clinical Encounter**: Narrative anamnesis + physiological vitals + demographics.
2. **Anamnesis Only**: Narrative text only (vitals ablated; demonstrates catastrophic logit collapse to `Z00`).
3. **Vitals Only**: Blood pressure, age, sex, and BMI only (narrative ablated).

Output: `runs/electra/ablation_results.json`.

---

### Step 6: Export to ONNX & Dynamic INT8 Quantization
Export the trained PyTorch model to ONNX format and apply dynamic INT8 post-training quantization (`QInt8`):
```bash
python scripts/export_electra_onnx.py --model-dir runs/electra --out-dir models/electra_system1
```
Outputs:
- `models/electra_system1/model.onnx`: Full FP32 ONNX computational graph.
- `models/electra_system1/model_int8.onnx`: Compressed 13.2 MB INT8 model.
- Validates numerical parity against PyTorch ($|y_{\text{PyTorch}} - y_{\text{ONNX}}| < 10^{-5}$) and benchmarks CPU latency (~14 ms).

---

### Step 7: Regenerate Publication Figures
To regenerate the 3-panel empirical validation figure:
```bash
python scripts/plot_empirical_panels.py
```
Output: `figures/figure2_empirical_panels.png`.

---

## ⚖️ Data Protection & Privacy

Due to patient confidentiality safeguards and statutory regulations governing medical records under Indonesian health data protection laws (Law No. 27/2022 on Personal Data Protection), raw de-identified patient consultation records cannot be shared publicly.

The provided `data/dummy_pusk.csv` is completely synthetic and generated using statistical sampling across the 29 target ICD-10 classes to enable full code execution and pipeline verification without compromising patient privacy.

---

## 📖 Citation

```bibtex
@article{pusk_cdss_2026,
  title={System-1 Decision Architectures for Primary Care Diagnostic Coding: Comparing Lexical Classifiers, Cross-Encoders, and Bidirectional Language Models under Population Shift},
  author={[Authors]},
  journal={arXiv preprint},
  year={2026}
}
```
