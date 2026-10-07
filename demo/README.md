# Poncol System-1: Fully Static Client-Side Decision Support Demo

Fully static, zero-server web application for rapid ICD-10 primary care differential diagnosis.
Runs the **ELECTRA-Small 13.5M INT8 ONNX model directly inside the browser** using WebAssembly SIMD (`onnxruntime-web`).

**Zero Flask. Zero Python backend. Zero cloud dependencies.**

---

## 1. Architecture Highlights

1. **Model Backbone**:
   - `google/electra-small-discriminator` (13.5M parameters, 12 layers, $d = 256$).
   - Trained across 16 epochs on 17,163 primary care patient encounters.
   - Vocabulary expanded with 100 domain-specific Indonesian/Javanese clinical tokens.
2. **Portability & Footprint**:
   - Dynamic INT8 Quantized ONNX model: **13.2 MB** (`demo/models/electra/model_int8.onnx`).
   - Runs client-side in browser WebAssembly in **~14 ms** per encounter.
3. **Clinical Differential Triage**:
   - Classifies across **29 primary care ICD-10 diagnoses**.
   - Outputs calibrated Top-1, Top-2, and Top-3 probabilities with temperature scaling ($T = 0.90$, ECE = 0.0418).
   - Achieves **67.78% Top-1** and **90.69% Top-3** accuracy on the out-of-city population shift benchmark (`kdesa = XXX`).

---

## 2. Directory Layout

```
demo/
├── index.html                     # Fully static Single Page Application (SPA)
├── app.js                         # Client-side WordPiece tokenizer & ONNX inference runner
├── ort.min.js                     # ONNX Runtime Web client library
├── ort-wasm-simd-threaded.wasm    # WebAssembly SIMD runtime
├── presets.json                   # Clinical encounter test cases
├── run_demo.sh                    # 1-click launcher (starts static file server)
└── models/
    └── electra/
        ├── model_int8.onnx        # 13.2 MB INT8 quantized ONNX decision model
        ├── vocab.json             # WordPiece vocabulary (30,622 tokens)
        └── criteria.json          # 29 ICD-10 criteria in Indonesian & English
```

---

## 3. How to Run

### Option A: Standard Local Launcher
```bash
./demo/run_demo.sh
```
Open your browser at `http://127.0.0.1:8085`.

### Option B: Built-in Python Static Server (No Flask)
```bash
python3 -m http.server 8085 --directory demo
```

### Option C: Bun Static Server
```bash
bun x serve demo -p 8085
```

### Option D: Any Static Host / CDN
Deploy the `demo/` directory directly to GitHub Pages, Cloudflare Pages, Netlify, or an S3 bucket with zero server configuration.

---

## 4. Presets Included
- **Preset 1**: ISPA / Batuk Pilek (`J06` / `J00`) with Javanese dialect complaints (*badan gregesi*).
- **Preset 2**: Hipertensi Esensial (`I10`) with tension headaches (*cekot-cekot*).
- **Preset 3**: Kontrol ARV / VCT (`Z21`) routine checkup.
- **Preset 4**: Gastritis Akut / Maag (`K29.7`) epigastric pain.
- **Preset 5**: Nekrosis Pulpa Gigi (`K04.1`) dental pulpitis.
- **Preset 6**: Vertigo Perifer (`H81.3`) rotational dizziness (*nggliyeng*).
