# Meat Classification TFLite API

This folder contains:

- `convert_to_tflite.py` — converts the provided `.keras` model into a `.tflite` model
- `app.py` — FastAPI server that runs inference using the TFLite Interpreter

## 1) Install (runtime)

```powershell
# from this folder
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If you plan to convert `.keras` to `.tflite`, install the conversion dependencies too:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-convert.txt
```

## 2) Convert `.keras` → `.tflite`

```powershell
.\.venv\Scripts\python.exe convert_to_tflite.py --keras-model meat_classification_trained_mobilenetv5_model.keras --tflite-out meat_model.tflite
```

Optional float16 conversion:

```powershell
.\.venv\Scripts\python.exe convert_to_tflite.py --float16 --tflite-out meat_model_float16.tflite
```

For memory-limited deployments, the float16 model is strongly recommended.

## 3) Run the API

```powershell
$env:TFLITE_MODEL_PATH = "meat_model.tflite"
.\.venv\Scripts\python.exe -m uvicorn app:app --host 0.0.0.0 --port 8000
```

If you have `meat_model_float16.tflite` or `meat_model_int8.tflite` present,
the API will automatically prefer the smaller model unless you set
`TFLITE_MODEL_PATH` explicitly.

Health check:

- http://localhost:8000/health

## 4) Call `/predict`

```powershell
curl -X POST "http://localhost:8000/predict?top_k=5" -F "file=@path\to\image.jpg"
```

## Optional: labels

If you know your 7 class names, you can set them as a comma-separated list:

```powershell
$env:LABELS = "class0,class1,class2,class3,class4,class5,class6"
```

Or edit `labels.txt` (one class per line). `labels.txt` is loaded automatically
when `LABELS` is not set.

```text
beef
chicken
lamb
pork
fish
goat
other
```

After updating labels, restart the API server.

## 5) Frontend sample (HTML/CSS/JS)

A ready-made frontend client is available in `frontend/`:

- `frontend/index.html`
- `frontend/styles.css`
- `frontend/app.js`

Run a static server from the project root:

```powershell
.\.venv\Scripts\python.exe -m http.server 5500
```

Open in browser:

- http://127.0.0.1:5500/frontend/

Then in the UI:

1. Set API Base URL (default: `http://127.0.0.1:8000`)
2. Click **Check API**
3. Upload an image
4. Click **Run Prediction**
