# Commands

Use these commands from this folder:
C:\Users\ebita\Downloads\CUISINA TRAINING MODEL

## 1) Install dependencies

.\.venv\Scripts\python.exe -m pip install -r requirements.txt

## 2) Convert Keras model to TFLite

.\.venv\Scripts\python.exe convert_to_tflite.py --keras-model meat_classification_trained_mobilenetv5_model.keras --tflite-out meat_model.tflite

## 3) Start API server

.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000

## 4) Start frontend static server (new terminal)

.\.venv\Scripts\python.exe -m http.server 5500
python.exe -m http.server 5500

## 5) Open frontend

http://127.0.0.1:5500/frontend/

## 5.1) Frontend dataset evaluation visuals

In the frontend page:

1. Click "Select dataset folder"
2. Choose the dataset root that contains class subfolders
3. Ensure folder names match labels in labels.txt
4. Click "Run Dataset Evaluation"

The page will render:

- Confusion Matrix
- Classification Report Graph
- Prediction Probability Stats Graph

## 6) Check API health

Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:8000/health" | Select-Object -ExpandProperty Content

## 7) Optional: set labels from environment (overrides labels.txt)

$env:LABELS = "chicken,cow,donkey,goat,pig,sheep,unknown"

## 8) Stop all local sessions if needed

$ports = @(8000, 5500)
$connections = Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | Where-Object { $ports -contains $_.LocalPort }
$procIds = $connections | Select-Object -ExpandProperty OwningProcess -Unique
foreach ($procId in $procIds) { Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue }

## 9) Generate evaluation graphs (confusion matrix, classification report, probability stats)

# Replace this with your dataset folder path.

$datasetDir = "C:\path\to\your\test_dataset"

.\.venv\Scripts\python.exe evaluate_visuals.py --dataset-dir $datasetDir --model-path meat_model.tflite --labels-path labels.txt --output-dir evaluation_reports

# Optional: evaluate only first N images for a quick run

.\.venv\Scripts\python.exe evaluate_visuals.py --dataset-dir $datasetDir --max-images 200

## 10) Output files

# Generated in evaluation_reports/

# - confusion_matrix.png

# - classification_report_heatmap.png

# - prediction_probability_stats.png

# - classification_report.csv

# - prediction_probability_stats.csv

# - per_image_predictions.csv
