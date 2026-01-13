# Activate the virtual environment
.venv\Scripts\Activate.ps1

# Step 1: Run the training script (ensure the file exists)
if (-not (Test-Path "src\train.py")) {
    Write-Host "File train.py is missing!" -ForegroundColor Red
    exit
}
Write-Host "Running training script..." -ForegroundColor Green
python src\train.py

# Step 2: Run the file peek_columns.py (ensure the file exists)
if (-not (Test-Path "src\peek_columns.py")) {
    Write-Host "File peek_columns.py is missing!" -ForegroundColor Red
    exit
}
Write-Host "Running peek_columns.py to inspect dataset columns..." -ForegroundColor Green
python src\peek_columns.py --input data/raw/CICIDS2017.csv

# Step 3: Run the file prepare_dataset.py (ensure the file exists)
if (-not (Test-Path "src\prepare_dataset.py")) {
    Write-Host "File prepare_dataset.py is missing!" -ForegroundColor Red
    exit
}
Write-Host "Preparing dataset using prepare_dataset.py..." -ForegroundColor Green
python src\prepare_dataset.py --rows 100000 --dtype CICIDS

# Step 4: Run experiments (rf, svm, xgb)
if (-not (Test-Path "src\train_experiment.py")) {
    Write-Host "File train_experiment.py is missing!" -ForegroundColor Red
    exit
}
Write-Host "Running experiments with RF, SVM, and XGB..." -ForegroundColor Green
python src\train_experiment.py --model rf
python src\train_experiment.py --model svm
python src\train_experiment.py --model xgb

# Step 5: Compare models and generate reports
if (-not (Test-Path "src\compare.py")) {
    Write-Host "File compare.py is missing!" -ForegroundColor Red
    exit
}
Write-Host "Comparing models and generating reports..." -ForegroundColor Green
python src\compare.py

# Step 6: Launch Streamlit UI
# --- Step 6: Launch Streamlit UI (SAFE WAY) ---

$PROJECT_ROOT = Get-Location
$STREAMLIT_APP = Join-Path $PROJECT_ROOT "app\app.py"

Write-Host "Launching Streamlit UI..." -ForegroundColor Green
Write-Host "App path: $STREAMLIT_APP" -ForegroundColor Yellow

python -m streamlit run "$STREAMLIT_APP"
