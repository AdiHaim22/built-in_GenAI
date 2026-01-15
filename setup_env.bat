@echo off
echo ================================
echo Setting up GenAI environment
echo ================================

REM ---------- 1. Create venv ----------
if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv
)

REM ---------- 2. Activate venv ----------
call .venv\Scripts\activate

REM ---------- 3. Upgrade pip ----------
python -m pip install --upgrade pip

REM ---------- 4. Install requirements ----------
if exist requirements.txt (
    echo Installing Python requirements...
    pip install -r requirements.txt
) else (
    echo No requirements.txt found, installing core packages...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install transformers diffusers accelerate safetensors huggingface-hub opencv-python pillow numpy
)

REM ---------- 5. Create Models directory ----------
if not exist Models (
    mkdir Models
)

REM ---------- 6. Download models from Hugging Face ----------
echo Downloading models from Hugging Face...

python - <<EOF
from huggingface_hub import snapshot_download
from pathlib import Path

BASE = Path("Models")

models = {
    "ChangeTypeClassifier": "AdiHaim/ChangeTypeClassifier",
    "Sam_Checkpoint": "AdiHaim/Sam_Checkpoint",
    "detr-finetuned-floorplans": "AdiHaim/detr-finetuned-floorplans",
}

for name, repo in models.items():
    out_dir = BASE / name
    print(f"Downloading {repo} -> {out_dir}")
    snapshot_download(
        repo_id=repo,
        local_dir=out_dir,
        local_dir_use_symlinks=False
    )

print("All models downloaded successfully.")
EOF

echo ================================
echo Setup complete!
echo ================================
pause
