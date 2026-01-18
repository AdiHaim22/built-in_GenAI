@echo off
setlocal
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
    pip install transformers diffusers accelerate safetensors huggingface-hub opencv-python pillow numpy segment-anything timm
)

REM ---------- 5. Create Models directory ----------
if not exist Models (
    mkdir Models
)

REM ---------- 6. Download models from Hugging Face ----------
echo Downloading models from Hugging Face...

REM --- Generate Temporary Python Script ---
echo from huggingface_hub import snapshot_download > _temp_download.py
echo from pathlib import Path >> _temp_download.py
echo import shutil >> _temp_download.py
echo import os >> _temp_download.py
echo. >> _temp_download.py
echo BASE = Path("Models") >> _temp_download.py
echo models = { >> _temp_download.py
echo     "ChangeTypeClassifier": "AdiHaim/ChangeTypeClassifier", >> _temp_download.py
echo     "Sam_Checkpoint": "AdiHaim/Sam_Checkpoint", >> _temp_download.py
echo     "detr-finetuned-floorplans": "AdiHaim/detr-finetuned-floorplans", >> _temp_download.py
echo } >> _temp_download.py
echo. >> _temp_download.py
echo def flatten_folder(parent_path, nested_name): >> _temp_download.py
echo     nested_path = parent_path / nested_name >> _temp_download.py
echo     if nested_path.exists() and nested_path.is_dir(): >> _temp_download.py
echo         print(f"Fixing nested structure in {parent_path}...") >> _temp_download.py
echo         for item in nested_path.iterdir(): >> _temp_download.py
echo             dest = parent_path / item.name >> _temp_download.py
echo             try: >> _temp_download.py
echo                 if not dest.exists(): >> _temp_download.py
echo                     shutil.move(str(item), str(dest)) >> _temp_download.py
echo                 elif item.is_dir(): >> _temp_download.py
echo                     shutil.rmtree(item) >> _temp_download.py
echo                 else: >> _temp_download.py
echo                     item.unlink() >> _temp_download.py
echo             except Exception as e: >> _temp_download.py
echo                 print(f"  Warning: Could not move {item.name}: {e}") >> _temp_download.py
echo         try: >> _temp_download.py
echo             nested_path.rmdir() >> _temp_download.py
echo         except: pass >> _temp_download.py
echo. >> _temp_download.py
echo for name, repo in models.items(): >> _temp_download.py
echo     out_dir = BASE / name >> _temp_download.py
echo     print(f"Downloading {repo} -> {out_dir}") >> _temp_download.py
echo     snapshot_download(repo_id=repo, local_dir=out_dir, local_dir_use_symlinks=False) >> _temp_download.py
echo     # Auto-fix nested folders if they appear >> _temp_download.py
echo     flatten_folder(out_dir, name) >> _temp_download.py
echo     if name == "ChangeTypeClassifier": flatten_folder(out_dir, "Change_Type_Classifier") >> _temp_download.py
echo. >> _temp_download.py
echo print("All models downloaded successfully.") >> _temp_download.py

REM --- Run the Python Script ---
python _temp_download.py

REM --- Cleanup ---
if exist _temp_download.py del _temp_download.py

echo ================================
echo Setup complete!
echo ================================
pause