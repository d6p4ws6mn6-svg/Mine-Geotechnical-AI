"""Run this on whichever machine is failing to load model_artifacts/, using
the exact same Python/venv that runs `streamlit run mine_app_streamlit.py`.
It prints installed versions and the full traceback for each artifact file,
instead of the app's truncated error message.
"""
import os
import sys
import traceback

print("python:", sys.executable)
print("python version:", sys.version)

for pkg in ["xgboost", "sklearn", "joblib", "numpy"]:
    try:
        mod = __import__(pkg)
        print(f"{pkg}: {mod.__version__}")
    except Exception as e:
        print(f"{pkg}: FAILED TO IMPORT ({e})")

print()
print("Expected (trained on original machine): xgboost 3.2.0, scikit-learn "
      "1.9.0, joblib 1.5.3, numpy 2.4.6")
print()

art_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_artifacts")
for fname in ["xgb_classifier.joblib", "xgb_regressor.joblib", "scaler.joblib",
              "label_encoder.joblib"]:
    path = os.path.join(art_dir, fname)
    print(f"--- {fname} ---")
    if not os.path.exists(path):
        print("  MISSING FILE")
        continue
    print("  size on disk:", os.path.getsize(path), "bytes")
    try:
        import joblib
        obj = joblib.load(path)
        print("  loaded OK:", type(obj))
    except Exception:
        print("  FAILED:")
        traceback.print_exc()
    print()
