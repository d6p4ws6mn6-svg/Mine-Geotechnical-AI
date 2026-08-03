# Mine Geotechnical AI

Piezometer-driven slope stability prediction (XGBoost hybrid regressor/classifier) with a parametric
geometry solver and 3D visualization, built with Streamlit.

## Repository layout

Streamlit Community Cloud only needs three things to be right: the entrypoint file, a
`requirements.txt` next to it (or at repo root), and every file the app reads from disk actually
committed to the repo. Lay the repo out like this:

```
your-repo/
├── mine_app_streamlit.py        # entrypoint
├── requirements.txt
├── .gitignore
└── model_artifacts/
    ├── xgb_classifier.json
    ├── xgb_regressor.json
    ├── scaler.joblib
    ├── label_encoder.joblib
    └── config.json
```

`model_artifacts/` is **not optional** — the app calls `st.stop()` on import if any of those five
files are missing, so the app will fail immediately on first load if this folder isn't pushed.
Git has no problem with `.joblib`/`.json` files, so no Git LFS is needed as long as none of them
exceed GitHub's 100 MB hard limit (the comments in the code suggest they're all well under 5 MB).

## Before you push: two things worth checking locally

1. **scikit-learn version skew.** `scaler.joblib` and `label_encoder.joblib` are pickled
   scikit-learn objects (`StandardScaler`, `LabelEncoder`). Unlike the XGBoost models — which the
   code already migrated to the version-stable JSON format for exactly this reason — these two are
   still plain pickles, so they're only guaranteed to load cleanly with the *same major.minor*
   scikit-learn version they were saved with. If you don't remember which version that was, run
   this once in the environment where they were created:
   ```bash
   python -c "import sklearn; print(sklearn.__version__)"
   ```
   and pin `requirements.txt` to that exact minor version (e.g. `scikit-learn==1.4.2`) rather than
   the open-ended `>=1.4` in the file below. If you want, I can rewrite `load_models()` to extract
   just the `mean_`/`scale_`/`classes_` arrays into `config.json` and drop the scikit-learn
   dependency entirely — same fix you already applied to the XGBoost artifacts — just send me the
   two `.joblib` files.

2. **Run it locally first** with the exact `requirements.txt` below in a clean virtualenv:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   streamlit run mine_app_streamlit.py
   ```
   If it loads locally in a fresh venv, it will load on Community Cloud — the two environments use
   the same resolution logic (pip against `requirements.txt`).

## requirements.txt

Included in this bundle. Notes on the choices:

- **No upper-bound pins.** Community Cloud's Python-version selector (in "Advanced settings" at
  deploy time) has had bugs where the `runtime.txt`/dropdown selection doesn't consistently match
  what actually builds. Leaving versions open (`>=`) lets pip resolve whatever's compatible with
  the Python it actually gets, instead of failing to resolve a tight pin against an unexpected
  interpreter.
- **`scikit-learn` is listed even though the app never does `import sklearn`.** `joblib.load()` on
  the scaler/label-encoder pickles needs the class definitions available to unpickle, or you'll get
  a `ModuleNotFoundError: No module named 'sklearn'` at runtime.
- **`google-genai`, not `google-generativeai`.** The code does `from google import genai`, which is
  the new unified SDK package name — the old `google-generativeai` package exposes a different
  import path and won't work here.

## Deploying

1. Push the repo (with `model_artifacts/` included) to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Pick the repo, branch, and set the main file path to `mine_app_streamlit.py`.
4. In **Advanced settings**, explicitly select a Python version (3.11 or 3.12 is the safest
   choice for this dependency set — xgboost's newest releases require 3.12+, so avoid selecting
   anything older).
5. Deploy. No `st.secrets` are required to boot the app — the Gemini key is entered by the user
   at runtime via the sidebar text box — so you can deploy without configuring **Secrets** at all.

## Optional: pre-fill the Gemini key via Secrets instead of manual paste

If you'd rather not have users paste their own key, you could switch the sidebar input to default
from `st.secrets["GEMINI_API_KEY"]`. That's a code change (not a deployment-config change) — happy
to make it if you want that behavior; just say so.
