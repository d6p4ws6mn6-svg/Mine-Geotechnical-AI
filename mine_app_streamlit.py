import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json
import os
import warnings
import xgboost
warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mine Geotechnical AI",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background: #f0f2f5; }
    .stMetric { background: white; border-radius: 10px; padding: 10px; }
    .stMetric [data-testid="stMetricLabel"] * { color: #1a1a2e !important; }
    .stMetric [data-testid="stMetricValue"] * { color: #1a1a2e !important; }
    .stMetric [data-testid="stMetricLabel"] { color: #1a1a2e !important; }
    .stMetric [data-testid="stMetricValue"] { color: #1a1a2e !important; }
    .safe-box   { background:#d5f5e3; color:#145a32; border-left:6px solid #2ecc71;
                  padding:16px; border-radius:8px; margin:10px 0; }
    .unsafe-box { background:#fadbd8; color:#78281f; border-left:6px solid #e74c3c;
                  padding:16px; border-radius:8px; margin:10px 0; }
    h1 { color: #1a1a2e; }
</style>
""", unsafe_allow_html=True)

# ── Load models ───────────────────────────────────────────────────────────────
# XGBoost-only pipeline (Random Forest removed — it only fed the probability
# bars and was ~58 MB; the XGB classifier gives near-identical probabilities
# at ~1 MB). Path is anchored to this file so the app works no matter which
# directory `streamlit run` is launched from.
ART_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_artifacts")

@st.cache_resource
def load_models():
    # XGBoost's own pickle format embeds a version-sensitive internal buffer
    # (XGBoosterUnserializeFromBuffer) that isn't guaranteed readable across
    # major xgboost versions — loading a model pickled by one version with a
    # different version installed raises "input stream corrupted", which bit
    # someone cloning this repo onto a machine with a newer xgboost. XGBoost's
    # own save_model()/load_model() JSON format is explicitly cross-version
    # stable, so the two XGBoost artifacts use that instead of joblib.
    xgb_clf = xgboost.XGBClassifier()
    xgb_clf.load_model(os.path.join(ART_DIR, "xgb_classifier.json"))
    xgb_reg = xgboost.XGBRegressor()
    xgb_reg.load_model(os.path.join(ART_DIR, "xgb_regressor.json"))
    scaler  = joblib.load(os.path.join(ART_DIR, "scaler.joblib"))
    le      = joblib.load(os.path.join(ART_DIR, "label_encoder.joblib"))
    with open(os.path.join(ART_DIR, "config.json")) as f:
        cfg = json.load(f)
    return xgb_clf, xgb_reg, scaler, le, cfg

try:
    xgb_clf, xgb_reg, scaler, le, cfg = load_models()
    models_loaded = True
except Exception as e:
    st.error(f"⚠️ Could not load model artifacts: {e}\nRun the notebook first to generate model_artifacts/")
    models_loaded = False
    st.stop()

ALL_FEATURES      = cfg["all_features"]
FS_THRESHOLD      = cfg.get("fs_threshold", 1.3)
CONFIDENCE_MARGIN = cfg.get("confidence_margin", 0.10)

ROCK_TYPES = cfg.get("rock_types", ["Shale","Granite","Limestone","Sandstone","Basalt","Weathered Clay"])

# ── Core functions ────────────────────────────────────────────────────────────
def fs_to_label(f): return "Safe" if f >= FS_THRESHOLD else "Unsafe"

def compute_features(d: dict) -> np.ndarray:
    """Build the full 47-feature vector from site input dict."""
    import math
    d = d.copy()
    d["month"]          = (d.get("day_of_year", 180) // 30) + 1
    d["quarter"]        = ((d["month"] - 1) // 3) + 1
    d["is_wet_season"]  = 1 if d["month"] in [4,5,6,7,8,9] else 0
    d["drainage_factor"]= 1 if d.get("drainage_installed", False) else 0

    for w in [7, 14, 30]:
        if f"pwp_roll{w}"  not in d: d[f"pwp_roll{w}"]  = d["pore_water_pressure_kPa"]
        if f"pwp_std{w}"   not in d: d[f"pwp_std{w}"]   = 0.0
        if f"wl_roll{w}"   not in d: d[f"wl_roll{w}"]   = d["water_level_m"]
        if f"rain_cum{w}"  not in d: d[f"rain_cum{w}"]  = d.get("rainfall_index", 0.4) * w
    d.setdefault("pwp_roc1", 0.0); d.setdefault("pwp_roc3", 0.0); d.setdefault("wl_roc1", 0.0)

    a   = math.radians(d["slope_angle_deg"])
    g   = d["unit_weight_kN_m3"]; H = d["slope_height_m"]
    c   = d["cohesion_kPa"];      phi = math.radians(d["friction_angle_deg"])
    U   = d["pore_water_pressure_kPa"]
    den = g * H * math.sin(a) * math.cos(a) + 1e-9

    d["fs_proxy"]         = min(max((c + g*H*math.cos(a)**2*math.tan(phi)) / den, 0.1), 5.0)
    d["fs_proxy_pwp"]     = min(max((c + (g*H*math.cos(a)**2 - U/(H+1e-9))*math.tan(phi)) / den, 0.1), 5.0)
    d["ru_coefficient"]   = min(U / (g*H*math.cos(a)**2 + 1e-9), 2.0)
    d["bench_ratio"]      = d["bench_width_m"] / (H + 1e-9)
    d["saturation_proxy"] = min(d["water_level_m"] / (d.get("piezo_depth_m", 30) + 1e-9), 1.0)
    d["pore_pressure_ratio"]   = U / (g*H + 1e-9)
    d["slope_water_interact"]  = d["slope_angle_deg"] * d["water_level_m"]
    d["overall_slope_angle_deg"] = d.get("overall_slope_angle_deg", d["slope_angle_deg"] - 5)

    for rt in ROCK_TYPES:
        # Must match config.json's all_features names exactly (e.g.
        # "rock_Weathered Clay" — with a space, no underscore). A mismatched
        # key here silently zeros out that rock type's one-hot feature since
        # compute_features falls back to d.get(f, 0.0) for every ALL_FEATURES
        # entry.
        d[f"rock_{rt}"] = 1 if d.get("rock_type","") == rt else 0
    for ft in ALL_FEATURES:
        d.setdefault(ft, 0.0)

    return np.array([float(d.get(f, 0.0)) for f in ALL_FEATURES])

def predict(vec: np.ndarray) -> dict:
    """Hybrid prediction: XGB regressor decides unless the predicted FS falls
    inside the boundary zone (±CONFIDENCE_MARGIN of the 1.3 threshold), in
    which case the XGB classifier breaks the tie. Probabilities come from the
    XGB classifier (Random Forest removed from the pipeline)."""
    vec_sc  = scaler.transform(vec.reshape(1, -1))
    fs_pred = float(xgb_reg.predict(vec_sc)[0])
    dist    = abs(fs_pred - FS_THRESHOLD)
    if dist > CONFIDENCE_MARGIN:
        label = fs_to_label(fs_pred)
    else:
        label = le.inverse_transform(xgb_clf.predict(vec_sc))[0]
    clf_prob = xgb_clf.predict_proba(vec_sc)[0]
    probs    = {cls: float(p) for cls, p in zip(le.classes_, clf_prob)}
    # Confidence = classifier probability of the predicted label. (The old
    # formula 1 - dist/0.3 was inverted: it gave LOW confidence to slopes far
    # from the threshold, i.e. the most clear-cut cases.)
    confidence = probs.get(label, max(probs.values()))
    return {"label": label, "fs": round(fs_pred, 3),
            "confidence": round(confidence, 3), "probs": probs}

def parametric_solver(d: dict) -> dict:
    import math
    orig_angle = d["slope_angle_deg"]
    orig_bench = d["bench_width_m"]
    orig_n     = d.get("number_of_benches", 5)
    orig_ba    = d.get("bench_angle_deg", 75)

    base_pred  = predict(compute_features(d))
    if base_pred["label"] == "Safe":
        return {"adjustments": [], "best_config": d.copy(),
                "best_label": "Safe", "best_fs": base_pred["fs"],
                "current_fs": base_pred["fs"], "current_label": "Safe"}

    best_fs, best_cfg = base_pred["fs"], None

    for drain in [d.get("drainage_installed", False), True]:
        for angle in np.linspace(max(orig_angle - 18, 18), orig_angle, 7):
            for ba in np.linspace(max(orig_ba - 15, 45), orig_ba, 3):
                for bench_mult in np.linspace(1.0, 1.8, 6):
                    for extra_benches in [0, 1, 2]:
                        adj = d.copy()
                        adj["slope_angle_deg"]          = round(float(angle), 2)
                        adj["overall_slope_angle_deg"]   = round(float(angle) - 5, 2)
                        adj["bench_angle_deg"]          = round(float(ba), 2)
                        adj["bench_width_m"]             = round(orig_bench * bench_mult, 2)
                        adj["number_of_benches"]         = orig_n + extra_benches
                        adj["drainage_installed"]        = drain
                        res = predict(compute_features(adj))
                        if res["fs"] > best_fs:
                            best_fs, best_cfg = res["fs"], (adj.copy(), res)
                        if res["label"] == "Safe":
                            break
                    if best_cfg and best_cfg[1]["label"] == "Safe":
                        break
                if best_cfg and best_cfg[1]["label"] == "Safe":
                    break
            if best_cfg and best_cfg[1]["label"] == "Safe":
                break
        if best_cfg and best_cfg[1]["label"] == "Safe":
            break

    adj_list = []
    if best_cfg:
        bc = best_cfg[0]
        if abs(orig_angle - bc["slope_angle_deg"]) > 0.5:
            adj_list.append(f"Reduce slope angle: {orig_angle:.1f}° → {bc['slope_angle_deg']:.1f}°")
        if abs(orig_ba - bc.get("bench_angle_deg", orig_ba)) > 0.5:
            adj_list.append(f"Reduce bench face angle: {orig_ba:.1f}° → {bc['bench_angle_deg']:.1f}°")
        if abs(orig_bench - bc["bench_width_m"]) > 0.2:
            adj_list.append(f"Widen bench: {orig_bench:.1f} m → {bc['bench_width_m']:.1f} m")
        if bc.get("number_of_benches", orig_n) != orig_n:
            adj_list.append(f"Add benches: {orig_n} → {bc['number_of_benches']}")
        if bc["drainage_installed"] and not d.get("drainage_installed", False):
            adj_list.append("Install drainage system")
        return {"adjustments": adj_list,
                "best_config": best_cfg[0], "best_label": best_cfg[1]["label"],
                "best_fs": round(best_fs, 3),
                "current_fs": base_pred["fs"], "current_label": base_pred["label"]}
    return {"adjustments": ["No safe configuration found — detailed site investigation required"],
            "best_config": d.copy(), "best_label": "Unsafe",
            "best_fs": round(best_fs, 3),
            "current_fs": base_pred["fs"], "current_label": base_pred["label"]}

# Slider default for slope_angle_deg — at this value the stretch factor below
# is 1.0, so bench_angle_deg alone reproduces the original unscaled geometry.
REF_SLOPE_ANGLE_DEG = 35.0

def bench_face_run(H: float, nb: int, bench_angle_deg: float, slope_angle_deg: float) -> float:
    """Horizontal run of one bench face.

    bench_angle_deg sets the natural per-bench steepness (steeper face → shorter
    run). slope_angle_deg then stretches/compresses that run relative to the
    slider default, so the overall slope visibly flattens or steepens with the
    parametric solver's main lever without cancelling out bench_angle_deg's
    own effect.
    """
    bench_h      = H / nb
    ba_rad       = np.deg2rad(np.clip(bench_angle_deg, 30, 89))
    natural_run  = bench_h / (np.tan(ba_rad) + 1e-9)
    stretch      = (np.tan(np.deg2rad(REF_SLOPE_ANGLE_DEG)) /
                     (np.tan(np.deg2rad(np.clip(slope_angle_deg, 5, 85))) + 1e-9))
    return max(natural_run * stretch, 0.3)

def plot_3d_slope(cfg_dict: dict, safe: bool, title: str):
    """Interactive Plotly 3D benched slope — drag to rotate, scroll to zoom,
    right-drag to pan. Works in Streamlit fullscreen view too."""
    import plotly.graph_objects as go

    angle = cfg_dict.get("slope_angle_deg", 35)
    H     = cfg_dict.get("slope_height_m", 50)
    bw    = cfg_dict.get("bench_width_m", 10)
    ba    = cfg_dict.get("bench_angle_deg", 75)
    nb    = max(int(cfg_dict.get("number_of_benches", 5)), 1)
    W     = 80.0

    bench_h  = H / nb
    face_run = bench_face_run(H, nb, ba, angle)

    profile = [(0.0, 0.0)]
    x, z = 0.0, 0.0
    for _ in range(nb):
        x_top = x + face_run
        z_top = z + bench_h
        profile.append((x_top, z_top))
        profile.append((x_top + bw, z_top))
        x, z = x_top + bw, z_top

    face_c = "#2ecc71" if safe else "#e74c3c"
    alt_c  = "#27ae60" if safe else "#c0392b"
    max_x  = max(p[0] for p in profile)

    # Build one Mesh3d: each quad panel (face or berm) → 4 vertices, 2 triangles
    vx, vy, vz, tri_i, tri_j, tri_k, tri_color = [], [], [], [], [], [], []

    def add_quad(p0, p1, p2, p3, color):
        base = len(vx)
        for (qx, qy, qz) in (p0, p1, p2, p3):
            vx.append(qx); vy.append(qy); vz.append(qz)
        tri_i.extend([base, base]);     tri_j.extend([base + 1, base + 2])
        tri_k.extend([base + 2, base + 3])
        tri_color.extend([color, color])

    for i in range(len(profile) - 1):
        x0, z0 = profile[i]; x1, z1 = profile[i + 1]
        is_berm = abs(z1 - z0) < 0.01
        color = "#7f8c8d" if is_berm else (face_c if i % 2 == 0 else alt_c)
        add_quad((x0, 0, z0), (x1, 0, z1), (x1, W, z1), (x0, W, z0), color)

    # Ground plane at the toe
    add_quad((0, 0, 0), (max_x + 20, 0, 0), (max_x + 20, W, 0), (0, W, 0), "#34495e")
    # Crest plateau behind the top bench
    top_x, top_z = profile[-1]
    add_quad((top_x, 0, top_z), (top_x + 15, 0, top_z),
             (top_x + 15, W, top_z), (top_x, W, top_z), "#5d6d7e")

    mesh = go.Mesh3d(
        x=vx, y=vy, z=vz, i=tri_i, j=tri_j, k=tri_k,
        facecolor=tri_color, opacity=0.95, flatshading=True,
        lighting=dict(ambient=0.55, diffuse=0.7, specular=0.15),
        hoverinfo="skip", name="slope"
    )

    # Bench crest outlines at both walls for definition
    edge_traces = []
    for yy in (0.0, W):
        exs = [p[0] for p in profile]; ezs = [p[1] for p in profile]
        edge_traces.append(go.Scatter3d(
            x=exs, y=[yy] * len(exs), z=ezs, mode="lines",
            line=dict(color="#ecf0f1", width=3), hoverinfo="skip", showlegend=False))

    status = "✅ SAFE" if safe else "🔴 UNSAFE"
    fig = go.Figure(data=[mesh] + edge_traces)
    fig.update_layout(
        title=dict(
            text=f"{title}<br><sup>{status}  |  Slope {angle:.1f}°  H={H:.0f} m  "
                 f"Bench {bw:.1f} m  n={nb}</sup>",
            font=dict(color="white", size=15), x=0.5, y=0.90, yanchor="top"),
        paper_bgcolor="#1a1a2e",
        scene=dict(
            bgcolor="#16213e",
            xaxis=dict(title="Horizontal (m)", color="white",
                       gridcolor="rgba(255,255,255,0.15)", backgroundcolor="rgba(0,0,0,0)"),
            yaxis=dict(title="Width (m)", color="white",
                       gridcolor="rgba(255,255,255,0.15)", backgroundcolor="rgba(0,0,0,0)"),
            zaxis=dict(title="Elevation (m)", color="white",
                       gridcolor="rgba(255,255,255,0.15)", backgroundcolor="rgba(0,0,0,0)"),
            aspectmode="data",
            camera=dict(eye=dict(x=1.6, y=-1.6, z=0.8)),
        ),
        margin=dict(l=0, r=0, t=130, b=0),
        height=480,
        updatemenus=[dict(
            type="buttons", direction="right",
            x=0.5, y=1.0, xanchor="center", yanchor="top",
            showactive=True, bgcolor="#0f3460", bordercolor="#4ecca3",
            font=dict(color="white", size=11), pad=dict(t=2, b=2),
            buttons=[
                dict(label="Isometric", method="relayout",
                     args=[{"scene.camera.eye": dict(x=1.6, y=-1.6, z=0.8)}]),
                dict(label="Front", method="relayout",
                     args=[{"scene.camera.eye": dict(x=0.01, y=-2.6, z=0.15)}]),
                dict(label="Side", method="relayout",
                     args=[{"scene.camera.eye": dict(x=2.6, y=0.01, z=0.15)}]),
                dict(label="Top", method="relayout",
                     args=[{"scene.camera.eye": dict(x=0.01, y=0.01, z=2.9)}]),
            ],
        )],
    )
    return fig

def plot_profile_2d(before: dict, after: dict) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#1a1a2e")
    for ax, cfg_d, safe, label in [
        (axes[0], before, False, "Before — UNSAFE"),
        (axes[1], after,  True,  "After — SAFE"),
    ]:
        angle  = cfg_d.get("slope_angle_deg", 35)
        H      = cfg_d.get("slope_height_m", 50)
        bw     = cfg_d.get("bench_width_m", 10)
        ba     = cfg_d.get("bench_angle_deg", 75)
        nb     = max(int(cfg_d.get("number_of_benches", 5)), 1)
        bench_h = H / nb

        # Same shared geometry as plot_3d_slope: bench_angle_deg sets face
        # steepness, slope_angle_deg stretches the whole profile.
        face_run = bench_face_run(H, nb, ba, angle)

        profile = [(0.0, 0.0)]
        x, z = 0.0, 0.0
        for _ in range(nb):
            xt = x + face_run
            zt = z + bench_h
            profile.append((xt, zt))
            profile.append((xt + bw, zt))
            x, z = xt + bw, zt

        xs = [p[0] for p in profile]; zs = [p[1] for p in profile]
        color = "#2ecc71" if safe else "#e74c3c"
        ax.fill_betweenx(zs, 0, xs, alpha=0.25, color=color)
        ax.plot(xs, zs, color=color, lw=2.5, marker="o", ms=5)
        ax.set_facecolor("#16213e")
        ax.set_xlabel("Horizontal (m)", color="white"); ax.set_ylabel("Elevation (m)", color="white")
        ax.tick_params(colors="white")
        ax.set_title(label, color="white", fontweight="bold")
        ax.grid(alpha=0.2, color="white")
        for spine in ax.spines.values(): spine.set_edgecolor("#444")
        stats = f"Slope {angle:.1f}°  |  Bench {bw:.1f}m  |  {nb} benches"
        ax.text(0.02, 0.97, stats, transform=ax.transAxes, color="white",
                fontsize=9, va="top", style="italic")
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ═══════════════════════════════════════════════════════════════════════════════

st.title("⛏️ Mine Geotechnical AI")
st.markdown("**Piezometer-driven slope stability prediction with 3D geometry redesign**")
st.markdown("---")

# ── Sidebar: API key ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("Gemini API key (optional)", type="password",
                             help="Paste your Google Gemini API key for AI-generated engineering reports")
    st.markdown("---")
    st.markdown("**Decision model:** XGBoost Hybrid (Regressor + Classifier)")
    st.markdown(f"**Accuracy:** {cfg.get('best_accuracy',0)*100:.1f}%")
    st.markdown(f"**AUC:** {cfg.get('best_auc',0):.3f}")
    st.markdown(f"**FS threshold:** {FS_THRESHOLD}")
    st.markdown(f"**Boundary margin:** ±{CONFIDENCE_MARGIN:.3f}")
    st.markdown("---")
    st.info("Enter site parameters → predict → if Unsafe, the solver finds the safe geometry and renders the 3D model.")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["⚡ Predict & Design", "📋 How it works"])

with tab2:
    st.markdown("""
    ### Pipeline
    1. **Enter site parameters** — sensor readings + slope geometry + rock properties
    2. **Predict** — XGBoost Regressor predicts the safety factor; if the value falls in the
       boundary zone around FS = 1.3, the XGBoost Classifier breaks the tie (hybrid decision)
    3. **If Unsafe** — Parametric solver searches combinations of slope angle, bench width,
       bench count, and drainage to find the minimum geometry change that achieves FS ≥ 1.3
    4. **3D visualisation** — before (unsafe) and after (safe) slope geometry rendered side-by-side
    5. **Engineering report** — optional Gemini AI narrative with justification

    ### Safety factor interpretation
    | FS | Class | Action |
    |---|---|---|
    | ≥ 1.3 | ✅ Safe | Normal operations |
    | < 1.3 | 🔴 Unsafe | Halt, redesign |
    """)

with tab1:
    col_left, col_right = st.columns([1, 1.4])

    with col_left:
        st.subheader("📥 Site Parameters")

        with st.expander("🌊 Sensor Readings", expanded=True):
            pore_wp = st.number_input("Pore water pressure (kPa)", 10.0, 300.0, 145.0, 5.0)
            wl      = st.number_input("Water level (m)",           0.0,  60.0,  6.5,   0.5)
            rain    = st.slider("Rainfall index",                   0.0,  2.5,   0.5,  0.05)
            temp    = st.number_input("Temperature (°C)",          10.0, 45.0,  22.0,  1.0)

        with st.expander("📐 Slope Geometry", expanded=True):
            slope_angle = st.slider("Slope angle (°)",          18.0, 55.0, 38.0, 0.5)
            slope_h     = st.number_input("Slope height (m)",   10.0, 100.0, 52.0, 1.0)
            bench_w     = st.number_input("Bench width (m)",     3.0,  30.0,  9.0, 0.5)
            bench_angle = st.slider("Bench face angle (°)",     55.0,  89.0, 75.0, 1.0)
            n_benches   = st.slider("Number of benches",           2,    15,    5,   1)
            drainage    = st.checkbox("Drainage system installed", value=False)

        with st.expander("🪨 Rock & Geomechanical Properties"):
            rock_type   = st.selectbox("Rock type", ROCK_TYPES)
            cohesion    = st.number_input("Cohesion (kPa)",           1.0, 60.0, 14.0, 1.0)
            friction    = st.slider("Friction angle (°)",            10.0, 55.0, 30.0, 0.5)
            unit_wt     = st.number_input("Unit weight (kN/m³)",    16.0, 30.0, 22.0, 0.5)
            piezo_depth = st.number_input("Piezometer depth (m)",     5.0, 60.0, 30.0, 1.0)
            doy         = st.slider("Day of year",                       1,  366,  180,   1)

        predict_btn = st.button("⚡ Predict Stability", width="stretch", type="primary")

    with col_right:
        st.subheader("📊 Results")

        if predict_btn:
            site = {
                "pore_water_pressure_kPa": pore_wp, "water_level_m": wl,
                "rainfall_index": rain, "temperature_C": temp,
                "slope_angle_deg": slope_angle, "slope_height_m": slope_h,
                "bench_width_m": bench_w, "bench_angle_deg": bench_angle,
                "overall_slope_angle_deg": slope_angle - 5,
                "number_of_benches": n_benches, "drainage_installed": drainage,
                "rock_type": rock_type, "cohesion_kPa": cohesion,
                "friction_angle_deg": friction, "unit_weight_kN_m3": unit_wt,
                "piezo_depth_m": piezo_depth, "piezo_elevation_masl": 200.0,
                "day_of_year": doy,
            }

            with st.spinner("Running prediction..."):
                vec    = compute_features(site)
                result = predict(vec)

            # ── Prediction result ─────────────────────────────────────────────
            label  = result["label"]
            fs_val = result["fs"]
            color  = "#2ecc71" if label == "Safe" else "#e74c3c"
            icon   = "✅" if label == "Safe" else "🔴"

            c1, c2, c3 = st.columns(3)
            c1.metric("Stability Class", f"{icon} {label}")
            c2.metric("Safety Factor",   f"{fs_val:.3f}", f"threshold {FS_THRESHOLD}")
            c3.metric("Confidence",      f"{result['confidence']*100:.0f}%")

            # Probability bars (from XGBoost classifier)
            st.markdown("**Class probabilities**")
            for cls, prob in sorted(result["probs"].items(), key=lambda x: -x[1]):
                clr = "#2ecc71" if cls=="Safe" else "#e74c3c"
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:8px;margin:3px 0">'
                    f'<span style="width:70px;font-size:13px">{cls}</span>'
                    f'<div style="flex:1;background:#eee;border-radius:4px;height:14px">'
                    f'<div style="width:{prob*100:.0f}%;background:{clr};height:100%;border-radius:4px"></div></div>'
                    f'<span style="font-size:12px;width:45px">{prob*100:.1f}%</span></div>',
                    unsafe_allow_html=True)

            st.markdown("---")

            if label == "Safe":
                st.markdown('<div class="safe-box"><b>✅ Site is SAFE</b><br>Normal mining operations may continue. Continue regular piezometer monitoring.</div>', unsafe_allow_html=True)
                fig3d = plot_3d_slope(site, safe=True, title="Current Slope — SAFE")
                st.plotly_chart(fig3d, width="stretch",
                                config={"displayModeBar": True})
                st.caption("🖱️ Drag to rotate · scroll to zoom · right-drag to pan")

            else:
                st.markdown('<div class="unsafe-box"><b>🔴 Site is UNSAFE — Operations must halt</b><br>Running parametric geometry solver...</div>', unsafe_allow_html=True)

                with st.spinner("🔧 Finding safe geometry configuration..."):
                    solver = parametric_solver(site)

                st.markdown("### 🔧 Recommended Design Adjustments")
                if solver["adjustments"]:
                    for adj in solver["adjustments"]:
                        st.markdown(
                            f'<div style="background:#fff3cd;color:#1a1a2e;'
                            f'border-left:6px solid #f39c12;padding:12px 16px;'
                            f'border-radius:8px;margin:6px 0;font-weight:600;'
                            f'font-size:15px">🔧 {adj}</div>',
                            unsafe_allow_html=True)
                else:
                    st.warning("No standard adjustment achieved FS ≥ 1.3. Full geotechnical investigation required.")

                bc = solver["best_config"]
                d1, d2, d3 = st.columns(3)
                d1.metric("Post-adjustment Class", f"{'✅' if solver['best_label']=='Safe' else '🔴'} {solver['best_label']}")
                d2.metric("Post-adjustment FS",    f"{solver['best_fs']:.3f}", f"was {solver['current_fs']:.3f}")
                d3.metric("FS improvement",        f"+{solver['best_fs']-solver['current_fs']:.3f}")

                st.markdown("---")
                st.markdown("### 🏗️ 3D Geometry — Before vs After Redesign")

                with st.spinner("Rendering 3D geometry..."):
                    fig_2d = plot_profile_2d(site, bc)
                st.pyplot(fig_2d, width="stretch")
                plt.close(fig_2d)

                col3d_a, col3d_b = st.columns(2)
                with col3d_a:
                    st.markdown("**Before (Unsafe)**")
                    fig_before = plot_3d_slope(site, safe=False,
                                               title=f"UNSAFE  FS={solver['current_fs']:.3f}")
                    st.plotly_chart(fig_before, width="stretch",
                                    config={"displayModeBar": True})

                with col3d_b:
                    st.markdown("**After (Redesigned)**")
                    fig_after = plot_3d_slope(bc, safe=(solver['best_label']=='Safe'),
                                              title=f"{'SAFE' if solver['best_label']=='Safe' else 'UNSAFE'}"
                                                    f"  FS={solver['best_fs']:.3f}")
                    st.plotly_chart(fig_after, width="stretch",
                                    config={"displayModeBar": True})
                st.caption("🖱️ Drag to rotate · scroll to zoom · right-drag to pan")

                # ── Geometry comparison table ─────────────────────────────────
                st.markdown("### 📋 Geometry Parameter Comparison")
                comp_df = pd.DataFrame({
                    "Parameter":  ["Slope angle (°)", "Bench face angle (°)", "Bench width (m)",
                                   "No. benches", "Drainage"],
                    "Before":     [f'{site["slope_angle_deg"]:.1f}', f'{site["bench_angle_deg"]:.1f}',
                                   f'{site["bench_width_m"]:.1f}', str(site["number_of_benches"]),
                                   "Yes" if site["drainage_installed"] else "No"],
                    "After":      [f'{bc.get("slope_angle_deg", site["slope_angle_deg"]):.1f}',
                                   f'{bc.get("bench_angle_deg", site["bench_angle_deg"]):.1f}',
                                   f'{bc.get("bench_width_m", site["bench_width_m"]):.1f}',
                                   str(bc.get("number_of_benches", site["number_of_benches"])),
                                   "Yes" if bc.get("drainage_installed", False) else "No"],
                })
                # Force a string dtype: pyarrow's type-inference on an object
                # column of mostly numeric-looking strings ("38.0", "9.0")
                # guesses "double" and then chokes on "Yes"/"No", crashing the
                # Arrow serialization Streamlit needs to render the table.
                comp_df = comp_df.astype("string")
                st.dataframe(comp_df, width="stretch", hide_index=True)

                # ── LLM report ────────────────────────────────────────────────
                if api_key:
                    st.markdown("### 🤖 AI Engineering Report")
                    with st.spinner("Generating report via Gemini API..."):
                        try:
                            from google import genai
                            client = genai.Client(api_key=api_key)
                            adj_text = "\n".join(solver["adjustments"])
                            prompt = f"""You are a senior geotechnical engineer. Write a concise 3-paragraph site assessment report.

SITE DATA:
- Rock: {rock_type} | Slope: {slope_angle:.1f}° × {slope_h:.0f}m | Bench face angle: {bench_angle:.1f}° | Bench: {bench_w:.1f}m | Benches: {n_benches}
- Drainage installed: {"Yes" if drainage else "No"}
- Pore pressure: {pore_wp:.0f} kPa | ru coefficient: {vec[ALL_FEATURES.index('ru_coefficient')]:.3f}
- Cohesion: {cohesion:.1f} kPa | Friction: {friction:.1f}° | Unit weight: {unit_wt:.1f} kN/m³
- FS predicted: {fs_val:.3f} (UNSAFE, threshold {FS_THRESHOLD})

RECOMMENDED ADJUSTMENTS:
{adj_text}
Post-adjustment FS: {solver['best_fs']:.3f}

Cover: (1) why the site is unsafe referencing ru and Bishop FS, (2) engineering justification for each adjustment, (3) monitoring plan."""
                            from google.genai import errors as genai_errors
                            import time
                            response = None
                            last_err = None
                            for attempt in range(3):
                                try:
                                    response = client.models.generate_content(
                                        model="gemini-2.5-flash", contents=prompt)
                                    break
                                except genai_errors.ServerError as e:
                                    # Transient overload (503/500) — Google's
                                    # shared capacity, not our credentials or
                                    # request. Retry a couple of times before
                                    # giving up.
                                    last_err = e
                                    if attempt < 2:
                                        time.sleep(2 * (attempt + 1))
                            if response is not None:
                                st.markdown(response.text)
                            else:
                                st.warning(
                                    "Gemini is temporarily overloaded (503) after "
                                    f"3 attempts — try again in a minute. ({last_err})")
                        except Exception as e:
                            st.warning(f"API report failed: {e}")
                elif label == "Unsafe":
                    st.info("💡 Add your Gemini API key in the sidebar for an AI-generated engineering report.")

        else:
            st.info("👈 Fill in site parameters and click **Predict Stability**")

st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#888; font-size:13px'>"
    "Powered by Manuel, Alex, and Mercy</p>",
    unsafe_allow_html=True,
)