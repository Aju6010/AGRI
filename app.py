from __future__ import annotations

import json
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from agri_ml.config import ARTIFACT_DIR, CROP_FEATURES, YIELD_FEATURES
from agri_ml.models import predict_crop
from agri_ml.train import train_project


st.set_page_config(page_title="AgriSense", page_icon="AG", layout="wide", initial_sidebar_state="expanded")

TOPBAR_RENDERED = False


@dataclass(frozen=True)
class CropProfile:
    name: str
    group: str
    season: str
    water: str
    duration: str
    soil: str
    story: str
    image: str


@st.cache_resource(show_spinner="Preparing AgriSense models...")
def load_or_train() -> dict[str, object]:
    metadata_path = ARTIFACT_DIR / "metadata.json"
    required = [
        ARTIFACT_DIR / "crop_classifier.joblib",
        ARTIFACT_DIR / "yield_regressor.joblib",
        ARTIFACT_DIR / "crop_metrics.csv",
        ARTIFACT_DIR / "yield_metrics.csv",
        ARTIFACT_DIR / "crop_feature_importance.csv",
        ARTIFACT_DIR / "yield_feature_importance.csv",
        ARTIFACT_DIR / "completed_yield_data.csv",
        ARTIFACT_DIR / "yield_2025_forecast.csv",
    ]
    if not metadata_path.exists() or any(not path.exists() for path in required):
        train_project()

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {
        "metadata": metadata,
        "crop_model": joblib.load(ARTIFACT_DIR / "crop_classifier.joblib"),
        "yield_model": joblib.load(ARTIFACT_DIR / "yield_regressor.joblib"),
        "crop_metrics": pd.read_csv(ARTIFACT_DIR / "crop_metrics.csv"),
        "yield_metrics": pd.read_csv(ARTIFACT_DIR / "yield_metrics.csv"),
        "crop_importance": pd.read_csv(ARTIFACT_DIR / "crop_feature_importance.csv"),
        "yield_importance": pd.read_csv(ARTIFACT_DIR / "yield_feature_importance.csv"),
        "completed_yield": pd.read_csv(ARTIFACT_DIR / "completed_yield_data.csv"),
        "yield_forecast": pd.read_csv(ARTIFACT_DIR / "yield_2025_forecast.csv"),
    }


def crop_profiles() -> list[CropProfile]:
    field = "https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&w=1400&q=80"
    grain = "https://images.unsplash.com/photo-1574323347407-f5e1ad6d020b?auto=format&fit=crop&w=1400&q=80"
    fruit = "https://images.unsplash.com/photo-1464226184884-fa280b87c399?auto=format&fit=crop&w=1400&q=80"
    orchard = "https://images.unsplash.com/photo-1560493676-04071c5f467b?auto=format&fit=crop&w=1400&q=80"
    cotton = "https://images.unsplash.com/photo-1598515214211-89d3c73ae83b?auto=format&fit=crop&w=1400&q=80"
    coffee = "https://images.unsplash.com/photo-1525088553748-01d6e210e00b?auto=format&fit=crop&w=1400&q=80"
    return [
        CropProfile("Rice", "Cereal", "Monsoon", "High", "120-150 days", "Clay loam", "Performs best where rainfall and humidity stay high.", field),
        CropProfile("Wheat", "Cereal", "Rabi", "Medium", "110-140 days", "Well drained loam", "A planning anchor for winter acreage and yield forecasting.", grain),
        CropProfile("Maize", "Cereal", "Kharif", "Medium", "90-120 days", "Fertile loam", "A flexible option when rainfall is moderate and nutrients are balanced.", grain),
        CropProfile("Jute", "Fiber", "Monsoon", "High", "100-120 days", "Alluvial soil", "Strong under warm, humid, rainfall-rich field conditions.", field),
        CropProfile("Cotton", "Fiber", "Kharif", "Medium", "150-180 days", "Black soil", "Needs nutrient support and a reliable warm season.", cotton),
        CropProfile("Banana", "Fruit", "Annual", "High", "10-15 months", "Deep loam", "Rewards consistent moisture, potassium, and warm temperatures.", fruit),
        CropProfile("Mango", "Fruit", "Perennial", "Low", "Perennial", "Sandy loam", "A dry-warm orchard crop with lower humidity preference.", orchard),
        CropProfile("Coffee", "Plantation", "Perennial", "Medium", "Perennial", "Slightly acidic", "Needs shade, moderate rainfall, and stable warm conditions.", coffee),
    ]


def crop_presets() -> dict[str, dict[str, float]]:
    return {
        "Balanced field": {"N": 80.0, "P": 45.0, "K": 40.0, "temperature": 25.0, "humidity": 80.0, "ph": 6.5, "rainfall": 120.0},
        "High rainfall lowland": {"N": 78.0, "P": 47.0, "K": 40.0, "temperature": 24.0, "humidity": 84.0, "ph": 6.4, "rainfall": 220.0},
        "Warm dry plot": {"N": 35.0, "P": 55.0, "K": 28.0, "temperature": 31.0, "humidity": 48.0, "ph": 6.7, "rainfall": 58.0},
        "Fruit orchard": {"N": 24.0, "P": 110.0, "K": 190.0, "temperature": 23.0, "humidity": 88.0, "ph": 6.0, "rainfall": 100.0},
        "Commercial cotton": {"N": 118.0, "P": 47.0, "K": 22.0, "temperature": 25.0, "humidity": 78.0, "ph": 6.9, "rainfall": 82.0},
    }


def yield_presets(base: dict[str, float]) -> dict[str, dict[str, float]]:
    return {
        "Baseline": dict(base),
        "Dry season": {
            **base,
            "average_rain_fall_mm_per_year": max(0.0, base["average_rain_fall_mm_per_year"] * 0.82),
            "avg_temp": base["avg_temp"] + 0.8,
        },
        "Wet season": {
            **base,
            "average_rain_fall_mm_per_year": base["average_rain_fall_mm_per_year"] * 1.14,
            "avg_temp": base["avg_temp"] - 0.2,
        },
        "High input": {**base, "pesticides_tonnes": base["pesticides_tonnes"] * 1.12},
        "Low input": {**base, "pesticides_tonnes": base["pesticides_tonnes"] * 0.88},
    }


def apply_theme(mode: str) -> None:
    dark = mode == "Night"
    ink = "#f8f0e8" if dark else "#17201a"
    muted = "#c8c0b6" if dark else "#667269"
    bg = "#061a2f" if dark else "#fbfaf5"
    panel = "#0b2742" if dark else "#ffffff"
    border = "#31506a" if dark else "#d9dfd5"
    green = "#95c9a8" if dark else "#245d45"
    accent = "#cf8f7d" if dark else "#245f7f"
    hero_overlay = "rgba(3, 18, 33, 0.62)" if dark else "rgba(11, 31, 24, 0.46)"

    st.markdown(
        f"""
        <style>
        :root {{
            --ag-ink: {ink};
            --ag-muted: {muted};
            --ag-bg: {bg};
            --ag-panel: {panel};
            --ag-border: {border};
            --ag-green: {green};
            --ag-accent: {accent};
            --ag-hero-overlay: {hero_overlay};
        }}

        .stApp {{
            background: var(--ag-bg);
            color: var(--ag-ink);
        }}

        .block-container {{
            max-width: 1360px;
            padding-top: 1.1rem;
            padding-bottom: 3rem;
        }}

        [data-testid="stSidebar"] {{
            background: {"#06182b" if dark else "#f2f6f0"};
            border-right: 1px solid var(--ag-border);
        }}

        h1, h2, h3 {{
            letter-spacing: 0;
            color: var(--ag-ink);
        }}

        h1 {{
            font-size: 2.35rem;
            line-height: 1.1;
            margin-bottom: 0.25rem;
        }}

        h2 {{
            font-size: 1.35rem;
        }}

        h3 {{
            font-size: 1rem;
        }}

        .era-hero {{
            min-height: 640px;
            border: 1px solid var(--ag-border);
            border-radius: 8px;
            background:
                linear-gradient(var(--ag-hero-overlay), var(--ag-hero-overlay)),
                url("https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&w=2200&q=80");
            background-size: cover;
            background-position: center;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 1.2rem;
            overflow: hidden;
            position: relative;
            transform-style: preserve-3d;
        }}

        .era-hero:before {{
            content: "";
            position: absolute;
            inset: 6%;
            border: 1px solid rgba(255,255,255,0.28);
            border-radius: 8px;
            transform: translateZ(40px);
            pointer-events: none;
        }}

        .era-hero:after {{
            content: "AG";
            position: absolute;
            right: 4%;
            bottom: 5%;
            color: rgba(255,255,255,0.24);
            font-size: 9rem;
            font-weight: 900;
            line-height: 1;
            transform: rotate(-8deg) translateZ(70px);
        }}

        .hero-inner {{
            max-width: 860px;
            padding: 2rem;
            text-align: center;
            transform: translateZ(80px);
        }}

        .hero-kicker, .product-kicker {{
            color: var(--ag-accent);
            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: 0.1em;
            text-transform: uppercase;
        }}

        .hero-title {{
            color: #fff8ef;
            font-size: clamp(2.8rem, 8vw, 6.6rem);
            line-height: 0.9;
            margin: 0.6rem 0;
            text-transform: uppercase;
        }}

        .hero-copy {{
            color: #fff8ef;
            font-size: 1.05rem;
            line-height: 1.7;
            margin: 0 auto;
            max-width: 690px;
        }}

        .product-subtitle {{
            color: var(--ag-muted);
            font-size: 1rem;
            margin-bottom: 1rem;
        }}

        .topbar {{
            align-items: center;
            background: var(--ag-panel);
            border: 1px solid var(--ag-border);
            border-radius: 8px;
            display: flex;
            justify-content: space-between;
            margin-bottom: 1rem;
            padding: 0.72rem 0.9rem;
            position: sticky;
            top: 0.55rem;
            z-index: 30;
            backdrop-filter: blur(16px);
        }}

        .scroll-nav {{
            background: {"rgba(6, 24, 43, 0.78)" if dark else "rgba(255, 255, 255, 0.82)"};
            border: 1px solid var(--ag-border);
            border-radius: 8px;
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
            margin-bottom: 1rem;
            padding: 0.55rem;
            position: sticky;
            top: 4.35rem;
            z-index: 25;
            backdrop-filter: blur(14px);
        }}

        .scroll-nav a {{
            border: 1px solid var(--ag-border);
            border-radius: 999px;
            color: var(--ag-muted);
            font-size: 0.78rem;
            font-weight: 800;
            padding: 0.35rem 0.65rem;
            text-decoration: none;
            text-transform: uppercase;
        }}

        .scroll-nav a:hover {{
            color: var(--ag-ink);
            border-color: var(--ag-green);
        }}

        .scroll-section {{
            border-top: 1px solid var(--ag-border);
            margin-top: 2rem;
            padding-top: 2rem;
            scroll-margin-top: 8rem;
        }}

        .brand-mark {{
            color: var(--ag-green);
            font-size: 0.86rem;
            font-weight: 800;
        }}

        .chip {{
            display: inline-block;
            padding: 0.22rem 0.5rem;
            border: 1px solid var(--ag-border);
            border-radius: 999px;
            background: {"rgba(255,255,255,0.06)" if dark else "#ffffff"};
            color: var(--ag-muted);
            font-size: 0.76rem;
            margin: 0 0.28rem 0.28rem 0;
            white-space: nowrap;
        }}

        .crop-card {{
            background: var(--ag-panel);
            border: 1px solid var(--ag-border);
            border-radius: 8px;
            overflow: hidden;
            min-height: 360px;
            transform: perspective(900px) rotateX(0deg) rotateY(0deg);
            transition: transform 240ms ease, box-shadow 240ms ease;
        }}

        .crop-card:hover {{
            box-shadow: 0 24px 55px rgba(0,0,0,0.18);
            transform: perspective(900px) rotateX(2deg) rotateY(-3deg) translateY(-4px);
        }}

        .crop-card img {{
            width: 100%;
            height: 170px;
            object-fit: cover;
            display: block;
        }}

        .crop-card-body {{
            padding: 0.9rem;
        }}

        .crop-card h3 {{
            margin: 0.1rem 0 0.35rem 0;
        }}

        .crop-card p {{
            color: var(--ag-muted);
            font-size: 0.9rem;
            line-height: 1.5;
        }}

        .field-tile {{
            background: var(--ag-panel);
            border: 1px solid var(--ag-border);
            border-radius: 8px;
            padding: 0.75rem;
            transform-style: preserve-3d;
            transition: transform 220ms ease;
        }}

        .field-tile:hover {{
            transform: perspective(800px) rotateY(2deg) translateY(-3px);
        }}

        .leaf-layer {{
            inset: 0;
            overflow: hidden;
            pointer-events: none;
            position: fixed;
            z-index: 2;
        }}

        .leaf {{
            animation-name: leaf-fall;
            animation-timing-function: linear;
            animation-iteration-count: infinite;
            background: linear-gradient(135deg, #95c9a8, #4d7f4d 55%, #d1a05f);
            border-radius: 85% 0 85% 0;
            height: 18px;
            opacity: 0.46;
            position: absolute;
            top: -8vh;
            transform-origin: 50% 50%;
            width: 12px;
        }}

        .leaf:after {{
            background: rgba(255,255,255,0.28);
            content: "";
            height: 1px;
            left: 2px;
            position: absolute;
            top: 9px;
            transform: rotate(42deg);
            width: 13px;
        }}

        .leaf-1 {{ left: 7%; animation-duration: 16s; animation-delay: -2s; }}
        .leaf-2 {{ left: 18%; animation-duration: 21s; animation-delay: -8s; transform: scale(0.8); }}
        .leaf-3 {{ left: 31%; animation-duration: 18s; animation-delay: -12s; transform: scale(1.1); }}
        .leaf-4 {{ left: 43%; animation-duration: 24s; animation-delay: -4s; }}
        .leaf-5 {{ left: 56%; animation-duration: 19s; animation-delay: -10s; transform: scale(0.7); }}
        .leaf-6 {{ left: 68%; animation-duration: 25s; animation-delay: -6s; transform: scale(1.15); }}
        .leaf-7 {{ left: 81%; animation-duration: 17s; animation-delay: -14s; }}
        .leaf-8 {{ left: 92%; animation-duration: 22s; animation-delay: -3s; transform: scale(0.9); }}

        @keyframes leaf-fall {{
            0% {{ transform: translate3d(0, -12vh, 0) rotateX(0deg) rotateZ(0deg); }}
            25% {{ transform: translate3d(42px, 25vh, 70px) rotateX(130deg) rotateZ(95deg); }}
            50% {{ transform: translate3d(-26px, 52vh, 20px) rotateX(240deg) rotateZ(190deg); }}
            75% {{ transform: translate3d(34px, 78vh, 90px) rotateX(320deg) rotateZ(285deg); }}
            100% {{ transform: translate3d(-18px, 112vh, 0) rotateX(420deg) rotateZ(360deg); }}
        }}

        [data-testid="stMetric"] {{
            background: var(--ag-panel);
            border: 1px solid var(--ag-border);
            border-left: 4px solid var(--ag-green);
            border-radius: 8px;
            min-height: 104px;
            padding: 0.82rem 0.95rem;
            box-shadow: 0 1px 2px rgba(16, 24, 18, 0.04);
        }}

        [data-testid="stMetricLabel"] p {{
            color: var(--ag-muted);
            font-size: 0.8rem;
        }}

        [data-testid="stMetricValue"] {{
            color: var(--ag-ink);
            font-size: 1.3rem;
        }}

        div[data-testid="stForm"], div[data-testid="stDataFrame"] {{
            background: var(--ag-panel);
            border: 1px solid var(--ag-border);
            border-radius: 8px;
            overflow: hidden;
        }}

        div[data-testid="stForm"] {{
            padding: 1rem;
        }}

        .stButton > button,
        .stFormSubmitButton > button,
        .stDownloadButton > button {{
            border-radius: 6px;
            border: 1px solid var(--ag-green);
            background: var(--ag-green);
            color: {"#06182b" if dark else "#ffffff"};
            font-weight: 800;
        }}

        .stButton > button:hover,
        .stFormSubmitButton > button:hover,
        .stDownloadButton > button:hover {{
            opacity: 0.9;
            color: {"#06182b" if dark else "#ffffff"};
        }}

        hr {{
            border-color: var(--ag-border);
            margin: 1.25rem 0;
        }}

        html {{
            scroll-behavior: smooth;
        }}

        @media (prefers-reduced-motion: reduce) {{
            .leaf {{
                animation: none;
                display: none;
            }}

            .crop-card, .field-tile {{
                transition: none;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="leaf-layer" aria-hidden="true">
            <span class="leaf leaf-1"></span>
            <span class="leaf leaf-2"></span>
            <span class="leaf leaf-3"></span>
            <span class="leaf leaf-4"></span>
            <span class="leaf leaf-5"></span>
            <span class="leaf leaf-6"></span>
            <span class="leaf leaf-7"></span>
            <span class="leaf leaf-8"></span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def topbar(metadata: dict[str, object], mode: str) -> None:
    global TOPBAR_RENDERED
    if TOPBAR_RENDERED:
        return
    TOPBAR_RENDERED = True
    st.markdown(
        f"""
        <div class="topbar">
            <div><span class="brand-mark">AG AgriSense</span></div>
            <div>
                <span class="chip">{mode} mode</span>
                <span class="chip">Crop: {metadata["best_crop_model"]}</span>
                <span class="chip">Yield: {metadata["best_yield_model"]}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def scroll_nav() -> None:
    st.markdown(
        """
        <nav class="scroll-nav">
            <a href="#experience">Experience</a>
            <a href="#crop-studio">Crop Studio</a>
            <a href="#field-planner">Field Planner</a>
            <a href="#yield-simulator">Yield Simulator</a>
            <a href="#crop-library">Crop Library</a>
            <a href="#model-monitor">Model Monitor</a>
            <a href="#data-hub">Data Hub</a>
        </nav>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str) -> None:
    st.markdown('<div class="product-kicker">AgriSense Decision Cloud</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="product-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def format_metric_frame(frame: pd.DataFrame, percent_columns: list[str]) -> pd.DataFrame:
    display = frame.copy()
    for column in percent_columns:
        if column in display.columns:
            display[column] = display[column].map(percent)
    for column in ["mae", "mse", "rmse", "std_abs_error"]:
        if column in display.columns:
            display[column] = display[column].map(lambda value: f"{value:,.2f}")
    return display


def readiness_score(values: dict[str, float]) -> tuple[int, str]:
    ph_score = max(0.0, 1.0 - abs(values["ph"] - 6.5) / 2.8)
    moisture_score = min(1.0, (values["humidity"] / 90.0) * 0.55 + (min(values["rainfall"], 220.0) / 220.0) * 0.45)
    climate_score = max(0.0, 1.0 - abs(values["temperature"] - 26.0) / 18.0)
    nutrients = np.array([values["N"], values["P"], values["K"]], dtype=float)
    nutrient_score = min(1.0, nutrients.mean() / 95.0) * (1.0 - min(0.45, nutrients.std() / 180.0))
    score = int(round(100 * (0.28 * ph_score + 0.28 * moisture_score + 0.22 * climate_score + 0.22 * nutrient_score)))
    band = "Excellent" if score >= 82 else "Good" if score >= 68 else "Watch" if score >= 52 else "Risk"
    return score, band


def recommendation_actions(values: dict[str, float], top_crop: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Area": "Soil pH", "Signal": f"{values['ph']:.1f}", "Action": "Ready" if 5.8 <= values["ph"] <= 7.4 else "Review pH correction"},
            {"Area": "Water", "Signal": f"{values['rainfall']:.0f} mm", "Action": "Ready" if values["rainfall"] >= 70 else "Plan irrigation"},
            {"Area": "Climate", "Signal": f"{values['temperature']:.1f} C", "Action": "Ready" if 18 <= values["temperature"] <= 34 else "Check seasonal fit"},
            {"Area": "Nutrients", "Signal": f"NPK {values['N']:.0f}/{values['P']:.0f}/{values['K']:.0f}", "Action": f"Validate for {top_crop.title()}"},
        ]
    )


def crop_sensitivity(model: object, values: dict[str, float], feature: str) -> pd.DataFrame:
    ranges = {
        "rainfall": np.linspace(max(0, values[feature] - 80), values[feature] + 80, 11),
        "temperature": np.linspace(max(5, values[feature] - 7), min(45, values[feature] + 7), 11),
        "humidity": np.linspace(max(10, values[feature] - 25), min(100, values[feature] + 25), 11),
        "ph": np.linspace(max(3.5, values[feature] - 1.2), min(9.5, values[feature] + 1.2), 11),
        "N": np.linspace(max(0, values[feature] - 40), values[feature] + 40, 11),
        "P": np.linspace(max(0, values[feature] - 40), values[feature] + 40, 11),
        "K": np.linspace(max(0, values[feature] - 40), values[feature] + 40, 11),
    }
    rows = []
    for candidate in ranges[feature]:
        scenario = dict(values)
        scenario[feature] = float(candidate)
        prediction = predict_crop(model, scenario, top_n=1).iloc[0]
        confidence = prediction["confidence"]
        rows.append(
            {
                feature: float(candidate),
                "Leading crop": str(prediction["crop"]).title(),
                "Confidence": np.nan if pd.isna(confidence) else float(confidence),
            }
        )
    return pd.DataFrame(rows)


def yield_stress_table(model: object, base: dict[str, float]) -> pd.DataFrame:
    rows = []
    for rainfall_delta in [-20, -10, 0, 10, 20]:
        for temp_delta in [-1.0, 0.0, 1.0]:
            scenario = dict(base)
            scenario["average_rain_fall_mm_per_year"] = max(0.0, base["average_rain_fall_mm_per_year"] * (1 + rainfall_delta / 100))
            scenario["avg_temp"] = base["avg_temp"] + temp_delta
            frame = pd.DataFrame([scenario], columns=YIELD_FEATURES)
            rows.append(
                {
                    "Rainfall delta": f"{rainfall_delta:+d}%",
                    "Temp delta": f"{temp_delta:+.1f} C",
                    "Predicted yield": float(model.predict(frame)[0]),
                }
            )
    return pd.DataFrame(rows)


def render_sidebar(metadata: dict[str, object]) -> str:
    with st.sidebar:
        st.markdown("## AgriSense")
        st.caption("Operational crop intelligence")
        mode = st.radio("Visual mode", ["Day", "Night"], horizontal=True)
        st.markdown(
            """
            ### Scroll Map
            - [Experience](#experience)
            - [Crop Studio](#crop-studio)
            - [Field Planner](#field-planner)
            - [Yield Simulator](#yield-simulator)
            - [Crop Library](#crop-library)
            - [Model Monitor](#model-monitor)
            - [Data Hub](#data-hub)
            """
        )
        st.divider()
        st.markdown("### Deployment")
        st.write("Status: `Local`")
        st.write(f"Crop model: `{metadata['best_crop_model']}`")
        st.write(f"Yield model: `{metadata['best_yield_model']}`")
        st.divider()
        if st.button("Retrain Models", use_container_width=True):
            train_project()
            st.cache_resource.clear()
            st.rerun()
    return mode


def render_crop_cards(profiles: list[CropProfile], columns: int = 4) -> None:
    for index in range(0, len(profiles), columns):
        cols = st.columns(columns)
        for col, profile in zip(cols, profiles[index : index + columns]):
            with col:
                st.markdown(
                    f"""
                    <div class="crop-card">
                        <img src="{profile.image}" alt="{profile.name} field" />
                        <div class="crop-card-body">
                            <span class="chip">{profile.group}</span>
                            <span class="chip">{profile.water} water</span>
                            <h3>{profile.name}</h3>
                            <p>{profile.story}</p>
                            <p><b>Season:</b> {profile.season}<br/><b>Duration:</b> {profile.duration}<br/><b>Soil:</b> {profile.soil}</p>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_experience(state: dict[str, object], mode: str) -> None:
    metadata = state["metadata"]
    topbar(metadata, mode)
    st.markdown(
        """
        <section class="era-hero">
            <div class="hero-inner">
                <div class="hero-kicker">Field intelligence for crop decisions</div>
                <div class="hero-title">Grow with evidence</div>
                <p class="hero-copy">
                    Explore crop fit, scenario risk, model quality, and seasonal decisions through a visual,
                    interactive workspace built on the crop classification and wheat-yield models.
                </p>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    crop_metrics = state["crop_metrics"]
    yield_metrics = state["yield_metrics"]
    forecast = state["yield_forecast"]
    predicted_2025 = float(forecast["predicted_hg_per_ha_yield"].iloc[0])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Crop accuracy", percent(float(crop_metrics.iloc[0]["accuracy"])))
    m2.metric("Crop profiles", "22")
    m3.metric("Yield R2", percent(float(yield_metrics.iloc[0]["r2"])))
    m4.metric("2025 wheat outlook", f"{predicted_2025:,.0f} Hg/Ha")

    st.subheader("Featured Crop Collection")
    render_crop_cards(crop_profiles()[:4], columns=4)

    st.subheader("Interactive Product Areas")
    product_areas = pd.DataFrame(
        [
            {"Area": "Crop Studio", "Use": "Recommend crops and stress-test field variables", "Interaction": "Presets, sliders, ranking"},
            {"Area": "Field Planner", "Use": "Allocate parcels and compare crop readiness", "Interaction": "Field tiles, plan summary"},
            {"Area": "Yield Simulator", "Use": "Model weather and input scenarios", "Interaction": "Scenario presets, stress matrix"},
            {"Area": "Crop Library", "Use": "Explore crop fit by season and crop class", "Interaction": "Filters, visual cards"},
        ]
    )
    st.dataframe(product_areas, use_container_width=True, hide_index=True)


def crop_input_form(defaults: dict[str, float], key: str) -> tuple[dict[str, float], bool]:
    with st.form(key):
        st.subheader("Field Inputs")
        values: dict[str, float] = {}
        nutrient_a, nutrient_b, nutrient_c = st.columns(3)
        values["N"] = nutrient_a.number_input("Nitrogen", min_value=0.0, value=defaults["N"], step=1.0, key=f"{key}-n")
        values["P"] = nutrient_b.number_input("Phosphorus", min_value=0.0, value=defaults["P"], step=1.0, key=f"{key}-p")
        values["K"] = nutrient_c.number_input("Potassium", min_value=0.0, value=defaults["K"], step=1.0, key=f"{key}-k")
        climate_a, climate_b = st.columns(2)
        values["temperature"] = climate_a.slider("Temperature", 5.0, 45.0, defaults["temperature"], 0.5, key=f"{key}-temp")
        values["humidity"] = climate_b.slider("Humidity", 10.0, 100.0, defaults["humidity"], 0.5, key=f"{key}-humidity")
        soil_a, soil_b = st.columns(2)
        values["ph"] = soil_a.slider("Soil pH", 3.5, 9.5, defaults["ph"], 0.1, key=f"{key}-ph")
        values["rainfall"] = soil_b.number_input("Rainfall", min_value=0.0, value=defaults["rainfall"], step=1.0, key=f"{key}-rainfall")
        submitted = st.form_submit_button("Run Recommendation", use_container_width=True)
    return values, submitted


def render_crop_studio(state: dict[str, object], mode: str) -> None:
    topbar(state["metadata"], mode)
    page_header("Crop Studio", "Compare field scenarios, inspect confidence, and stress-test the selected recommendation.")

    preset_name = st.selectbox("Field scenario", list(crop_presets().keys()), index=0)
    values, submitted = crop_input_form(crop_presets()[preset_name], "crop-studio")

    recommendations = predict_crop(state["crop_model"], values, top_n=5)
    top_crop = str(recommendations.iloc[0]["crop"]).title()
    confidence = recommendations.iloc[0]["confidence"]
    confidence_value = 0.0 if pd.isna(confidence) else float(confidence)
    score, band = readiness_score(values)

    m1, m2, m3 = st.columns(3)
    m1.metric("Recommended Crop", top_crop)
    m2.metric("Confidence", "Ranked" if pd.isna(confidence) else percent(confidence_value))
    m3.metric("Field Readiness", f"{score}/100", band)
    st.progress(score / 100)

    if submitted:
        st.success(f"{top_crop} is now the active recommendation for this field scenario.")

    left, right = st.columns([0.9, 1.1])
    with left:
        display_recommendations = recommendations.assign(
            crop=recommendations["crop"].str.title(),
            confidence=recommendations["confidence"].map(lambda value: "Ranked" if pd.isna(value) else percent(float(value))),
        )
        st.subheader("Crop Ranking")
        st.dataframe(display_recommendations.rename(columns={"crop": "Crop", "confidence": "Confidence"}), use_container_width=True, hide_index=True)
        st.subheader("Action Checklist")
        st.dataframe(recommendation_actions(values, top_crop), use_container_width=True, hide_index=True)
    with right:
        st.subheader("Sensitivity Explorer")
        feature = st.selectbox("Stress variable", CROP_FEATURES, index=CROP_FEATURES.index("rainfall"))
        sensitivity = crop_sensitivity(state["crop_model"], values, feature)
        st.line_chart(sensitivity.set_index(feature)[["Confidence"]], height=260)
        table = sensitivity.copy()
        table["Confidence"] = table["Confidence"].map(lambda value: "Ranked" if pd.isna(value) else percent(float(value)))
        st.dataframe(table, use_container_width=True, hide_index=True)

    st.subheader("Model Drivers")
    st.bar_chart(state["crop_importance"].set_index("feature"), height=280)


def render_field_planner(state: dict[str, object], mode: str) -> None:
    topbar(state["metadata"], mode)
    page_header("Field Planner", "Build a crop plan across several parcels and see field-level readiness at a glance.")

    presets = crop_presets()
    field_rows = []
    cols = st.columns(3)
    selected_plots: list[tuple[str, str]] = []
    for index, field_name in enumerate(["North Field", "Canal Plot", "Orchard Edge", "Dry Ridge", "Trial Bed", "Market Block"]):
        with cols[index % 3]:
            st.markdown('<div class="field-tile">', unsafe_allow_html=True)
            scenario_name = st.selectbox(field_name, list(presets.keys()), key=f"field-{index}")
            acres = st.slider("Acres", 1, 25, 6 + index, key=f"acres-{index}")
            values = presets[scenario_name]
            recommendation = predict_crop(state["crop_model"], values, top_n=1).iloc[0]
            score, band = readiness_score(values)
            crop = str(recommendation["crop"]).title()
            st.metric(crop, f"{score}/100", band)
            st.progress(score / 100)
            st.markdown("</div>", unsafe_allow_html=True)
            selected_plots.append((field_name, crop))
            field_rows.append({"Field": field_name, "Scenario": scenario_name, "Acres": acres, "Crop": crop, "Readiness": score, "Band": band})

    plan = pd.DataFrame(field_rows)
    st.subheader("Acreage Plan")
    st.dataframe(plan, use_container_width=True, hide_index=True)

    crop_mix = plan.groupby("Crop", as_index=False)["Acres"].sum().sort_values("Acres", ascending=False)
    c1, c2 = st.columns([0.9, 1.1])
    with c1:
        st.subheader("Crop Mix")
        st.bar_chart(crop_mix.set_index("Crop"))
    with c2:
        st.subheader("Planning Notes")
        notes = pd.DataFrame(
            [
                {"Check": "Dominant crop", "Result": str(crop_mix.iloc[0]["Crop"])},
                {"Check": "Average readiness", "Result": f"{plan['Readiness'].mean():.0f}/100"},
                {"Check": "Fields needing review", "Result": str(int((plan["Readiness"] < 68).sum()))},
                {"Check": "Total acreage", "Result": str(int(plan["Acres"].sum()))},
            ]
        )
        st.dataframe(notes, use_container_width=True, hide_index=True)

    st.download_button("Download Field Plan", plan.to_csv(index=False).encode("utf-8"), "agrisense_field_plan.csv", "text/csv", use_container_width=True)


def render_yield_simulator(state: dict[str, object], mode: str) -> None:
    topbar(state["metadata"], mode)
    page_header("Yield Simulator", "Run weather and input scenarios against the trained Pakistan wheat yield model.")

    forecast = state["yield_forecast"].copy()
    base = forecast[YIELD_FEATURES].iloc[0].to_dict()
    presets = yield_presets(base)
    preset_name = st.selectbox("Yield scenario", list(presets.keys()), index=0)
    selected = presets[preset_name]
    baseline_prediction = float(forecast["predicted_hg_per_ha_yield"].iloc[0])

    left, right = st.columns([0.9, 1.1])
    with left:
        with st.form("yield-simulator-form"):
            st.subheader("Scenario Inputs")
            edited: dict[str, float] = {}
            edited["Year"] = st.number_input("Year", value=float(selected["Year"]), disabled=True)
            edited["average_rain_fall_mm_per_year"] = st.number_input("Average rainfall", min_value=0.0, value=float(selected["average_rain_fall_mm_per_year"]), step=1.0)
            edited["pesticides_tonnes"] = st.number_input("Pesticides", min_value=0.0, value=float(selected["pesticides_tonnes"]), step=100.0)
            edited["avg_temp"] = st.number_input("Average temperature", value=float(selected["avg_temp"]), step=0.1)
            submitted = st.form_submit_button("Run Forecast", use_container_width=True)

    scenario = pd.DataFrame([edited], columns=YIELD_FEATURES)
    scenario_prediction = float(state["yield_model"].predict(scenario)[0])
    delta = scenario_prediction - baseline_prediction

    with right:
        m1, m2, m3 = st.columns(3)
        m1.metric("Baseline", f"{baseline_prediction:,.0f} Hg/Ha")
        m2.metric("Scenario", f"{scenario_prediction:,.0f} Hg/Ha", f"{delta:+,.0f}")
        m3.metric("Scenario Type", preset_name)
        if submitted:
            st.success("Scenario forecast refreshed.")
        scenario_display = scenario.copy()
        scenario_display["predicted_hg_per_ha_yield"] = scenario_prediction
        st.dataframe(scenario_display, use_container_width=True, hide_index=True)

    left, right = st.columns([1.08, 0.92])
    with left:
        st.subheader("Stress Test Matrix")
        stress = yield_stress_table(state["yield_model"], edited)
        heat = stress.pivot(index="Temp delta", columns="Rainfall delta", values="Predicted yield")
        st.dataframe(heat.style.format("{:,.0f}"), use_container_width=True)
    with right:
        st.subheader("Yield Drivers")
        st.bar_chart(state["yield_importance"].set_index("feature"), height=300)

    st.download_button("Download Scenario CSV", scenario_display.to_csv(index=False).encode("utf-8"), "agrisense_yield_scenario.csv", "text/csv", use_container_width=True)


def render_crop_library(state: dict[str, object], mode: str) -> None:
    topbar(state["metadata"], mode)
    page_header("Crop Library", "Browse crop profiles by group, season, and water requirement.")

    profiles = crop_profiles()
    groups = ["All"] + sorted({profile.group for profile in profiles})
    seasons = ["All"] + sorted({profile.season for profile in profiles})
    water_levels = ["All"] + sorted({profile.water for profile in profiles})
    c1, c2, c3 = st.columns(3)
    group = c1.selectbox("Group", groups)
    season = c2.selectbox("Season", seasons)
    water = c3.selectbox("Water", water_levels)

    filtered = [
        profile
        for profile in profiles
        if (group == "All" or profile.group == group)
        and (season == "All" or profile.season == season)
        and (water == "All" or profile.water == water)
    ]
    render_crop_cards(filtered, columns=4)

    st.subheader("Crop Listing")
    listing = pd.DataFrame([profile.__dict__ for profile in filtered]).drop(columns=["image"])
    st.dataframe(listing, use_container_width=True, hide_index=True)


def render_model_monitor(state: dict[str, object], mode: str) -> None:
    topbar(state["metadata"], mode)
    page_header("Model Monitor", "Review model quality, cross-validation stability, and explainability outputs.")

    crop_metrics = state["crop_metrics"]
    yield_metrics = state["yield_metrics"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Best Classifier", str(crop_metrics.iloc[0]["model"]))
    m2.metric("CV Accuracy", percent(float(crop_metrics.iloc[0]["cv_accuracy_mean"])))
    m3.metric("Best Regressor", str(yield_metrics.iloc[0]["model"]))
    m4.metric("CV R2", percent(float(yield_metrics.iloc[0]["cv_r2_mean"])))

    st.subheader("Crop Model Comparison")
    st.dataframe(format_metric_frame(crop_metrics, ["accuracy", "precision", "recall", "f1", "cv_accuracy_mean"]), use_container_width=True, hide_index=True)
    st.bar_chart(crop_metrics.set_index("model")[["accuracy", "f1"]], height=260)

    st.subheader("Yield Model Comparison")
    st.dataframe(format_metric_frame(yield_metrics, ["r2", "cv_r2_mean"]), use_container_width=True, hide_index=True)
    st.bar_chart(yield_metrics.set_index("model")[["r2"]], height=260)


def render_data_hub(state: dict[str, object], mode: str) -> None:
    metadata = state["metadata"]
    topbar(metadata, mode)
    page_header("Data Hub", "Inspect data sources, generated artifacts, and completed yield series.")
    st.markdown(f'<span class="chip">Crop data: {metadata["crop_data_source"]}</span><span class="chip">Yield data: {metadata["yield_data_source"]}</span>', unsafe_allow_html=True)

    completed = state["completed_yield"].copy()
    st.subheader("MICE-Completed Yield Series")
    st.line_chart(completed[["Year", "hg_per_ha_yield"]].set_index("Year"), height=260)
    st.dataframe(completed, use_container_width=True, hide_index=True)

    st.subheader("Generated Artifacts")
    artifact_rows = [{"Name": key, "Path": value} for key, value in metadata["artifacts"].items()]
    st.dataframe(pd.DataFrame(artifact_rows), use_container_width=True, hide_index=True)

    archive = pd.concat([state["crop_metrics"].assign(kind="crop_metrics"), state["yield_metrics"].assign(kind="yield_metrics")], ignore_index=True, sort=False)
    st.download_button("Download Model Metrics", archive.to_csv(index=False).encode("utf-8"), "agrisense_model_metrics.csv", "text/csv", use_container_width=True)


state = load_or_train()
visual_mode = render_sidebar(state["metadata"])
apply_theme(visual_mode)
topbar(state["metadata"], visual_mode)
scroll_nav()

st.markdown('<div id="experience" class="scroll-section"></div>', unsafe_allow_html=True)
render_experience(state, visual_mode)

st.markdown('<div id="crop-studio" class="scroll-section"></div>', unsafe_allow_html=True)
render_crop_studio(state, visual_mode)

st.markdown('<div id="field-planner" class="scroll-section"></div>', unsafe_allow_html=True)
render_field_planner(state, visual_mode)

st.markdown('<div id="yield-simulator" class="scroll-section"></div>', unsafe_allow_html=True)
render_yield_simulator(state, visual_mode)

st.markdown('<div id="crop-library" class="scroll-section"></div>', unsafe_allow_html=True)
render_crop_library(state, visual_mode)

st.markdown('<div id="model-monitor" class="scroll-section"></div>', unsafe_allow_html=True)
render_model_monitor(state, visual_mode)

st.markdown('<div id="data-hub" class="scroll-section"></div>', unsafe_allow_html=True)
render_data_hub(state, visual_mode)
