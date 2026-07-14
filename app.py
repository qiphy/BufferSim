import os
import re
import time

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import InferenceClient


FIELD_AREA_M2 = 7_500.0
ROWS, COLUMNS = 18, 14
CELL_AREA_M2 = FIELD_AREA_M2 / (ROWS * COLUMNS)
WEST, EAST = 101.527045, 101.528057
SOUTH, NORTH = 3.029751, 3.031057

# Load local development credentials before reading environment variables.
load_dotenv()


st.set_page_config(page_title="Flood Heatmap Simulator", layout="wide")
st.title("🌊 SMART Eco-Field: Flood Heatmap Simulator")
st.markdown("A cell-based rainfall, drainage, infiltration, and downhill-flow model.")


@st.cache_data
def simulate_flood(rainfall_mm, duration_hrs, pump_capacity_pct, infiltration_mm_hr):
    """Return surface-water depth frames and a volume-conserving water balance."""
    dt_seconds = 60
    steps = duration_hrs * 60
    rain_depth_per_step = (rainfall_mm / 1000) / steps
    infiltration_depth_per_step = (infiltration_mm_hr / 1000) * dt_seconds / 3600
    rain_volume_per_step = rain_depth_per_step * FIELD_AREA_M2
    pump_volume_per_step = rain_volume_per_step * pump_capacity_pct

    # A gentle north-east-to-south-west slope with a shallow low basin.
    y, x = np.meshgrid(np.linspace(0, 1, ROWS), np.linspace(0, 1, COLUMNS), indexing="ij")
    terrain = 0.45 * (x + y) + 0.08 * ((x - 0.22) ** 2 + (y - 0.22) ** 2)
    water = np.zeros((ROWS, COLUMNS), dtype=float)
    frames = np.zeros((steps, ROWS, COLUMNS), dtype=np.float32)
    stored = np.zeros(steps)
    pumped = np.zeros(steps)
    infiltrated = np.zeros(steps)
    rainfall_in = np.zeros(steps)

    # Pump intakes are deliberately at the low, south-west end of the field.
    pump_mask = np.zeros((ROWS, COLUMNS), dtype=bool)
    pump_mask[:3, :3] = True

    for step in range(steps):
        water += rain_depth_per_step
        rainfall_in[step] = rain_volume_per_step + (rainfall_in[step - 1] if step else 0)

        infiltrated_depth = np.minimum(water, infiltration_depth_per_step)
        water -= infiltrated_depth
        infiltrated[step] = infiltrated_depth.sum() * CELL_AREA_M2 + (infiltrated[step - 1] if step else 0)

        # Route water by hydraulic head. Each cell sends at most 35% of its
        # current depth per minute, which keeps the explicit solver stable.
        head = terrain + water
        change = np.zeros_like(water)
        for row in range(ROWS):
            for column in range(COLUMNS):
                neighbours = []
                for d_row, d_column in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    n_row, n_column = row + d_row, column + d_column
                    if 0 <= n_row < ROWS and 0 <= n_column < COLUMNS:
                        head_drop = head[row, column] - head[n_row, n_column]
                        if head_drop > 0:
                            neighbours.append((n_row, n_column, head_drop))

                if neighbours and water[row, column] > 0:
                    total_drop = sum(drop for _, _, drop in neighbours)
                    transferred = min(water[row, column] * 0.35, total_drop * 0.10)
                    change[row, column] -= transferred
                    for n_row, n_column, head_drop in neighbours:
                        change[n_row, n_column] += transferred * head_drop / total_drop

        water = np.maximum(0, water + change)

        intake_volume = water[pump_mask].sum() * CELL_AREA_M2
        removed_volume = min(intake_volume, pump_volume_per_step)
        if intake_volume > 0:
            water[pump_mask] *= 1 - removed_volume / intake_volume
        pumped[step] = removed_volume + (pumped[step - 1] if step else 0)

        frames[step] = water
        stored[step] = water.sum() * CELL_AREA_M2

    balance_error = rainfall_in - pumped - infiltrated - stored
    return frames, terrain, stored, pumped, infiltrated, rainfall_in, balance_error


@st.cache_data
def grid_geometry():
    cells = []
    for row in range(ROWS):
        for column in range(COLUMNS):
            lon0 = WEST + (EAST - WEST) * column / COLUMNS
            lon1 = WEST + (EAST - WEST) * (column + 1) / COLUMNS
            lat0 = SOUTH + (NORTH - SOUTH) * row / ROWS
            lat1 = SOUTH + (NORTH - SOUTH) * (row + 1) / ROWS
            cells.append([[lon0, lat0], [lon1, lat0], [lon1, lat1], [lon0, lat1]])
    return cells


def render_map(depth_grid):
    field_outline = pd.DataFrame([{
        "coordinates": [[WEST, NORTH], [EAST, NORTH], [EAST, SOUTH], [WEST, SOUTH], [WEST, NORTH]]
    }])
    layers = [pdk.Layer(
        "PolygonLayer",
        field_outline,
        get_polygon="coordinates",
        get_fill_color=[60, 125, 70, 115],
        get_line_color=[35, 90, 40, 230],
    )]

    cells = []
    for coordinates, depth in zip(grid_geometry(), depth_grid.ravel()):
        if depth < 0.001:
            continue
        # Light blue is shallow water; dark blue marks deeper pooled water.
        intensity = min(depth / 0.25, 1.0)
        cells.append({
            "coordinates": coordinates,
            "depth": float(depth),
            "depth_label": f"{depth:.3f} m",
            "color": [15, int(155 - 55 * intensity), int(235 - 35 * intensity), int(95 + 135 * intensity)],
        })

    if cells:
        layers.append(pdk.Layer(
            "PolygonLayer",
            pd.DataFrame(cells),
            get_polygon="coordinates",
            get_elevation="depth",
            elevation_scale=18,
            extruded=True,
            get_fill_color="color",
            get_line_color=[20, 90, 160, 100],
            pickable=True,
        ))

    return pdk.Deck(
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=3.030394, longitude=101.527550, zoom=16.5, pitch=55, bearing=15),
        tooltip={"text": "Water depth: {depth_label}"},
    )


def calculate_risk_inputs(frames, stored, pumped, rainfall_in, duration_hrs):
    """Prepare transparent inputs for the AI risk assessment."""
    peak_step = int(np.argmax(stored))
    peak_depth = float(frames[peak_step].max())
    flooded_coverage = float((frames[peak_step] >= 0.02).mean() * 100)
    pump_deficit = max(0.0, 1 - pumped[-1] / rainfall_in[-1]) if rainfall_in[-1] else 0.0
    score = round(min(100, (
        45 * min(peak_depth / 0.30, 1)
        + 0.25 * flooded_coverage
        + 20 * pump_deficit
        + 10 * min(duration_hrs / 8, 1)
    )))
    return {
        "peak_depth": peak_depth,
        "peak_storage": float(stored[peak_step]),
        "flooded_coverage": flooded_coverage,
        "pumped": float(pumped[-1]),
        "rainfall_in": float(rainfall_in[-1]),
        "pump_deficit": pump_deficit * 100,
        "risk_score": score,
    }


def get_hf_token():
    try:
        return st.secrets.get("HF_TOKEN") or os.getenv("HF_TOKEN")
    except Exception:
        return os.getenv("HF_TOKEN")


def request_huggingface_assessment(metrics):
    """Ask a Hugging Face chat model to turn calculated metrics into an alert."""
    token = get_hf_token()
    if not token:
        raise RuntimeError("No HF_TOKEN is configured.")

    formula = (
        "risk score = round(min(100, 45*min(peak_depth/0.30,1) + "
        "0.25*flooded_coverage_pct + 20*(pump_deficit_pct/100) + "
        "10*min(duration_hours/8,1)))"
    )
    prompt = f"""Assess this conceptual flood simulation. Apply the formula exactly and do not invent data.
Formula: {formula}
Peak depth: {metrics['peak_depth']:.3f} m
Peak stored water: {metrics['peak_storage']:.1f} m³
Flooded coverage: {metrics['flooded_coverage']:.1f}%
Pump deficit: {metrics['pump_deficit']:.1f}%
Storm duration: {metrics['duration_hours']} hours

Reply using exactly two lines:
RISK_SCORE: integer from 0 to 100
NOTIFICATION: one concise operational warning of at most 25 words"""
    client = InferenceClient(api_key=token)
    model = os.getenv("HF_MODEL", "google/gemma-4-31B-it")
    response = client.chat_completion(
        messages=[
            {"role": "system", "content": "You are a precise flood-risk decision-support assistant."},
            {"role": "user", "content": prompt},
        ],
        model=model,
        max_tokens=100,
        temperature=0.1,
    )
    content = response.choices[0].message.content.strip()
    score_match = re.search(r"RISK_SCORE:\s*(\d{1,3})", content)
    note_match = re.search(r"NOTIFICATION:\s*(.+)", content)
    if not score_match or not note_match:
        raise RuntimeError("The Hugging Face model returned an unexpected response format.")

    return min(100, int(score_match.group(1))), note_match.group(1).strip()


with st.sidebar:
    st.header("🎛️ Storm Parameters")
    rainfall = st.slider("Total rainfall (mm)", 50, 600, 300, step=10)
    storm_duration = st.slider("Storm duration (hours)", 2, 8, 4)
    pump_capacity = st.slider("Pump capacity (% of rainfall inflow)", 0, 100, 50, step=5) / 100
    infiltration = st.slider("Infiltration (mm/hour)", 0, 30, 8, step=1)
    st.caption("Water flows toward the south-west pump intake. Values are conceptual and can be calibrated with site data.")

frames, terrain, stored, pumped, infiltrated, rainfall_in, balance_error = simulate_flood(
    rainfall, storm_duration, pump_capacity, infiltration
)

peak_step = int(np.argmax(stored))
ai_metrics = calculate_risk_inputs(frames, stored, pumped, rainfall_in, storm_duration)
ai_metrics["duration_hours"] = storm_duration
summary = st.columns(4)
summary[0].metric("Peak water depth", f"{frames[peak_step].max():.3f} m")
summary[1].metric("Peak stored water", f"{stored[peak_step]:,.0f} m³")
summary[2].metric("Pumped", f"{pumped[-1]:,.0f} m³")
summary[3].metric("Mass-balance error", f"{balance_error[-1]:.3f} m³")

st.subheader("🤖 Hugging Face AI flood assessment")
st.caption("The model applies a visible risk-score formula to the simulation results, then produces an operational notification.")
if st.button("Calculate AI risk notification"):
    if not get_hf_token():
        st.info("Add `HF_TOKEN` to `.streamlit/secrets.toml` or your environment, then run the AI assessment.")
    else:
        with st.spinner("Hugging Face AI is calculating the flood-risk notification..."):
            try:
                ai_score, ai_notification = request_huggingface_assessment(ai_metrics)
                st.success(f"AI risk score: {ai_score}/100")
                st.warning(f"⚠️ {ai_notification}")
            except Exception as error:
                st.error(f"Hugging Face assessment could not be completed: {error}")

st.subheader("▶️ Flood Playback")
play_column, status_column = st.columns([1, 3])
status = status_column.empty()
map_container = st.empty()
map_container.pydeck_chart(render_map(frames[0]))

if play_column.button("Play simulation", type="primary"):
    for step in range(0, len(frames), 3):
        minute = step
        status.markdown(
            f"**Time:** {minute // 60:02d}:{minute % 60:02d}  |  "
            f"**Stored water:** {stored[step]:,.0f} m³  |  "
            f"**Deepest cell:** {frames[step].max():.3f} m"
        )
        map_container.pydeck_chart(render_map(frames[step]))
        time.sleep(0.05)
    st.success("Simulation complete.")

st.subheader("Water balance")
chart = pd.DataFrame({
    "Rainfall in": rainfall_in,
    "Pumped": pumped,
    "Infiltrated": infiltrated,
    "Stored on field": stored,
})
st.line_chart(chart)
