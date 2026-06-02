import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pydeck as pdk
import os

# ==========================================
# 1. CONFIG
# ==========================================
st.set_page_config(page_title="Flood Particle Simulator", layout="wide")

st.title("🌊 Flood Particle Simulation Model")
st.markdown("Rainfall → Flow → Particle-like flood propagation over terrain")

# ==========================================
# 2. HYDROLOGY ENGINE
# ==========================================
class FloodGridEngine:
    def __init__(self, n=60):
        x = np.linspace(0, 10, n)
        y = np.linspace(0, 10, n)
        X, Y = np.meshgrid(x, y)

        # bowl-shaped terrain (stable + smooth)
        self.elevation = 0.08 * ((X - 5)**2 + (Y - 5)**2)

        self.n = n
        self.water = np.zeros((n, n))

        # spatial infiltration 
        self.infiltration = np.full((n, n), 0.02)

    def add_rainfall(self, mm_hr):
        rain_factor = 0.01
        self.water += rain_factor * mm_hr

    def apply_infiltration(self, rate):
        # spatial infiltration
        self.water = np.maximum(
            0,
            self.water - rate * self.infiltration
        )

    def step_flow(self, flow_speed):
        # Total hydraulic head (Ground Elevation Contour + Water Depth)
        h = self.elevation + self.water
        w = self.water.copy()
        new_w = np.zeros_like(w)

        # Conceptual roughness coefficient (e.g., concrete vs grass)
        manning_n = 0.035  

        for i in range(1, self.n - 1):
            for j in range(1, self.n - 1):
                current_water = w[i, j]
                if current_water <= 0.001:
                    continue  # Skip dry cells to optimize performance

                current_h = h[i, j]
                neighbors = [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]
                
                lower_neighbors = []
                flow_factors = []

                for ni, nj in neighbors:
                    # Elevation drop acting as the slope driver (S)
                    elevation_drop = current_h - h[ni, nj]
                    
                    if elevation_drop > 0:
                        # 1. Calculate Slope Factor (S^0.5)
                        slope_factor = np.sqrt(elevation_drop)
                        
                        # 2. Calculate Depth Factor (d^2/3) using local water depth
                        depth_factor = current_water ** (2/3)
                        
                        # 3. Combine using Manning's approach for localized velocity
                        local_velocity = (1.0 / manning_n) * depth_factor * slope_factor
                        
                        lower_neighbors.append((ni, nj))
                        flow_factors.append(local_velocity)

                if len(lower_neighbors) == 0:
                    new_w[i, j] += current_water
                    continue

                # Normalize the directional flow factors
                flow_factors = np.array(flow_factors)
                total_factor = flow_factors.sum()
                
                if total_factor == 0:
                    new_w[i, j] += current_water
                    continue
                    
                weights = flow_factors / total_factor

                # Outflow scales dynamically with flow speed and localized physical factors
                outflow = current_water * flow_speed * (total_factor * 0.1)
                outflow = min(outflow, current_water)  # Conservation of mass clamp

                for k, (ni, nj) in enumerate(lower_neighbors):
                    new_w[ni, nj] += outflow * weights[k]

                new_w[i, j] += current_water - outflow

        self.water = new_w
        self.water *= 0.995  # Global drainage coefficient

    def get_state(self):
        return self.water


# ==========================================
# 3. TIME CONTROL & DYNAMIC SIDEBAR SLIDERS
# ==========================================
with st.sidebar:
    st.header("🎛️ Storm & Ground Settings")
    
    # --- WEATHER CONFIG CONFIGURATIONS ---
    st.subheader("🌧️ Rainfall Settings")
    total_steps = st.slider("Storm Duration (Minutes)", 30, 120, 60, step=10)
    peak_rain = st.slider("Peak Rain Intensity (mm/hr)", 50, 250, 120, step=10)
    peak_time = st.slider("Minute of Peak Intensity", 10, total_steps - 10, int(total_steps / 3))

    # --- ENVIRONMENT CONFIGURATIONS ---
    st.subheader("🌱 Ground Conditions")
    soil_absorption = st.slider(
        "Soil Infiltration Rate", 
        0.005, 0.08, 0.02, step=0.005, 
        help="Higher values mean the ground absorbs water faster (concrete vs sand)."
    )
    water_velocity = st.slider(
        "Water Flow Speed Factor", 
        1.0, 6.0, 3.0, step=0.5, 
        help="Controls how quickly gravity drags water downhill."
    )

    st.markdown("---")
    st.header("⏳ Timeline Control")
    
    # Builds the weather graph values dynamically using your slider inputs
    rain_series = np.concatenate([
        np.linspace(0, peak_rain, peak_time),
        np.linspace(peak_rain, 20, total_steps - peak_time)
    ])

    # Dynamic Simulation Cache runner (Re-fires automatically when settings change)
    @st.cache_data
    def run_dynamic_simulation(steps, series, absorption, speed):
        engine = FloodGridEngine(n=60)
        frames = []
        terrain_elevation = engine.elevation.copy()
        scaled_speed = speed * 0.02  
        
        for i in range(steps):
            engine.add_rainfall(series[i])
            engine.step_flow(scaled_speed)
            engine.apply_infiltration(absorption)
            frames.append(engine.get_state().copy())
            
        return frames, series, terrain_elevation

    # Unpack all three variables safely in your sidebar thread
    frames, rain_trace, terrain_elevation = run_dynamic_simulation(
        total_steps, rain_series, soil_absorption, water_velocity
    )
    
    # Interactive timestamp selector
    t = st.slider("Select Time Step (Minutes into Storm)", 0, len(frames) - 1, 0)
    
    # Live frame variables processing
    water_t = frames[t]
    max_depth = np.max(water_t)
    total_water = np.sum(water_t)
    n = water_t.shape[0]
    
    center_min, center_max = int(n * 0.35), int(n * 0.65)
    center_zone = water_t[center_min:center_max, center_min:center_max]
    basin_water = np.sum(center_zone)
    basin_percentage = (basin_water / total_water * 100) if total_water > 0 else 0

    st.markdown("---")
    st.header("🗺️ The Neighborhood Story")
    st.info(f"""
    * **The Shape of the Land:** We have modeled this neighborhood like a giant **kitchen bowl**.
    * **The Speed Bumps:** The rows of dots are water getting temporarily trapped behind tiny ridges—like **puddles behind street speed humps**.
    * **The Flood Pocket:** Right now, **{basin_percentage:.0f}%** of all runoff has pooled in the center bowl.
    * **Deepest Point:** Maximum depth is **{max_depth:.2f} meters** in the middle of town.
    """)
    
# ==========================================
# 4. HUMAN-READABLE DASHBOARD (MAIN DISPLAY TOP)
# ==========================================
st.subheader("📊 Live Flood Status")

flood_cells = np.sum(water_t > 0.01)
total_cells = n * n
flooded_percentage = (flood_cells / total_cells) * 100

col1, col2, col3 = st.columns(3)
col1.metric("Water on Streets (Volume Index)", f"{total_water:.1f}")
col2.metric("Worst-Case Depth", f"{max_depth:.2f} meters")
col3.metric("Neighborhood Area Flooded", f"{flooded_percentage:.1f} %")

if basin_percentage > 50:
    st.error(f"🚨 **Emergency Status:** Over half of the storm's runoff ({basin_percentage:.1f}%) has accumulated in the center basin. Streets are heavily inundated, indicating a total drainage failure or blocked discharge pumps.")
elif total_water > 0:
    st.warning("⚠️ **Active Runoff:** Rain is falling. Water is actively sheeting off the higher perimeter roads and flowing down toward residential areas.")
else:
    st.success("🟢 **Dry Conditions:** No significant surface flooding or standing water detected on the roadways.")

# ==========================================
# 5. RAINFALL PLOT (TIMELINE)
# ==========================================
st.subheader("🌧️ Storm Timeline (Rainfall Intensity)")

fig, ax = plt.subplots(figsize=(12, 2.5))
ax.plot(rain_trace, color="#1f77b4", linewidth=2.5, label="Rainfall Amount")
ax.axvline(t, color="red", linestyle="--", linewidth=2, label="You Are Here")
ax.set_ylabel("Rain Intensity (mm/hr)")
ax.set_xlabel("Minutes into Storm")
ax.legend(loc="upper right")
ax.grid(True, linestyle=":", alpha=0.6)
st.pyplot(fig)

# ==========================================
# 6. 🌊 3D FLOOD DEPTH & CONTOUR VIEW
# ==========================================
st.subheader("🌊 Interactive 3D Flood Depth & Contour View")

center_lat = 3.029628
center_lon = 101.528775
geo_step = 0.00008 

lon_array = center_lon + (np.arange(n) - n // 2) * geo_step
lat_array = center_lat + (np.arange(n) - n // 2) * geo_step

yy, xx = np.meshgrid(lat_array, lon_array, indexing='ij')

elevation_flat = terrain_elevation.flatten()
lon_flat = xx.flatten()
lat_flat = yy.flatten()
water_flat = water_t.flatten()

df = pd.DataFrame({
    "lon": lon_flat,
    "lat": lat_flat,
    "water": water_flat,
    "elevation": elevation_flat
})

df_water = df[df["water"] > 0.02].copy()

def assign_depth_color(row):
    w = row["water"]
    if w < 0.05:
        return [30, 144, 255, 140]   
    elif w < 0.2:
        return [0, 90, 220, 180]      
    elif w < 0.5:
        return [255, 140, 0, 210]     
    else:
        return [180, 0, 0, 230]       

if not df_water.empty:
    df_water["color"] = df_water.apply(assign_depth_color, axis=1)
else:
    df_water["color"] = [0, 0, 0, 0]

water_3d_layer = pdk.Layer(
    "GridCellLayer",
    df_water,
    get_position=["lon", "lat"],
    get_elevation="water",
    elevation_scale=30,           
    cell_size=12,                 
    get_fill_color="color",
    pickable=True,
    extruded=True,                
)

st.pydeck_chart(pdk.Deck(
    map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
    layers=[water_3d_layer],
    initial_view_state=pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=15.8,
        pitch=45,                 
        bearing=0               
    ),
    tooltip={"text": "Water Depth: {water}m\nTerrain Elevation: {elevation}m"}
))

# ==========================================
# 7. 🗺️ MAP COLOR LEGEND & VISUAL GUIDE
# ==========================================
st.markdown("### 🗺️ Map Color Legend & Hazard Guide")
col_blue, col_red = st.columns(2)

with col_blue:
    st.info("""
    🔵 **The Blue Patterns = Shallow & Moving Water**
    * **Light Blue Fringes:** Shallow rainwater sheeting across roads (roughly ankle-deep).
    * **Rows of Blue Dots:** Isolated puddles getting blocked by minor layout elevations.
    """)

with col_red:
    st.error("""
    🔴 **The Dark Red Clusters = Deep & Dangerous Pooling**
    * **The Central Red Core:** Deep, trapped water accumulating at the lowest topographic location. 
    * **Real-World Meaning:** Heavy road inundation where vehicle operation is strictly compromised.
    """)

# ==========================================
# 8. 🤖 AI EMERGENCY ADVISORY ENGINE (SECURE INGESTION)
# ==========================================
st.markdown("---")
st.subheader("🤖 AI Real-Time Command Advisory")

# Ingest credentials securely from standard cloud environment configurations or user inputs
ROUTER_TOKEN = os.environ.get("ROUTER_TOKEN") or st.sidebar.text_input("Enter AI Router Token Key", type="password")

if not ROUTER_TOKEN:
    st.info("💡 **Inference Engine Standby:** Provide your Access Token in the sidebar input box to activate live tactical bulletins.")
else:
    from openai import OpenAI

    try:
        client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=ROUTER_TOKEN,
        )
        
        storm_progress_percentage = (t / total_steps) * 100
        
        emergency_prompt = f"""You are an AI Emergency Management Commander responding to a simulated flash flood event in Taman Sri Muda.
        
        CRITICAL TIME CONTEXT:
        - Current Simulation Time: Minute {t} out of a total {total_steps} minute storm duration ({storm_progress_percentage:.0f}% progressed).
        - Storm Peak Intensity: {peak_rain} mm/hr scheduled at Minute {peak_time}.
        
        CURRENT HYDROMETRIC SNAPSHOT:
        - Total Surface Water Volume Index: {total_water:.2f}
        - Worst-Case Peak Depth: {max_depth:.2f} meters
        - Basin Pooling Concentration: {basin_percentage:.1f}% concentrated in the residential center bowl.
        - Geographic Area Inundated: {flooded_percentage:.1f}%
        
        TACTICAL EVALUATION RULE:
        - If the simulation time is under 5-10 minutes, water is just beginning to sheet off surfaces. Stand down from major evacuation orders unless worst-case depth already exceeds safe thresholds. Do not overreact to initial runoff.
        
        Provide a professional, clear, 3-bullet-point tactical advisory based EXACTLY on this timeline and data:
        1. Resident status directive (e.g., Monitoring/Standby vs Shelter-in-Place vs Evacuate).
        2. Emergency rescue deployment status (BOMBA / APM mobilization status).
        3. Infrastructure drainage pump directive.
        
        Keep the tone professional, direct, and realistic. Do not mention code, matrices, or simulation parameters."""

        with st.spinner("🧠 Querying emergency inference engine..."):
            completion = client.chat.completions.create(
                model="google/gemma-4-31B-it:novita", 
                messages=[
                    {"role": "system", "content": "You are a professional crisis management AI specializing in local hydrological emergencies."},
                    {"role": "user", "content": emergency_prompt}
                ],
                temperature=0.4
            )
            
            st.markdown("### 📢 Live Tactical Bulletin")
            st.write(completion.choices[0].message.content)

    except Exception as e:
        st.warning(f"📡 **Connection Issue:** Unable to reach the inference endpoint. Details: {e}")