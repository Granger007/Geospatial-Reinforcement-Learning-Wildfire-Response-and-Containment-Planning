import streamlit as st
import numpy as np
import pandas as pd

# Import your custom Python files (classes)
from simulator import FireSimulator, UNBURNED, BURNING, BURNED_OUT, FIREBREAK, STRUCTURE
from agents import PerceptionAgent, PredictionAgent, PlanningAgent

# --- 1. CONFIGURATION ---

MAP_WIDTH = 50
MAP_HEIGHT = 50

# Extra code-only constant for "unknown" cells in the Perception view
UNKNOWN = -1

COLOR_MAP = {
    UNBURNED: (0, 102, 0),      # Dark Green (forest)
    BURNING: (255, 0, 0),       # Bright Red
    BURNED_OUT: (40, 40, 40),   # Dark Gray (ash)
    FIREBREAK: (139, 69, 19),   # Brown (dirt)
    STRUCTURE: (0, 0, 200),     # Blue (house/town)
    UNKNOWN: (160, 160, 160)    # Light gray (smoke / unknown)
}

# --- 2. MAP HELPERS ---------------------------------------------------------

def create_initial_map():
    """Deterministic center fire + structure (for nice demos)."""
    fuel_map = np.full((MAP_HEIGHT, MAP_WIDTH), UNBURNED, dtype=int)

    mid_y = MAP_HEIGHT // 2
    mid_x = MAP_WIDTH // 2
    fuel_map[mid_y, mid_x] = BURNING
    fuel_map[mid_y + 5, mid_x + 5] = STRUCTURE
    return fuel_map

def create_random_map(rng):
    """Random ignition + random structure for batch evaluation / new sims."""
    grid = np.full((MAP_HEIGHT, MAP_WIDTH), UNBURNED, dtype=int)

    # random ignition
    fy = rng.integers(0, MAP_HEIGHT)
    fx = rng.integers(0, MAP_WIDTH)
    grid[fy, fx] = BURNING

    # random structure at different cell
    while True:
        sy = rng.integers(0, MAP_HEIGHT)
        sx = rng.integers(0, MAP_WIDTH)
        if (sy, sx) != (fy, fx):
            grid[sy, sx] = STRUCTURE
            break

    return grid

def convert_grid_to_image(grid, markers=None):
    """
    Convert 2D grid (or float grid with NaNs) into enlarged RGB image.

    markers: list of (x, y) cells to outline in yellow (e.g., last firebreak).
    """
    if markers is None:
        markers = []

    working = grid.astype(float)
    nan_mask = np.isnan(working)
    working[nan_mask] = UNKNOWN
    working = working.astype(int)

    h, w = working.shape
    base_img = np.zeros((h, w, 3), dtype=np.uint8)

    for state_number, color_rgb in COLOR_MAP.items():
        mask = (working == state_number)
        base_img[mask] = color_rgb

    # upscale each cell to a block so grid is visible
    scale = 8  # change if you want bigger/smaller cells
    enlarged = np.kron(base_img, np.ones((scale, scale, 1), dtype=np.uint8))

    # draw yellow outline around markers
    outline_color = np.array([255, 255, 0], dtype=np.uint8)
    for (mx, my) in markers:
        mx = int(mx)
        my = int(my)
        if 0 <= mx < w and 0 <= my < h:
            sx, sy = mx * scale, my * scale
            ex, ey = (mx + 1) * scale, (my + 1) * scale

            enlarged[sy, sx:ex] = outline_color        # top
            enlarged[ey - 1, sx:ex] = outline_color    # bottom
            enlarged[sy:ey, sx] = outline_color        # left
            enlarged[sy:ey, ex - 1] = outline_color    # right

    return enlarged

# --- 3. EPISODE / BATCH EVAL HELPERS ---------------------------------------

def run_episode(use_agents=True, max_steps=50, seed=0):
    """
    Run ONE simulation episode.
    If use_agents=False ⇒ baseline (no interventions).
    Returns metrics dict.
    """
    rng = np.random.default_rng(seed)
    fuel_map = create_random_map(rng)
    sim = FireSimulator(MAP_WIDTH, MAP_HEIGHT, fuel_map, random_state=seed)

    # find structure position once
    structure_pos = np.argwhere(fuel_map == STRUCTURE)[0]
    sy, sx = int(structure_pos[0]), int(structure_pos[1])

    if use_agents:
        pa = PerceptionAgent()
        pred = PredictionAgent()
        plan = PlanningAgent()

    for _ in range(max_steps):
        if use_agents:
            real = sim.get_state()
            belief = pa.scan(real)
            future = pred.predict_future(sim, steps_to_predict=5)
            actions = plan.create_plan(
                believed_state=belief,
                future_state=future,
                objectives=["Protect structure"],
                resources={"ground_crews": 3}
            )
        else:
            actions = []  # baseline: do nothing

        sim.step(actions)

        if np.sum(sim.get_state() == BURNING) == 0:
            break

    final = sim.get_state()
    burned_area = int(np.sum(final != UNBURNED))
    structure_survived = final[sy, sx] != BURNING

    return {
        "mode": "Agent" if use_agents else "Baseline",
        "burned_area": burned_area,
        "structure_survived": int(structure_survived),
        "steps": sim.time_step,
        "seed": seed
    }

def run_many_episodes(num_episodes=20, max_steps=50):
    """
    Run multiple episodes for both Baseline and Agent (same seeds).
    Returns a pandas DataFrame with all episode metrics.
    """
    records = []

    for i in range(num_episodes):
        # same seed for fair comparison
        seed = i
        baseline_metrics = run_episode(use_agents=False, max_steps=max_steps, seed=seed)
        agent_metrics = run_episode(use_agents=True, max_steps=max_steps, seed=seed)
        records.append(baseline_metrics)
        records.append(agent_metrics)

    df = pd.DataFrame(records)
    return df

# --- 4. STREAMLIT APP SETUP ------------------------------------------------

st.set_page_config(layout="wide", page_title="Wildfire Agents")
st.title("🔥 Agentic Wildfire Response System")

# --- 5. INITIALIZATION (Session State) -------------------------------------

if "simulator" not in st.session_state:
    print("--- FIRST RUN: INITIALIZING APP STATE ---")
    rng = np.random.default_rng(42)

    initial_map = create_initial_map()
    st.session_state.sim_seed = 42
    st.session_state.simulator = FireSimulator(MAP_WIDTH, MAP_HEIGHT, initial_map,
                                               random_state=st.session_state.sim_seed)

    st.session_state.perception_agent = PerceptionAgent()
    st.session_state.prediction_agent = PredictionAgent()
    st.session_state.planning_agent = PlanningAgent()

    st.session_state.resources = {"ground_crews": 5, "drones": 2}
    st.session_state.objectives = ["Minimize burned area", "Protect all structures"]

    st.session_state.believed_state = None
    st.session_state.future_scenarios = None
    st.session_state.last_actions = []
    st.session_state.eval_results = None  # for batch plots

# --- 6. APP LAYOUT: SIM + CONTROL PANEL ------------------------------------

col1, col2 = st.columns([3, 1])

with col1:
    st.header("Geospatial Simulation")
    image_placeholder = st.empty()

with col2:
    st.header("Agent Control Panel (MCP)")

    # Run one step
    if st.button("▶️ Run Next Timestep"):
        sim = st.session_state.simulator
        pa = st.session_state.perception_agent
        pred = st.session_state.prediction_agent
        plan = st.session_state.planning_agent

        real = sim.get_state()
        believed_state = pa.scan(real)
        st.session_state.believed_state = believed_state

        future_state = pred.predict_future(simulator=sim, steps_to_predict=5)
        st.session_state.future_scenarios = future_state

        plan_actions = plan.create_plan(
            believed_state=believed_state,
            future_state=future_state,
            objectives=st.session_state.objectives,
            resources=st.session_state.resources
        )
        st.session_state.last_actions = plan_actions

        st.session_state.simulator.step(actions=plan_actions)

    # New random simulation
    if st.button("🔄 New Simulation"):
        rng = np.random.default_rng()
        new_seed = int(rng.integers(0, 1_000_000))
        st.session_state.sim_seed = new_seed

        fuel_map = create_random_map(rng)
        st.session_state.simulator = FireSimulator(
            MAP_WIDTH, MAP_HEIGHT, fuel_map, random_state=new_seed
        )

        st.session_state.perception_agent = PerceptionAgent()
        st.session_state.prediction_agent = PredictionAgent()
        st.session_state.planning_agent = PlanningAgent()

        st.session_state.believed_state = None
        st.session_state.future_scenarios = None
        st.session_state.last_actions = []

    # Reset to deterministic center-fire demo
    if st.button("Reset Simulation"):
        initial_map = create_initial_map()
        st.session_state.sim_seed = 42
        st.session_state.simulator = FireSimulator(
            MAP_WIDTH, MAP_HEIGHT, initial_map, random_state=42
        )
        st.session_state.perception_agent = PerceptionAgent()
        st.session_state.prediction_agent = PredictionAgent()
        st.session_state.planning_agent = PlanningAgent()
        st.session_state.believed_state = None
        st.session_state.future_scenarios = None
        st.session_state.last_actions = []

    # Agent status text
    st.subheader("Agent Status")
    st.write("**Objectives (what we care about):**", st.session_state.objectives)
    st.write("**Resources (what we have):**", st.session_state.resources)

    st.write("**What the agent did last step:**")
    actions = st.session_state.last_actions
    if not actions:
        st.write("• No action taken yet – the agent is just watching the fire.")
    else:
        for action_type, pos in actions:
            x, y = int(pos[0]), int(pos[1])
            if action_type == "build_break":
                st.write(f"• Built a firebreak at ({x}, {y}).")
            else:
                st.write(f"• {action_type} at ({x}, {y}).")

    st.subheader("Legend")
    st.markdown(
        "- 🟩 **Green**: Unburned forest\n"
        "- 🔴 **Red**: Burning\n"
        "- ⚫ **Dark gray**: Burned out\n"
        "- 🟫 **Brown**: Firebreak\n"
        "- 🔵 **Blue**: Structure to protect\n"
        "- ⚪ **Light gray**: Unknown (smoke / not visible to scout)\n"
        "- 🟨 **Yellow outline**: Last firebreak chosen by the agent"
    )

# --- 7. MAIN VISUALIZATION --------------------------------------------------

current_grid = st.session_state.simulator.get_state()

markers = []
for action_type, pos in st.session_state.last_actions:
    if action_type == "build_break":
        x, y = int(pos[0]), int(pos[1])
        markers.append((x, y))

display_image = convert_grid_to_image(current_grid, markers=markers)

image_placeholder.image(
    display_image,
    caption=f"Current Fire Grid (Time step: {st.session_state.simulator.time_step})",
    use_container_width=True,
    output_format="PNG"
)

active_fire_cells = int(np.sum(current_grid == BURNING))
if active_fire_cells == 0:
    st.success("✅ Fire fully contained: no active burning cells remain.")
else:
    st.info(f"🔥 Active fire cells right now: {active_fire_cells}")

# --- 8. PERCEPTION + PREDICTION VIEWS --------------------------------------

st.markdown("---")
st.subheader("Agent Views: Perception & Prediction")

v1, v2, v3, v4 = st.columns(4)

if st.session_state.believed_state is not None:
    belief_img = convert_grid_to_image(st.session_state.believed_state)
    v1.image(belief_img, caption="Perception: Believed State",
             use_container_width=True, output_format="PNG")
else:
    v1.info("Run a timestep to see the Perception view.")

scenarios = st.session_state.future_scenarios

if isinstance(scenarios, dict) and "no_intervention" in scenarios:
    no_int_img = convert_grid_to_image(scenarios["no_intervention"])
    v2.image(no_int_img, caption="Prediction: No Intervention",
             use_container_width=True, output_format="PNG")
else:
    v2.info("No prediction yet. Run a timestep.")

if isinstance(scenarios, dict) and "with_firebreak" in scenarios:
    fb_img = convert_grid_to_image(scenarios["with_firebreak"])
    v3.image(fb_img, caption="Prediction: With Firebreak Plan",
             use_container_width=True, output_format="PNG")
else:
    v3.info("Firebreak scenario will appear after first prediction.")

if isinstance(scenarios, dict) and "wind_shift_east" in scenarios:
    wind_img = convert_grid_to_image(scenarios["wind_shift_east"])
    v4.image(wind_img, caption="Prediction: Wind Shift East",
             use_container_width=True, output_format="PNG")
else:
    v4.info("Wind-shift scenario will appear after first prediction.")

# --- 9. BATCH EVALUATION: BASELINE VS AGENT --------------------------------

st.markdown("---")
st.subheader("System Evaluation: Baseline vs Agentic System")

col_eval_left, col_eval_right = st.columns([1, 2])

with col_eval_left:
    num_eps = st.slider("Number of random simulations",
                        min_value=5, max_value=50, value=20, step=5)
    max_steps = st.slider("Max steps per simulation",
                          min_value=20, max_value=100, value=50, step=10)

    if st.button("📊 Run Batch Evaluation"):
        with st.spinner("Running simulations..."):
            df = run_many_episodes(num_episodes=num_eps, max_steps=max_steps)
            st.session_state.eval_results = df

with col_eval_right:
    df = st.session_state.eval_results
    if df is None:
        st.info("Run the batch evaluation to see summary metrics and plots.")
    else:
        st.write("Raw episode metrics (first few rows):")
        st.dataframe(df.head())

        # --- CREATE SIDE-BY-SIDE COMPARISON TABLE ---
    if df is not None:
        # Pivot so each seed has Baseline + Agent side by side
        paired = df.pivot(index="seed", columns="mode", values=["burned_area", "steps", "structure_survived"])
    
        # Flatten column names
        paired.columns = [f"{metric}_{mode}" for metric, mode in paired.columns]
        paired = paired.reset_index()

        st.write("### 🔍 Seed-by-Seed Comparison (Baseline vs Agent)")
        st.dataframe(paired)

        # Summary table
        st.write("### 📊 Summary Metrics (Averages Across All Runs)")
        summary = df.groupby("mode").agg(
            avg_burned_area=("burned_area", "mean"),
            avg_steps=("steps", "mean"),
            survival_rate=("structure_survived", "mean")
        ).reset_index()
        summary["survival_rate (%)"] = summary["survival_rate"] * 100
        st.dataframe(summary)

        st.write("### 🔥 Burned Area Comparison (lower is better)")
        st.bar_chart(summary.set_index("mode")[["avg_burned_area"]])

        st.write("### 🏡 Structure Survival Rate (%)")
        st.bar_chart(summary.set_index("mode")[["survival_rate (%)"]])

        summary["survival_rate (%)"] = summary["survival_rate"] * 100
