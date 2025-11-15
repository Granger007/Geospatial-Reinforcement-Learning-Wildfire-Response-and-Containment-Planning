import streamlit as st
import numpy as np
import time

# Import your custom Python files (classes)
from simulator import FireSimulator, UNBURNED, BURNING, BURNED_OUT, FIREBREAK, STRUCTURE
from agents import PerceptionAgent, PredictionAgent, PlanningAgent

# --- 1. CONFIGURATION ---

# Set the size of our map
MAP_WIDTH = 50
MAP_HEIGHT = 50

# This dictionary is for visualization. It maps the *numbers* from our
# grid (e.g., 0, 1, 2) to an (R, G, B) color tuple.
# This is how we turn our data into a colorful image.
COLOR_MAP = {
    UNBURNED: (0, 102, 0),    # Dark Green (for forest)
    BURNING: (255, 0, 0),     # Bright Red
    BURNED_OUT: (40, 40, 40),   # Dark Gray (ash)
    FIREBREAK: (139, 69, 19),   # Brown (dirt)
    STRUCTURE: (0, 0, 200)     # Blue (house/town)
}

def create_initial_map():
    """Helper function to create the starting map."""
    # Start with a grid full of 'UNBURNED'
    fuel_map = np.full((MAP_HEIGHT, MAP_WIDTH), UNBURNED)
    
    # Start a fire in the middle
    # Note: NumPy arrays are (row, col) which is (y, x)
    fuel_map[MAP_HEIGHT // 2, MAP_WIDTH // 2] = BURNING
    
    # Add a "structure" to protect, offset from the fire
    fuel_map[MAP_HEIGHT // 2 + 5, MAP_WIDTH // 2 + 5] = STRUCTURE
    return fuel_map

def convert_grid_to_image(grid):
    """
    Helper function to convert our 2D number grid into a 3D RGB image
    that Streamlit can display.
    """
    # Create an empty 3D array (h, w, 3) filled with zeros
    # 'np.uint8' is the data type for images (unsigned 8-bit integer, 0-255)
    image_grid = np.zeros((MAP_HEIGHT, MAP_WIDTH, 3), dtype=np.uint8)
    
    # Loop through our color map
    for state_number, color_rgb in COLOR_MAP.items():
        # 'grid == state_number' creates a True/False mask
        # 'image_grid[mask]' selects all the pixels in the image
        # that correspond to that state.
        # We then set those pixels to the correct color.
        image_grid[grid == state_number] = color_rgb
        
    return image_grid

# --- 2. STREAMLIT APP SETUP ---

# This sets the browser tab title and makes the app use the full page width
st.set_page_config(layout="wide", page_title="Wildfire Agents")

st.title("🔥 Agentic Wildfire Response System")

# --- 3. INITIALIZATION (The "Memory") ---

# This is the MOST IMPORTANT part of a Streamlit app.
# Streamlit re-runs this *entire script* from top to bottom every
# time you interact with it (e.g., click a button).
#
# 'st.session_state' is a special dictionary that *persists*
# between re-runs. It's the app's "memory."
#
# We check if we've already initialized our simulator.
# If not, this is the *first time* the user is running the app.
if 'simulator' not in st.session_state:
    print("--- FIRST RUN: INITIALIZING APP STATE ---")
    
    # 1. Initialize the World
    initial_map = create_initial_map()
    st.session_state.simulator = FireSimulator(MAP_WIDTH, MAP_HEIGHT, initial_map)
    
    # 2. Initialize the Agents
    st.session_state.perception_agent = PerceptionAgent()
    st.session_state.prediction_agent = PredictionAgent()
    st.session_state.planning_agent = PlanningAgent()
    
    # 3. Initialize Resources
    st.session_state.resources = {'ground_crews': 5, 'drones': 2}
    
    # 4. Initialize Objectives
    st.session_state.objectives = ["Minimize burned area", "Protect all structures"]


# --- 4. APP LAYOUT (The UI) ---

# Create two columns, with the map (col1) being 3x wider than the panel (col2)
col1, col2 = st.columns([3, 1])

with col1:
    st.header("Geospatial Simulation")
    
    # 'st.empty()' creates an empty placeholder.
    # We will fill this placeholder with our image later.
    # This is crucial for *updating* the image without
    # redrawing the whole page.
    image_placeholder = st.empty()

with col2:
    st.header("Agent Control Panel (MCP)")
    
    # 'st.button' creates a button.
    # The 'if' block *only* runs if the button is clicked.
    if st.button("▶️ Run Next Timestep"):
        
        print("--- BUTTON CLICKED: RUNNING MAIN LOOP ---")
        
        # === THIS IS THE MAIN AGENTIC LOOP ===
        
        # We get our objects *out* of the session_state "memory"
        sim = st.session_state.simulator
        perception_agent = st.session_state.perception_agent
        prediction_agent = st.session_state.prediction_agent
        planning_agent = st.session_state.planning_agent

        # 1. PERCEIVE: The agent scans the world.
        current_real_state = sim.get_state()
        believed_state = perception_agent.scan(current_real_state)
        
        # 2. PREDICT: The agent forecasts the future.
        future_state = prediction_agent.predict_future(
            simulator=sim,  # Pass the *real* sim for copying
            steps_to_predict=5 # Look 5 steps ahead
        )
        
        # 3. PLAN: The "Commander" agent makes a decision.
        plan_actions = planning_agent.create_plan(
            believed_state=believed_state,
            future_state=future_state,
            objectives=st.session_state.objectives,
            resources=st.session_state.resources
        )
        
        # 4. ACT: The simulator ticks forward with the agent's plan.
        # The 'step' function modifies the 'sim.grid' in-place.
        st.session_state.simulator.step(actions=plan_actions)
        
        # The script will now finish, and Streamlit will
        # re-run it from the top. The 'if' button block will be false,
        # but the code below will run, redrawing the *updated* grid.

    if st.button("Reset Simulation"):
        # This is simple: just clear the "memory" and re-run the script.
        # When the script re-runs, the 'if 'simulator' not in st.session_state'
        # block will be TRUE, and it will re-initialize everything.
        st.session_state.clear()
        st.rerun()

    # --- Display Agent Logs/Info ---
    st.subheader("Agent Status")
    # We can read directly from session state to display info
    st.write("**Objectives:**", st.session_state.objectives)
    st.write("**Resources:**", st.session_state.resources)
    

# --- 5. UPDATE VISUALIZATION ---

# This code runs *every* time the script runs (including after a button click).
# 1. Get the *latest* grid from our "memory".
current_grid = st.session_state.simulator.get_state()

# 2. Convert that grid into a colorful image.
display_image = convert_grid_to_image(current_grid)

# 3. Put the new image into the placeholder we created earlier.
#    'use_column_width=True' makes the image fit nicely.
image_placeholder.image(display_image, 
                        caption=f"Current Fire Grid (Time: {st.session_state.get('time', 0)})", 
                        use_column_width=True, 
                        output_format="PNG") # Use PNG for sharp pixels