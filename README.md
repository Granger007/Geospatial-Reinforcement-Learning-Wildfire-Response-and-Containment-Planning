# 🔥 Agentic Wildfire Response System

Welcome to the Wildfire Response project\! This is a "simulation" that uses AI agents to try and control a wildfire.

Think of it like a video game:

  * `simulator.py`: This file is the "game world." It knows how fire spreads.
  * `agents.py`: These are the "AI players." Their job is to look at the fire and make a plan to stop it.
  * `dashboard.py`: This is the "TV screen" you watch. It runs the whole simulation and shows you the fire map.

Our goal is to build and test how well our AI agents can fight the fire by digging "firebreaks" (patches of dirt where the fire can't spread).

-----

## Setup (Getting this to run on your computer)

You'll need to set up a few things just once to get the project working.

### 1\. Create a "Project Bubble" (Virtual Environment)

We need to create a clean, isolated "bubble" for this project so the Python code and its special packages don't mess up any other projects on your computer. This is called a **virtual environment**.

  * **Open your terminal** (like Command Prompt on Windows or Terminal on Mac).

  * **Go to this project's folder:**

    ```bash
    # 'cd' stands for 'change directory'. 
    # Drag your project folder onto the terminal to get its path.
    cd /path/to/your/agentic-wildfire-system
    ```

  * **Create the bubble:**

    ```bash
    # This command tells Python to create a virtual environment
    # and put it in a new folder named 'venv'
    python -m venv venv
    ```

  * **Activate the bubble:**
    You must do this *every time* you work on the project.

      * **On Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
      * **On Mac/Linux:**
        ```bash
        source venv/bin/activate
        ```

    > **Success\!** You'll know it worked if you see `(venv)` at the start of your terminal prompt.

### 2\. Install the Project's "Tools" (Dependencies)

This project needs two special "tools" (Python libraries) to work:

  * `numpy`: A library that is very good at handling the grid of numbers for our map.
  * `streamlit`: The library that creates the web-based dashboard and runs our code.

We'll install them from a file called `requirements.txt`.

  * **First, create a file** in your project folder named `requirements.txt`.
  * **Copy and paste this text** into that new file:
    ```
    streamlit
    numpy
    # Add 'rasterio', 'gdal' later when you add real data
    # Add 'openai' later if you add an LLM to the PlanningAgent
    ```
  * **Now, run the installer** in your terminal (make sure your `(venv)` is still active\!):
    ```bash
    # 'pip' is Python's installer.
    # This command reads your list and installs everything on it.
    pip install -r requirements.txt
    ```

-----

## How to Run the Simulation

This is the easy part\!

1.  Make sure your virtual environment is **active** (you see `(venv)` in your terminal).

2.  Run this command:

    ```bash
    streamlit run dashboard.py
    ```

That's it\! Your web browser will automatically open, showing you the dashboard.

  * Click the **"▶️ Run Next Timestep"** button to make the fire spread and watch the agents react.
  * Click the **"Reset Simulation"** button to start over.

-----

## What Each File Does (Project Structure)

Here's a map of the project so you know where everything is.

```
.
├── venv/                 # The "project bubble" you created. You can ignore it.
├── data/                 # (Optional) A folder you can create for real map data later.
│
├── dashboard.py          # ⭐️ The MAIN file. This is the one you run.
│                         # It creates the web page and acts as the "Master
│                         # Control Program," telling the agents and the
│                         # simulator what to do.
│
├── simulator.py          # The "World" or "Digital Twin."
│                         # This file defines the 'FireSimulator' class.
│                         # It knows the rules of the fire (e.g., "fire spreads
│                         # to neighbors") and doesn't know anything about agents.
│
├── agents.py             # The "Brains."
│                         # This file defines your AI agent classes:
│                         # - PerceptionAgent (The "Scout" - What is happening?)
│                         # - PredictionAgent (The "Oracle" - What will happen?)
│                         # - PlanningAgent (The "Commander" - What should we do?)
│
├── requirements.txt      # The "Shopping List."
│                         # This file tells Python which tools your project needs.
│
└── README.md             # This file! Your guide to the project.
```

-----

## Your Step-by-Step Mission (Project Roadmap)

Follow these steps to build this project from its simple boilerplate into a powerful simulation.

### Step 1: Get the Boilerplate Running. (You are here\!)

1.  Create the project folder.
2.  Set up the Python virtual environment (`venv`).
3.  Install the libraries (`pip install -r requirements.txt`).
4.  Save the 3 files (`simulator.py`, `agents.py`, `dashboard.py`).
5.  Run the dashboard (`streamlit run dashboard.py`).
6.  Click the "Run Next Timestep" button and watch the fire evolve.

### Step 2: Make the Simulator Realistic (The "Digital Twin").

  * This is where you stop using the simple map and load *real* data.
  * Read data (MODIS, VIIRS, DEM) using new libraries like `rasterio` and `gdal`.
  * Modify `simulator.py` to use this real data.
  * Implement the *real* fire spread logic in `simulator.step()`. Use the physics you planned (wind, slope, fuel). **This is the hardest *simulation* part.**

### Step 3: Make the Agents Smarter (The "Agentic System").

  * **PerceptionAgent:** Modify `scan()` to simulate "smoke" (e.g., randomly hide parts of the map) and "drones" (which can "un-hide" those parts).
  * **PredictionAgent:** Run the multiple scenarios you designed (e.g., "what if wind shifts?").
  * **PlanningAgent:** This is the **core AI part**. Replace the simple `if`-statement with a real reasoning loop. You can even call an LLM (like OpenAI's) inside `create_plan()` to generate the "Thought, Critique, Refined Plan" sequence.

### Step 4: Integrate Executor Agents.

  * Your `PlanningAgent`'s actions are just text right now (`'build_break'`).
  * Create new "Executor" classes (like `GroundCrewAgent`).
  * When the `PlanningAgent` makes a plan, it should "dispatch" the task to a `GroundCrewAgent`.
  * The `GroundCrewAgent` will manage its *own* state (e.g., "I'm traveling for 3 timesteps," "I'm now building the break") and send the final actions to the simulator.

### Step 5: Evaluate Your System.

  * Run your dashboard against a real historical fire dataset.
  * Run it *without* agents (Baseline 1: No Intervention) and record the `Total area burned`.
  * Run it *with* your agents (Your System) and record the `Total area burned`.
  * **Compare the results.** You've now scientifically proven your AI system works\!