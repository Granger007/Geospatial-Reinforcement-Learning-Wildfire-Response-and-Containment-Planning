# Import the numpy library, which is the standard in Python for working with 
# large, multi-dimensional arrays (like our map grid). We give it the 
# nickname 'np' by convention.
import numpy as np

# --- Define Constants ---
# We use all-caps variable names to show they are "constants" - values
# that we don't plan on changing.
# This makes the code much more readable. Instead of writing 'grid[y, x] == 0',
# we can write 'grid[y, x] == UNBURNED'.

UNBURNED = 0    # Represents a grid cell with fuel that can burn (e.g., forest)
BURNING = 1     # Represents a grid cell that is currently on fire
BURNED_OUT = 2  # Represents a cell that has no fuel left
FIREBREAK = 3   # Represents a cell where agents have removed fuel (e.g., dirt)
STRUCTURE = 4   # Represents a cell with a house or town to protect

class FireSimulator:
    """
    This class is the "world." It holds the state of the fire map 
    and contains the core logic for how the fire spreads.
    """
    
    def __init__(self, width, height, fuel_map, random_state=None):
        """
        This is the constructor, called when we first create a FireSimulator.
        e.g., sim = FireSimulator(50, 50, my_map)
        
        Args:
            width (int): The width of the map in grid cells.
            height (int): The height of the map in grid cells.
            fuel_map (numpy.ndarray): A 2D array representing the initial
                                      state of the world.
            random_state (int | None): Optional seed for reproducibility.
        """
        print("Simulator: Initializing world...")
        self.width = width
        self.height = height
        
        # The 'grid' is the heart of the simulator. It's a 2D NumPy array
        # where each number represents the state of that cell (UNBURNED, BURNING, etc.).
        # We make a copy of the passed-in map to ensure we don't accidentally
        # change the original data.
        self.grid = np.array(fuel_map, dtype=int)

        # RNG for reproducible "physics"
        self.rng = np.random.default_rng(random_state)

        # --- SYNTHETIC "PHYSICS" LAYERS (FAKE DATA FOR POC) ---
        # In a real project, these would come from real rasters (wind, slope, etc.)
        #
        # For now, we just make them up so that:
        # - The PredictionAgent can modify wind_grid (Scenario B)
        # - You can say “we use additional geospatial layers” in your report

        # Wind strength grid: values around 1.0 (0.5 = weak, 1.5 = strong)
        self.wind_grid = self.rng.uniform(0.5, 1.5, size=(height, width))

        # Simple time counter (optional, but nice to display)
        self.time_step = 0

    def get_state(self):
        """
        A simple "getter" method. Other parts of our program (like the agents)
        will call this to get the most up-to-date map of the world.
        
        Returns:
            numpy.ndarray: The current state of the grid.
        """
        return self.grid

    def step(self, actions):
        """
        This is the main "tick" of the simulation. It advances the world 
        forward by one time-step. It does two things in order:
        1. Apply any actions that agents decided to take.
        2. Spread the fire based on the simulation's physics.
        
        Args:
            actions (list): A list of actions from the Planning Agent.
                            Example: [('build_break', (25, 30)), 
                                      ('build_break', (25, 31))]
        
        Returns:
            numpy.ndarray: The *new* state of the grid after the step.
        """
        print(f"Simulator: Running time-step {self.time_step}...")
        
        # --- 1. APPLY AGENT ACTIONS ---
        for action_type, pos in actions:
            if action_type == 'build_break':
                # 'pos' is an (x, y) tuple
                x, y = pos
                # Check to make sure the (x, y) is inside the map boundaries
                if 0 <= x < self.width and 0 <= y < self.height:
                    # If the cell is unburned, turn it into a firebreak.
                    # This "digs" the firebreak.
                    if self.grid[y, x] == UNBURNED:
                        print(f"Simulator: Digging firebreak at {(x, y)}")
                        self.grid[y, x] = FIREBREAK

        # --- 2. SPREAD THE FIRE (Simplified, Probabilistic Physics) ---
        
        # CRITICAL STEP: We must create a *copy* of the grid.
        new_grid = np.copy(self.grid)
        
        # Base probability that fire spreads from a burning cell to an
        # adjacent unburned one (before wind, etc.).
        base_spread_prob = 0.25

        # We loop through every single cell in the grid (y is row, x is column)
        for y in range(self.height):
            for x in range(self.width):
                
                # We only care about cells that are *currently* BURNING
                if self.grid[y, x] == BURNING:
                    
                    # Local "wind strength" at this burning cell
                    local_wind = self.wind_grid[y, x]

                    # --- Simplified Logic: Try to spread to all 8 neighbors ---
                    for dy in [-1, 0, 1]:  # dy is the change in y (delta y)
                        for dx in [-1, 0, 1]:  # dx is the change in x (delta x)
                            if dx == 0 and dy == 0:
                                continue  # Don't check ourself
                            
                            # Calculate the neighbor's coordinates
                            nx, ny = x + dx, y + dy
                            
                            # Check if the neighbor is inside the map
                            if 0 <= nx < self.width and 0 <= ny < self.height:
                                # Only UNBURNED cells can catch fire.
                                # (FIREBREAK, BURNED_OUT, STRUCTURE do not ignite in this POC.)
                                if self.grid[ny, nx] == UNBURNED:
                                    # Simple "physics":
                                    # - Start from a base probability
                                    # - Multiply by local wind strength
                                    spread_prob = base_spread_prob * local_wind
                                    spread_prob = float(np.clip(spread_prob, 0.05, 0.95))

                                    if self.rng.random() < spread_prob:
                                        new_grid[ny, nx] = BURNING
                    
                    # The cell that was burning is now burned out
                    new_grid[y, x] = BURNED_OUT

        # After checking all cells, we replace the old grid with the new one.
        self.grid = new_grid
        self.time_step += 1
        
        # Return the new state to whatever called the step function
        return self.grid