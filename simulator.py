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
    
    def __init__(self, width, height, fuel_map):
        """
        This is the constructor, called when we first create a FireSimulator.
        e.g., sim = FireSimulator(50, 50, my_map)
        
        Args:
            width (int): The width of the map in grid cells.
            height (int): The height of the map in grid cells.
            fuel_map (numpy.ndarray): A 2D array representing the initial
                                      state of the world.
        """
        print("Simulator: Initializing world...")
        self.width = width
        self.height = height
        
        # The 'grid' is the heart of the simulator. It's a 2D NumPy array
        # where each number represents the state of that cell (UNBURNED, BURNING, etc.).
        # We make a copy of the passed-in map to ensure we don't accidentally
        # change the original data.
        self.grid = np.array(fuel_map, dtype=int)
        
        # --- TODO: ADVANCED ---
        # In a real project, you would also initialize your other data layers here.
        # Example:
        # self.wind_speed_grid = ... 
        # self.slope_grid = ...
        # self.humidity_grid = ...

    def get_state(self):
        """
        A simple "getter" method. Other parts of our program (like the agents)
        will call this to get the most up-to-date map of the world.
        
        Returns:
            numpy.ndarray: The current state of the _grid.
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
        print("Simulator: Running one time-step...")
        
        # --- 1. APPLY AGENT ACTIONS ---
        # First, we loop through all the actions the agents sent us.
        for action_type, pos in actions:
            if action_type == 'build_break':
                # 'pos' is an (x, y) tuple
                x, y = pos
                # Check to make sure the (x, y) is inside the map boundaries
                if 0 <= x < self.width and 0 <= y < self.height:
                    # If the cell is unburned, turn it into a firebreak.
                    # This "digs" the firebreak.
                    if self.grid[y, x] == UNBURNED:
                        self.grid[y, x] = FIREBREAK

        # --- 2. SPREAD THE FIRE (Simplified Physics) ---
        
        # CRITICAL STEP: We must create a *copy* of the grid.
        # Why? We need to calculate the *new* state based on the *old* state.
        # If we modified 'self.grid' directly, a fire spreading to a neighbor
        # at the top of the loop would then cause *that* neighbor to spread
        # fire all in the *same time-step*. This is a chain reaction that
        # makes the fire spread instantly, which is wrong.
        new_grid = np.copy(self.grid)
        
        # We loop through every single cell in the grid (y is row, x is column)
        for y in range(self.height):
            for x in range(self.width):
                
                # We only care about cells that are *currently* BURNING
                if self.grid[y, x] == BURNING:
                    
                    # --- TODO: ADVANCED ---
                    # This is where you would add real physics!
                    # You would check 'self.wind_grid[y, x]', 'self.slope_grid[y, x]',
                    # etc., to calculate a *probability* of spread.
                    
                    # --- Simplified Logic: Spread to all 8 neighbors ---
                    # We check all 8 neighbors (and the cell itself)
                    for dy in [-1, 0, 1]:  # dy is the change in y (delta y)
                        for dx in [-1, 0, 1]:  # dx is the change in x (delta x)
                            if dx == 0 and dy == 0:
                                continue # Don't check ourself
                            
                            # Calculate the neighbor's coordinates
                            nx, ny = x + dx, y + dy
                            
                            # Check if the neighbor is inside the map
                            if 0 <= nx < self.width and 0 <= ny < self.height:
                                
                                # If the neighbor is UNBURNED, it catches fire!
                                if self.grid[ny, nx] == UNBURNED:
                                    # We modify the *new_grid*, not the old one.
                                    new_grid[ny, nx] = BURNING
                    
                    # The cell that was burning is now burned out
                    new_grid[y, x] = BURNED_OUT

        # After checking all cells, we replace the old grid with the new one.
        self.grid = new_grid
        
        # Return the new state to whatever called the step function
        return self.grid