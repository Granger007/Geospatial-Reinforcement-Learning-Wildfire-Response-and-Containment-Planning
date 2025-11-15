import numpy as np
import copy # We need this for the PredictionAgent

class PerceptionAgent:
    """
    AGENT 1: THE "SCOUT"
    Its job is to look at the "real world" (the simulator) and build a
    "believed world state" for the other agents.
    
    In a simple system, this is easy. In a complex one, this agent
    would handle "partial observability" (e.g., "I can't see this area
    because of smoke").
    """
    def scan(self, simulator_state):
        """
        Answers the question: "What is happening right now?"
        
        Args:
            simulator_state (numpy.ndarray): The "ground truth" from the simulator.
        
        Returns:
            numpy.ndarray: The "Believed World State."
        """
        print("Perception Agent: Scanning world... (Current belief: Perfect vision)")
        
        # --- TODO: ADVANCED ---
        # For a beginner, we'll assume the agent has perfect vision.
        # The "believed state" is exactly the "real state."
        #
        # Later, you could add logic here:
        # 1. Take 'simulator_state'
        # 2. Randomly hide 10% of the map (np.nan) and call it "smoke"
        # 3. Use "drone" resources to "un-hide" areas near the fire
        # 4. Return this "smoky_map" as the believed state.
        
        return simulator_state

class PredictionAgent:
    """
    AGENT 2: THE "ORACLE"
    Its job is to look into the future. It takes the "Believed World State"
    from the Perception Agent and runs its *own* simulation to see what
    *might* happen.
    """
    def predict_future(self, simulator, steps_to_predict):
        """
        Answers the question: "What *will* happen if we do nothing?"
        
        Args:
            simulator (FireSimulator): The *actual* simulator object.
                                       We need this to run our own copy.
            steps_to_predict (int): How many time-steps to look ahead.
        
        Returns:
            numpy.ndarray: The predicted *future* grid state.
        """
        print(f"Prediction Agent: Running 'what-if' scenario for {steps_to_predict} steps...")
        
        # CRITICAL STEP: We MUST use 'copy.deepcopy()'.
        # This makes a complete, independent copy of the simulator,
        # including its internal 'grid'.
        # If we just did 'temp_simulator = simulator', we would be
        # modifying the *real* world, which would break everything.
        temp_simulator = copy.deepcopy(simulator)
        
        # Run the 'temp_simulator' forward N times
        for _ in range(steps_to_predict):
            # We run the step with 'actions=[]' (an empty list)
            # to simulate what happens if we do *nothing*.
            temp_simulator.step(actions=[])
            
        # --- TODO: ADVANCED ---
        # This is where you would run *multiple* scenarios.
        # - Scenario A: 'actions=[]' (No intervention)
        # - Scenario B: 'actions=[(build_break, (x,y))]' (Test a specific plan)
        # - Scenario C: Modify 'temp_simulator.wind_grid' and *then* run
        
        # Return the final state of our *temporary* simulator
        return temp_simulator.get_state()

class PlanningAgent:
    """
    AGENT 3: THE "COMMANDER"
    This is the "brain" of the operation. It takes all the information
    (current state, future state, objectives) and decides what to do.
    """
    def create_plan(self, believed_state, future_state, objectives, resources):
        """
        Answers the question: "What is our plan?"
        
        Args:
            believed_state (numpy.ndarray): From the Perception Agent.
            future_state (numpy.ndarray): From the Prediction Agent.
            objectives (list): e.g., ["Protect all structures"]
            resources (dict): e.g., {'ground_crews': 5}
            
        Returns:
            list: A list of actions to send to the simulator.
                  e.g., [('build_break', (25, 30))]
        """
        print("Planning Agent: Analyzing states to create a plan...")
        
        # This will be the list of actions we return
        actions_list = []
        
        # --- TODO: ADVANCED ---
        # This is the CORE of your project. The simple heuristic below
        # should be replaced with a real "reasoning loop."
        # This is where you would call an LLM (like GPT)
        # prompt = f"""
        # World State: {believed_state}
        # Prediction: {future_state}
        # My objectives: {objectives}
        # My resources: {resources}
        # Based on this, what is the 'Thought, Critique, Refined Plan'?
        # """
        # response = openai.ChatCompletion.create(...)
        # plan = parse_response(response)
        # return plan
        
        # --- BEGINNER'S HEURISTIC LOGIC (Simple Rules) ---
        
        # Our simple logic:
        # 1. Find all cells that are BURNING *right now*.
        # 2. For the first burning cell we find...
        # 3. Look at its neighbors.
        # 4. If a neighbor is UNBURNED, build a FIREBREAK there to stop the spread.
        # 5. We'll only use one resource (one action) per step for simplicity.
        
        # `np.argwhere` finds the (y, x) coordinates of all cells
        # that match our condition (== BURNING).
        burning_cells = np.argwhere(believed_state == BURNING)
        
        if burning_cells.size > 0:
            # We have at least one fire.
            # Let's just grab the coordinates of the first one in the list.
            y, x = burning_cells[0] 
            
            print(f"Planning Agent: Threat detected at ({x}, {y})...")
            
            # Look at all 8 neighbors
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue # Skip ourself
                    
                    nx, ny = x + dx, y + dy
                    
                    # Check if neighbor is in-bounds
                    if 0 <= nx < believed_state.shape[1] and 0 <= ny < believed_state.shape[0]:
                        
                        # Is this neighbor unburned?
                        if believed_state[ny, nx] == UNBURNED:
                            # YES! This is a good place to build a firebreak.
                            action_pos = (nx, ny)
                            
                            print(f"Planning Agent: Recommending firebreak at {action_pos}")
                            
                            # Add our new action to the plan
                            actions_list.append(('build_break', action_pos))
                            
                            # Our simple agent only does one thing at a time.
                            # So, we return the plan immediately.
                            return actions_list
        
        # If the 'if' statement never ran or no unburned neighbors were found,
        # we return an empty list, meaning "no actions to take."
        print("Planning Agent: No immediate actions recommended.")
        return actions_list