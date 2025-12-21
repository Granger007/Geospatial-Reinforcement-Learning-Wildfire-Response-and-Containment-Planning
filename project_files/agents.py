import numpy as np
import copy # We need this for the PredictionAgent
from simulator import FireSimulator, UNBURNED, BURNING, BURNED_OUT, FIREBREAK, STRUCTURE
class PerceptionAgent:
    """
    AGENT 1: THE "SCOUT"
    Now handles partial observability for the proof-of-concept:
    - Some cells are hidden by 'smoke'
    - 'Drones' can reveal areas around active fire
    """

    def __init__(self, smoke_prob=0.15, drone_radius=2, max_drones=3, random_state=None):
        """
        Args:
            smoke_prob   (float): Fraction of cells to hide as 'smoke' (0–1).
            drone_radius (int)  : How far around a target cell a drone reveals.
            max_drones   (int)  : Max number of drones per timestep.
            random_state (int)  : Optional seed for reproducibility.
        """
        self.smoke_prob = smoke_prob
        self.drone_radius = drone_radius
        self.max_drones = max_drones
        self.rng = np.random.default_rng(random_state)

    def scan(self, simulator_state):
        """
        Answers: "What is happening right now (that I can actually see)?"

        Args:
            simulator_state (numpy.ndarray): The "ground truth" from the simulator.

        Returns:
            numpy.ndarray: The "Believed World State" with smoke + drone reveals.
                           Unknown cells are marked with np.nan.
        """
        print(
            f"Perception Agent: Scanning world with smoke_prob="
            f"{self.smoke_prob}, drones={self.max_drones}"
        )

        # 1) Start from the true state
        truth = simulator_state
        belief = truth.astype(float).copy()  # float so we can store np.nan safely

        # 2) Add SMOKE: randomly hide some cells
        smoke_mask = self.rng.random(size=truth.shape) < self.smoke_prob
        belief[smoke_mask] = np.nan   # np.nan = "I can't see this cell"

        # 3) DRONES: reveal areas around "interesting" cells
        # For the POC, treat any cell > 0 as "fire / important".
        # Later, you can replace this with (truth == BURNING), etc.
        interesting_cells = np.argwhere(truth > 0)

        if interesting_cells.size > 0:
            self.rng.shuffle(interesting_cells)
            h, w = truth.shape

            num_drones = min(len(interesting_cells), self.max_drones)
            for (cy, cx) in interesting_cells[:num_drones]:
                y_min = max(0, cy - self.drone_radius)
                y_max = min(h, cy + self.drone_radius + 1)
                x_min = max(0, cx - self.drone_radius)
                x_max = min(w, cx + self.drone_radius + 1)

                # Reveal the truth in this window
                belief[y_min:y_max, x_min:x_max] = truth[y_min:y_max, x_min:x_max]

            print(f"Perception Agent: Launched {num_drones} drones to peek through smoke.")
        else:
            print("Perception Agent: No active fire / interesting cells; smoky view only.")

        return belief

class PredictionAgent:
    """
    AGENT 2: THE "ORACLE"
    Its job is to look into the future. It takes the "Believed World State"
    from the Perception Agent and runs its *own* simulations to see what
    *might* happen under different scenarios.
    """

    def _run_scenario(self, base_simulator, steps_to_predict,
                      actions_schedule=None, wind_transform=None):
        """
        Internal helper to run a single scenario on a deep-copied simulator.

        Args:
            base_simulator (FireSimulator): The actual simulator (will be deep-copied).
            steps_to_predict (int): How many time-steps to look ahead.
            actions_schedule (dict[int, list] | None): Optional mapping
                timestep -> list of actions, e.g. {0: [('build_break', (x, y))]}.
            wind_transform (callable | None): Optional function that takes
                a wind_grid and returns a modified wind_grid.

        Returns:
            numpy.ndarray: The final grid state for this scenario.
        """
        # 1) Work on a deep copy so we NEVER touch the real simulator.
        temp_simulator = copy.deepcopy(base_simulator)

        # 2) Optionally modify wind (for "what if wind shifts?" scenarios)
        if wind_transform is not None and hasattr(temp_simulator, "wind_grid"):
            temp_simulator.wind_grid = wind_transform(temp_simulator.wind_grid)

        # 3) Roll the simulation forward with scheduled actions
        for t in range(steps_to_predict):
            if actions_schedule is not None and t in actions_schedule:
                actions = actions_schedule[t]
            else:
                actions = []
            temp_simulator.step(actions=actions)

        return temp_simulator.get_state()

    def predict_future(self, simulator, steps_to_predict):
        """
        Answers the question: "What *will* happen under different assumptions?"

        Args:
            simulator (FireSimulator): The *actual* simulator object.
            steps_to_predict (int): How many time-steps to look ahead.

        Returns:
            dict[str, numpy.ndarray]: A dictionary of scenario_name -> future grid.
                Example keys:
                - "no_intervention"
                - "wind_shift_east"
                - "with_firebreak"
        """
        print(f"Prediction Agent: Running 'what-if' scenarios for {steps_to_predict} steps...")

        scenarios = {}

        # ---------------------------------------------------
        # Scenario A: NO INTERVENTION (baseline)
        # ---------------------------------------------------
        print("Prediction Agent: Scenario A - No intervention.")
        scenarios["no_intervention"] = self._run_scenario(
            base_simulator=simulator,
            steps_to_predict=steps_to_predict,
            actions_schedule=None,
            wind_transform=None
        )

        # ---------------------------------------------------
        # Scenario B: WIND SHIFT (e.g., wind moves more to the east)
        # ---------------------------------------------------
        print("Prediction Agent: Scenario B - Wind shifts to the east (if wind_grid exists).")

        def shift_wind_east(wind_grid):
            # Simple POC: roll the wind_grid one column to the right
            # to simulate a change in dominant wind direction.
            return np.roll(wind_grid, shift=1, axis=1)

        scenarios["wind_shift_east"] = self._run_scenario(
            base_simulator=simulator,
            steps_to_predict=steps_to_predict,
            actions_schedule=None,
            wind_transform=shift_wind_east if hasattr(simulator, "wind_grid") else None
        )

        # ---------------------------------------------------
        # Scenario C: TRY A FIREBREAK near a burning cell
        # ---------------------------------------------------
        print("Prediction Agent: Scenario C - Build a hypothetical firebreak near the fire.")

        # Look at the *current* state of the real simulator
        current_state = simulator.get_state()
        burning_cells = np.argwhere(current_state == BURNING)

        actions_schedule = None

        if burning_cells.size > 0:
            # Pick the first burning cell and try to place a break on an
            # unburned neighbor (similar to PlanningAgent's heuristic).
            y, x = burning_cells[0]
            h, w = current_state.shape

            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue

                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        if current_state[ny, nx] == UNBURNED:
                            # Found a candidate firebreak position
                            action_pos = (nx, ny)
                            print(f"Prediction Agent: Testing firebreak at {action_pos} in Scenario C.")
                            actions_schedule = {0: [('build_break', action_pos)]}
                            break
                if actions_schedule is not None:
                    break

        if actions_schedule is None:
            print("Prediction Agent: No good firebreak spot found; Scenario C behaves like no intervention.")

        scenarios["with_firebreak"] = self._run_scenario(
            base_simulator=simulator,
            steps_to_predict=steps_to_predict,
            actions_schedule=actions_schedule,
            wind_transform=None
        )

        # You now have multiple futures to hand over to PlanningAgent.
        # For now, PlanningAgent can ignore this or you can later update it to
        # inspect these scenarios and choose the safest plan.
        return scenarios

class PlanningAgent:
    """
    AGENT 3: THE "COMMANDER"
    This is the "brain" of the operation. It takes all the information
    (current state, future state, objectives) and decides what to do.
    """

    def _select_future_map(self, future_state):
        """
        Helper to get a single future map from the PredictionAgent output.

        future_state can be:
        - a single numpy.ndarray (old behavior), or
        - a dict[str, numpy.ndarray] (new multi-scenario PredictionAgent).

        We prefer the 'no_intervention' scenario if available.
        """
        # Case 1: already a single grid
        if not isinstance(future_state, dict):
            return future_state

        # Case 2: dict of scenarios
        if "no_intervention" in future_state:
            return future_state["no_intervention"]

        # Fall back to any one scenario
        for _, grid in future_state.items():
            return grid

        return None  # Should not normally happen

    def _find_candidate_firebreaks(self, believed_state):
        """
        Find all candidate cells where we *could* build a firebreak.

        Logic:
        - Find all BURNING cells in the believed state.
        - For each burning cell, look at its 8 neighbors.
        - A neighbor is a candidate if it is UNBURNED and inside bounds.

        Returns:
            list[tuple[int, int]]: List of (x, y) positions.
        """
        candidates = []

        # np.argwhere gives (y, x) for cells that are BURNING
        burning_cells = np.argwhere(believed_state == BURNING)

        if burning_cells.size == 0:
            return candidates

        h, w = believed_state.shape

        for (y, x) in burning_cells:
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue  # skip the burning cell itself

                    nx, ny = x + dx, y + dy

                    # Check in-bounds
                    if 0 <= nx < w and 0 <= ny < h:
                        # Check that the neighbor is UNBURNED
                        if believed_state[ny, nx] == UNBURNED:
                            candidates.append((nx, ny))

        # Deduplicate in case multiple fires share neighbors
        candidates = list(dict.fromkeys(candidates))
        return candidates

    def _score_candidate_with_future(self, candidate, future_map, neighborhood_radius=1):
        """
        Score a candidate firebreak location based on the future fire map.

        Intuition:
        - Look at a small neighborhood around the candidate in the future_map.
        - Count how many cells are BURNING there.
        - Higher count = more fire expected near this tile = more important
          to protect.

        Returns:
            int: risk score (number of burning cells nearby).
        """
        if future_map is None:
            return 0

        x, y = candidate
        h, w = future_map.shape

        y_min = max(0, y - neighborhood_radius)
        y_max = min(h, y + neighborhood_radius + 1)
        x_min = max(0, x - neighborhood_radius)
        x_max = min(w, x + neighborhood_radius + 1)

        region = future_map[y_min:y_max, x_min:x_max]
        return int(np.sum(region == BURNING))

    def create_plan(self, believed_state, future_state, objectives, resources):
        """
        Answers the question: "What is our plan?"

        Args:
            believed_state (numpy.ndarray): From the Perception Agent.
            future_state (numpy.ndarray or dict): From the Prediction Agent.
            objectives (list): e.g., ["Protect all structures"]
            resources (dict): e.g., {'ground_crews': 5}

        Returns:
            list: A list of actions to send to the simulator.
                  e.g., [('build_break', (25, 30))]
        """
        print("Planning Agent: Analyzing states to create a plan...")

        actions_list = []

        # 1) THOUGHT: understand the situation
        future_map = self._select_future_map(future_state)
        burning_now = np.argwhere(believed_state == BURNING)
        print(f"Planning Agent (Thought): Found {len(burning_now)} active fire cells.")

        candidates = self._find_candidate_firebreaks(believed_state)
        print(f"Planning Agent (Thought): Found {len(candidates)} candidate firebreak locations.")

        if not candidates:
            print("Planning Agent (Critique): No safe unburned neighbors to protect. No action.")
            return actions_list  # empty plan

        # 2) CRITIQUE: use the future map to score each candidate
        best_score = -1
        best_pos = None

        for pos in candidates:
            score = self._score_candidate_with_future(pos, future_map)
            # Higher score = more future fire nearby = higher priority
            if score > best_score:
                best_score = score
                best_pos = pos

        print(f"Planning Agent (Critique): Best candidate {best_pos} has risk score {best_score}.")

        # If the future map says there is no fire near any candidate,
        # maybe it's not worth building a break this step.
        if best_score <= 0:
            print("Planning Agent (Critique): Future fire risk near candidates is low. No action this turn.")
            return actions_list

        # 3) REFINED PLAN: choose actions based on resources
        max_crews = resources.get('ground_crews', 1)
        max_actions = max(1, min(max_crews, len(candidates)))

        # Simple POC: we only use the single best position,
        # but this is where you'd extend to multiple crews.
        chosen_positions = [best_pos][:max_actions]

        for pos in chosen_positions:
            print(f"Planning Agent (Refined Plan): Recommending firebreak at {pos}")
            actions_list.append(('build_break', pos))

        return actions_list