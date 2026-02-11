import numpy as np
import random
import logging

class SimpleCarEnv:
    """
    A simplified 2D grid-world environment for an autonomous car.
    The car's goal is to reach a target while avoiding obstacles.
    """
    def __init__(self, map_size=10, obstacle_density=0.2, seed=None):
        self.map_size = map_size
        self.state_dim = 2 # (x, y) coordinates
        self.action_dim = 4 # 0: Up, 1: Down, 2: Left, 3: Right

        self.start_pos = (0, 0)
        self.target_pos = (map_size - 1, map_size - 1)
        
        self.rng = random.Random(seed) # Use instance-specific RNG
        self.np_rng = np.random.default_rng(seed) # Use instance-specific NumPy RNG

        self.obstacles = self._generate_obstacles(obstacle_density)
        self.current_pos = None
        self.steps_taken = 0
        self.max_episode_steps = map_size * map_size * 2 # Max steps to prevent infinite loops

        # Define action effects
        self.action_map = {
            0: (0, 1),  # Up (change in y)
            1: (0, -1), # Down (change in y)
            2: (-1, 0), # Left (change in x)
            3: (1, 0)   # Right (change in x)
        }
        
        logging.debug(f"Env initialized (seed={seed}). Target: {self.target_pos}, Obstacles: {len(self.obstacles)}")

    def _generate_obstacles(self, density):
        """Generates random obstacles on the map."""
        obstacles = set()
        num_obstacles = int(self.map_size * self.map_size * density)
        
        # Ensure start and target positions are clear
        forbidden_positions = {self.start_pos, self.target_pos}

        while len(obstacles) < num_obstacles:
            ox = self.rng.randint(0, self.map_size - 1)
            oy = self.rng.randint(0, self.map_size - 1)
            new_obstacle = (ox, oy)
            if new_obstacle not in forbidden_positions:
                obstacles.add(new_obstacle)
        return obstacles

    def reset(self):
        """Resets the environment to a new episode."""
        self.current_pos = self.start_pos
        self.steps_taken = 0
        state = np.array(self.current_pos, dtype=np.float32)
        # Normalize state to [0, 1] range
        state = state / (self.map_size - 1) if self.map_size > 1 else state
        return state

    def step(self, action):
        """
        Takes an action in the environment.
        Args:
            action (int): The action to take (0-3).
        Returns:
            tuple: (next_state, reward, done, info)
        """
        if not (0 <= action < self.action_dim):
            raise ValueError(f"Invalid action: {action}. Must be 0-{self.action_dim-1}.")

        current_x, current_y = self.current_pos
        dx, dy = self.action_map[action]
        
        new_x = max(0, min(self.map_size - 1, current_x + dx))
        new_y = max(0, min(self.map_size - 1, current_y + dy))
        new_pos = (new_x, new_y)

        reward = -0.01  # Small penalty for each step
        done = False
        
        # Check for boundary hit (car tried to move off-grid and bounced back)
        if new_pos == self.current_pos and (new_x != current_x + dx or new_y != current_y + dy):
             reward -= 0.1 # Small penalty for hitting boundary

        self.current_pos = new_pos
        self.steps_taken += 1

        if self.current_pos == self.target_pos:
            reward += 10.0  # Big reward for reaching target
            done = True
        elif self.current_pos in self.obstacles:
            reward -= 5.0  # Big penalty for hitting an obstacle
            done = True
        elif self.steps_taken >= self.max_episode_steps:
            reward -= 1.0 # Penalty for running out of time
            done = True

        state = np.array(self.current_pos, dtype=np.float32)
        # Normalize state to [0, 1] range
        state = state / (self.map_size - 1) if self.map_size > 1 else state
        
        info = {} # Optional debug info
        return state, reward, done, info

if __name__ == '__main__':
    # Setup basic logging for environment test
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    # Simple test for the environment
    env = SimpleCarEnv(map_size=5, obstacle_density=0.1, seed=42)
    state = env.reset()
    print(f"Initial State: {state}")

    done = False
    total_reward = 0
    steps = 0

    # Simulate random actions until episode ends
    while not done:
        action = random.randint(0, env.action_dim - 1)
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        print(f"Step {steps}: Action {action}, New State: {next_state}, Reward: {reward}, Done: {done}")
        if steps > 100: # Emergency break for very long episodes
            print("Episode too long, breaking.")
            break
            
    print(f"\nEpisode finished in {steps} steps with total reward: {total_reward}")

    # Test another episode
    state = env.reset()
    print(f"New Episode Initial State: {state}")