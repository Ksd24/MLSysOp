import torch

class Config:
    """
    Configuration class for all hyperparameters and settings.
    """
    # Environment settings
    MAP_SIZE = 5
    OBSTACLE_DENSITY = 0.0
    STATE_DIM = 2 # (x,y)
    ACTION_DIM = 4 # Up, Down, Left, Right
    MAX_EPISODE_STEPS = 50 # To prevent infinite loops

    # DQN Agent Hyperparameters
    LR = 1e-4                       # Learning rate
    GAMMA = 0.99                    # Discount factor
    BUFFER_CAPACITY = 100_000       # Replay buffer capacity
    BUFFER_MIN_SIZE = 5_000         # Min experiences in buffer before training starts
    BATCH_SIZE = 32                 # Batch size for learning
    TARGET_UPDATE_INTERVAL = 100    # Learner steps between target network updates

    # Epsilon-greedy exploration
    EPSILON_START = 1.0             # Initial epsilon
    EPSILON_MIN = 0.05              # Minimum epsilon
    EPSILON_DECAY_STEPS = 5_000   # Total environment steps for epsilon decay

    # Distributed System Specifics
    NUM_ACTORS = 8                 # Number of parallel environment workers
    ACTOR_SYNC_INTERVAL = 5         # Seconds, how often actors check for new model weights
    LEARNER_PUBLISH_INTERVAL = 10   # Seconds, how often learner publishes new model weights
    EXPERIENCE_QUEUE_MAX_SIZE = 10_000 # Max size of the queue for experiences from actors to learner

    # Training Duration/Termination
    TOTAL_TRAINING_SECONDS = 120    # Total wall-clock time to run the experiment
    TOTAL_LEARNER_STEPS_TARGET = 10_000 # Alternative termination: total learner steps (useful for analysis)

    # Device settings
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Logging and Seed
    LOG_LEVEL = 'INFO'              # DEBUG, INFO, WARNING, ERROR
    SEED = 42                       # Random seed for reproducibility

    def print_config(self):
        """Prints all configuration settings."""
        print("\n--- Configuration Settings ---")
        for attr, value in sorted(self.__dict__.items()):
            if not attr.startswith('__'):
                print(f"{attr:<25}: {value}")
        print("----------------------------\n")

# Instantiate config for easy access
config = Config()

if __name__ == '__main__':
    config.print_config()
    print(f"Device being used: {config.DEVICE}")