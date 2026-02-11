import torch
import torch.optim as optim
import torch.nn.functional as F
import multiprocessing as mp
import time
import logging
import queue # Correct import for queue.Empty
import random
import numpy as np
import os # For checking if the model file exists

# Import components from local files
from q_network import QNetwork
from simple_car_env import SimpleCarEnv
from replay_buffer import ReplayBuffer
from config import config # Import the Config instance

# --- Setup Logging ---
# Configure logging to show process name, which is crucial for distributed debugging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL),
                    format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s')

# --- Helper Functions ---
def set_seeds(seed):
    """Sets random seeds for reproducibility."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    logging.info(f"Random seed set to {seed}")

# --- Actor Process Function ---
def actor_fn(actor_id, env_creator, model_weights_queue, experience_queue, stop_event, config):
    """
    Function executed by each actor process.
    Collects experiences from its environment instance using a local policy.
    """
    process_name = f"Actor-{actor_id}"
    logging.info(f"{process_name} starting.")

    set_seeds(config.SEED + actor_id) # Use different seed for each actor's environment randomness

    # Pass config parameters explicitly to the environment creator
    env = env_creator(map_size=config.MAP_SIZE, obstacle_density=config.OBSTACLE_DENSITY,  seed=config.SEED + actor_id + 100)
    
    # Initialize a local Q-network for inference on CPU
    policy_net = QNetwork(config.STATE_DIM, config.ACTION_DIM).to('cpu') 
    policy_net.eval() # Set to evaluation mode (no gradients)

    state = env.reset()
    episode_reward = 0
    episode_steps = 0
    total_env_steps = 0 # Total steps taken by this specific actor

    last_model_sync_time = time.time()
    
    while not stop_event.is_set():
        # --- Model Synchronization (Pull weights from learner) ---
        if time.time() - last_model_sync_time > config.ACTOR_SYNC_INTERVAL:
            try:
                # Get the latest weights without blocking. If queue is empty, continue with current model.
                new_weights = model_weights_queue.get(timeout=0.01) 
                policy_net.load_state_dict(new_weights)
                last_model_sync_time = time.time()
                logging.debug(f"{process_name} synced model weights.")
            except queue.Empty: # Corrected: used 'queue.Empty'
                logging.debug(f"{process_name} model weights queue empty, using old weights.")
            except Exception as e:
                logging.error(f"{process_name} error syncing model weights: {e}")
                # For robustness, we'll continue with old weights.

        # --- Epsilon-greedy action selection ---
        epsilon = max(config.EPSILON_MIN, config.EPSILON_START * (1 - total_env_steps / config.EPSILON_DECAY_STEPS))
        
        if random.random() < epsilon:
            action = random.randint(0, config.ACTION_DIM - 1)
        else:
            with torch.no_grad(): # No gradient calculation needed for inference
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0) # Add batch dim
                q_values = policy_net(state_tensor)
                action = q_values.argmax(dim=1).item() # Select action with max Q-value

        # --- Environment Interaction ---
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        episode_steps += 1
        total_env_steps += 1

        # --- Send experience to learner ---
        try:
            # Experience queue might block if full, set timeout to avoid infinite block
            experience_queue.put((state, action, reward, next_state, done), timeout=1.0)
        except queue.Full: # Corrected: used 'queue.Full' if the queue is full
            logging.warning(f"{process_name} experience queue full, dropping experience.")
            # This indicates learner cannot keep up, or queue size is too small

        state = next_state
        if done:
            logging.info(f"{process_name} finished episode (total_steps={total_env_steps}): reward={episode_reward:.2f}, steps={episode_steps}, epsilon={epsilon:.2f}")
            state = env.reset()
            episode_reward = 0
            episode_steps = 0
            
    logging.info(f"{process_name} stopping after {total_env_steps} environment steps.")

# --- Learner Process Function ---
def learner_fn(initial_q_network_state_dict, model_weights_queue, experience_queue, stop_event, config):
    """
    Function executed by the learner process.
    Receives experiences, trains the DQN, and sends updated weights to actors.
    """
    process_name = "Learner"
    logging.info(f"{process_name} starting. Using device: {config.DEVICE}")

    set_seeds(config.SEED + 200) # Learner uses its own seed

    # Initialize policy and target networks on the specified device (GPU if available)
    policy_net = QNetwork(config.STATE_DIM, config.ACTION_DIM).to(config.DEVICE)
    policy_net.load_state_dict(initial_q_network_state_dict) # Load initial weights
    
    target_net = QNetwork(config.STATE_DIM, config.ACTION_DIM).to(config.DEVICE)
    target_net.load_state_dict(policy_net.state_dict()) # Target network starts as copy of policy
    target_net.eval() # Target network is not trained, set to eval mode

    optimizer = optim.Adam(policy_net.parameters(), lr=config.LR)
    
    # Using Huber loss (Smooth L1 Loss) as it's often more stable for Q-learning
    criterion = F.smooth_l1_loss # Or nn.MSELoss()

    replay_buffer = ReplayBuffer(config.BUFFER_CAPACITY)
    
    total_learner_steps = 0
    last_model_publish_time = time.time()
    last_target_update_step = 0
    
    # Store metrics for plotting (could be passed back to main process via a queue if needed)
    # For simplicity, we'll just log them.
    
    while not stop_event.is_set() or not experience_queue.empty(): # Continue processing remaining experiences
        # --- Collect Experiences from Queue into Replay Buffer ---
        while not experience_queue.empty():
            try:
                state, action, reward, next_state, done = experience_queue.get(timeout=0.01)
                replay_buffer.add(state, action, reward, next_state, done)
            except queue.Empty: # Corrected: used 'queue.Empty'
                break # Queue is temporarily empty
            except Exception as e:
                logging.error(f"{process_name} error adding experience from queue: {e}")
        
        # --- Training Step ---
        if len(replay_buffer) < config.BUFFER_MIN_SIZE:
            logging.debug(f"{process_name} buffer filling: {len(replay_buffer)}/{config.BUFFER_MIN_SIZE}")
            time.sleep(0.1) # Wait for buffer to fill if not enough experiences
            continue

        batch = replay_buffer.sample(config.BATCH_SIZE)
        if batch is None:
            time.sleep(0.01) # Not enough experiences to sample (e.g., after initial fill)
            continue

        states, actions, rewards, next_states, dones = (b.to(config.DEVICE) for b in batch)
        
        # Compute Q(s_t, a) - the model predicts Q-values for all actions,
        # we gather the Q-values for the actions that were actually taken.
        q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute V(s_{t+1}) for target - max_a Q_target(s_{t+1}, a)
        # We use target_net for stability
        with torch.no_grad():
            next_q_values = target_net(next_states).max(1)[0]
            # If done, the next state has no future reward, so its Q-value is 0.0
            next_q_values[dones == 1] = 0.0 
            target_q_values = rewards + config.GAMMA * next_q_values

        # Compute loss
        loss = criterion(q_values, target_q_values)
        
        # Optimize the model
        optimizer.zero_grad() # Clear previous gradients
        loss.backward()       # Compute gradients
        # Clip gradients to prevent exploding gradients
        for param in policy_net.parameters():
            if param.grad is not None: # Ensure gradients exist before clipping
                param.grad.data.clamp_(-1, 1)
        optimizer.step()      # Update weights

        total_learner_steps += 1
        # learner_losses.append(loss.item()) # Removed for performance if not used to plot in main
        # learner_step_counts.append(total_learner_steps) # Removed for performance

        # --- Target Network Update ---
        if total_learner_steps - last_target_update_step >= config.TARGET_UPDATE_INTERVAL:
            target_net.load_state_dict(policy_net.state_dict())
            last_target_update_step = total_learner_steps
            logging.debug(f"{process_name} updated target network at step {total_learner_steps}.")
            
        # --- Publish Model Weights to Actors ---
        if time.time() - last_model_publish_time > config.LEARNER_PUBLISH_INTERVAL:
            # Clear old weights from queue before putting new ones to avoid actors getting stale data
            while not model_weights_queue.empty():
                try:
                    model_weights_queue.get_nowait()
                except queue.Empty: # Corrected: used 'queue.Empty'
                    break # Queue is now empty
            model_weights_queue.put(policy_net.state_dict())
            last_model_publish_time = time.time()
            # Log current loss (could track average over time for better metrics)
            logging.info(f"{process_name} published new model weights. Current Loss: {loss.item():.4f}, Buffer size: {len(replay_buffer)}, Learner Steps: {total_learner_steps}")
        
        # Stop condition for learner based on total steps
        if total_learner_steps >= config.TOTAL_LEARNER_STEPS_TARGET:
            logging.info(f"{process_name} reached total learner steps target: {total_learner_steps}. Signalling stop.")
            stop_event.set() # Signal all processes to stop
            break # Exit learner loop

    logging.info(f"{process_name} stopping. Total training steps: {total_learner_steps}.")
    
    # Ensure final weights are put into the queue one last time for any straggling actors
    # It's good practice for actors to end with the most updated policy, even if just for logging.
    while not model_weights_queue.empty():
        try: model_weights_queue.get_nowait()
        except queue.Empty: break
    model_weights_queue.put(policy_net.state_dict())

    # Save the final model state_dict directly to disk (preferred method)
    final_model_path = "final_dqn_model.pth"
    torch.save(policy_net.state_dict(), final_model_path)
    logging.info(f"{process_name} saved final model to {final_model_path}")


# --- Main Orchestration Script ---
def main():
    """
    Main function to orchestrate the distributed DQN training.
    """
    config.print_config()
    set_seeds(config.SEED) # Set main process seed

    # --- Initialize Shared Components ---
    manager = mp.Manager()
    
    # Queue for Learner to send updated model weights to Actors
    model_weights_queue = manager.Queue() 
    # Queue for Actors to send collected experiences to Learner
    experience_queue = manager.Queue(maxsize=config.EXPERIENCE_QUEUE_MAX_SIZE) 
    # Event to signal all processes to stop
    stop_event = manager.Event()

    # Initial Q-Network for the learner (transferred as state_dict)
    # This also serves as the initial weights for actors before they sync.
    initial_q_network = QNetwork(config.STATE_DIM, config.ACTION_DIM)
    # Important: Place initial_q_network to CPU before putting state_dict to queue
    # to avoid potential device issues if learner's device is GPU.
    initial_q_network_state_dict = initial_q_network.state_dict()
    
    # Put initial weights into the queue for actors to pick up
    model_weights_queue.put(initial_q_network_state_dict)

    # --- Create and Start Processes ---
    processes = []
    
    # Learner process
    learner_proc = mp.Process(target=learner_fn, name="Learner", 
                              args=(initial_q_network_state_dict, model_weights_queue, experience_queue, stop_event, config))
    processes.append(learner_proc)

    # Actor processes
    # We pass SimpleCarEnv as the class (callable) to env_creator, not an instance,
    # so each actor can create its own instance.
    for i in range(config.NUM_ACTORS):
        actor_proc = mp.Process(target=actor_fn, name=f"Actor-{i}", 
                                args=(i, SimpleCarEnv, model_weights_queue, experience_queue, stop_event, config))
        processes.append(actor_proc)

    for p in processes:
        p.start()
    
    logging.info(f"Main process started {config.NUM_ACTORS} actors and 1 learner.")

    # --- Main Process Monitoring/Termination ---
    start_time = time.time()
    try:
        # The learner might set the stop_event based on total_learner_steps_target.
        # Otherwise, the main process enforces a wall-clock time limit.
        while not stop_event.is_set():
            if time.time() - start_time > config.TOTAL_TRAINING_SECONDS:
                logging.info(f"Main process: Wall-clock time limit ({config.TOTAL_TRAINING_SECONDS}s) reached. Signalling stop.")
                stop_event.set()
            time.sleep(5) # Periodically check stop event and time limit

    except KeyboardInterrupt:
        logging.info("Main process: KeyboardInterrupt detected. Signalling processes to stop.")
        stop_event.set()

    # --- Wait for all processes to finish ---
    logging.info("Main process: Waiting for all processes to join...")
    for p in processes:
        p.join()
    logging.info("Main process: All processes stopped. Training complete.")
    
    # --- Verify Final Model Saved by Learner ---
    final_model_path = "final_dqn_model.pth"
    if os.path.exists(final_model_path):
        logging.info(f"Main process: Final DQN model confirmed saved by learner at {final_model_path}")
        # Optionally: Load and evaluate the model here to verify its content
        # loaded_model = QNetwork(config.STATE_DIM, config.ACTION_DIM)
        # loaded_model.load_state_dict(torch.load(final_model_path, map_location=torch.device('cpu')))
        # loaded_model.eval()
        # logging.info(f"Loaded model example output: {loaded_model(torch.zeros(1, config.STATE_DIM))}")
    else:
        logging.warning("Main process: Final model file not found. Learner might have stopped prematurely or failed to save.")


if __name__ == "__main__":
    # This is crucial for multiprocessing to work correctly on Windows and macOS.
    # 'spawn' creates entirely new, independent processes, avoiding issues with shared state.
    # On Linux, 'fork' is the default and generally works, but 'spawn' is safer cross-platform.
    mp.set_start_method('spawn', force=True) 
    main()
