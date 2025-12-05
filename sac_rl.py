#SAC
import os
import time
import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.wrappers import FlattenObservation

from rl import DymonopolyEnv   


#  1. Paths
LOG_DIR = "logs/"
MODEL_DIR = "models/"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model")
LAST_MODEL_PATH = os.path.join(MODEL_DIR, "last_model")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


#  2. Helper: Create wrapped environment
def make_env():
    env = DymonopolyEnv(num_properties=40, num_players=4)
    env = Monitor(env)                                 
    env = FlattenObservation(env)                      
    return env


# Create vectorized environment
env = DummyVecEnv([make_env])
env = VecNormalize(env, norm_obs=True, norm_reward=True)


#  3. Load existing model if available
if os.path.exists(LAST_MODEL_PATH + ".zip"):
    print("\n Previous model found — resuming training...\n")
    model = SAC.load(
        LAST_MODEL_PATH,
        env=env,
        device="cuda",
        print_system_info=True,
    )
else:
    print("\nStarting NEW SAC training...\n")
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=1e-4,           # slower LR for stability
        batch_size=512,               # larger batch reduces variance
        buffer_size=200000,
        tau=0.005,                    # softer target updates
        gamma=0.99,
        train_freq=4,                 # train every 4 steps
        gradient_steps=4,             # match with train_freq
        learning_starts=2000,         # fill buffer before learning
        ent_coef="auto_0.1",          # auto-tuned entropy
        verbose=1,
        tensorboard_log=LOG_DIR,
        device="cuda",
    )


#  4. Callbacks (Checkpoints + Best Model)
eval_env = DummyVecEnv([make_env])
eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=True)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=BEST_MODEL_PATH,
    log_path=LOG_DIR,
    eval_freq=10000,
    deterministic=True,
    render=False
)

checkpoint_callback = CheckpointCallback(
    save_freq=50000,
    save_path=MODEL_DIR,
    name_prefix="checkpoint"
)


#  5. Training
TOTAL_STEPS = 1_000_000   # 1 million steps for thorough training

print(" TRAINING STARTED")
print("---------------------------------------------------")
print(f"Total steps: {TOTAL_STEPS:,}")
print("Logs:   logs/")
print("Models: models/")

start_time = time.time()

try:
    model.learn(
        total_timesteps=TOTAL_STEPS,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )

except KeyboardInterrupt:
    print("\n Training interrupted by user — saving last model...\n")

finally:
    # Save progress no matter what
    model.save(LAST_MODEL_PATH)
    env.save(os.path.join(MODEL_DIR, "vec_normalize.pkl"))
    end_time = time.time()

    print(" TRAINING Ended")
    print(" Training session saved.")
    print(f" Total time: {(end_time - start_time)/60:.2f} min")
    print(" Last model saved to:", LAST_MODEL_PATH)