import contextlib
import io
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from agilerl.algorithms.maddpg import MADDPG
from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from agilerl.vector.pz_async_vec_env import AsyncPettingZooVecEnv

from MultiAgentPlatooningEnv import MultiAgentPlatooningEnv1

#Plot for training over time (reward per epiosde)
#Split data into training and testing 80/20
#Plot for time headway
#find average values for all testing trajectories

N_AGENTS  = 6 #Number of agents
NUM_ENVS  = 8 #Number of envs
MAX_EPISODES = 12000 

LOG_INTERVAL = 10 #Prints average stats every 10 episodes  

#Suppresses any stdout noise from AgileRL internals / subprocess workers at the OS fd level
@contextlib.contextmanager
def _suppress_fd_stdout():
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_fd   = os.dup(1)
    os.dup2(devnull_fd, 1)
    try:
        yield
    finally:
        os.dup2(saved_fd, 1)
        os.close(saved_fd)
        os.close(devnull_fd)

#Using the saved env file, mode="train" so it only samples from the 80% training car IDs
def make_env() -> MultiAgentPlatooningEnv1:
    return MultiAgentPlatooningEnv1(
        trajectory_file="VehicleSpeeds(NOMOTORCYCLES).txt",
        num_agents=N_AGENTS,
        dt=0.1,
        desired_time_gap=1.5,
        min_gap=4.0,
        max_gap=90.0,
        v_max=30.0,
        a_max=1.5,
        d_max=1.5,
        vehicle_length=5.0,
        render_mode=None,
        mode="train",
    )

#Same environment config but restricted to the 20% test car IDs so agents never saw these trajectories during training
def make_test_env() -> MultiAgentPlatooningEnv1:
    return MultiAgentPlatooningEnv1(
        trajectory_file="VehicleSpeeds(NOMOTORCYCLES).txt",
        num_agents=N_AGENTS,
        dt=0.1,
        desired_time_gap=1.5,
        min_gap=4.0,
        max_gap=90.0,
        v_max=30.0,
        a_max=1.5,
        d_max=1.5,
        vehicle_length=5.0,
        render_mode=None,
        mode="test",
    )

#This whole thing is requirede by AgileRL to convert NaN in termination/truncation so it doesnt crash
def _nan_to_bool(x, nan_value: bool) -> np.ndarray:
    x = np.asarray(x)
    if x.dtype == object:
        x = x.astype(np.float32)
    if np.issubdtype(x.dtype, np.floating):
        x = np.where(np.isnan(x), float(nan_value), x)
    return x.astype(bool)


#Prints average summary every 10 episodes
def log_episode_stats(
    episode_num: int,
    episode_rewards: dict,
    episode_steps: list,
    window: int = LOG_INTERVAL, #Window is the 10 epiosde interval
) -> None:
    
    if episode_num % window != 0 or episode_num == 0: #Only prints average every 10 episodes
        return

    recent_steps = episode_steps[-window:]
    avg_steps    = float(np.mean(recent_steps)) #Computes average steps from last 10 episodes

    # Gather per-agent stats for the window
    agent_stats = {}
    for aid, rewards in episode_rewards.items():
        recent = rewards[-window:]
        avg_r  = float(np.mean(recent)) if recent else float("nan")
        min_r  = float(np.min(recent))  if recent else float("nan")
        max_r  = float(np.max(recent))  if recent else float("nan")
        agent_stats[aid] = (avg_r, min_r, max_r)

    all_avgs = [v[0] for v in agent_stats.values() if not np.isnan(v[0])]
    fleet_avg = float(np.mean(all_avgs)) if all_avgs else float("nan")

    # header 
    W = 56
    lines = []
    lines.append(f"\n╔{'═' * (W - 2)}╗")
    lines.append(f"║{'TRAINING UPDATE':^{W - 2}}║")
    lines.append(f"╠{'═' * (W - 2)}╣")
    lines.append(f"║  Episodes  {episode_num - window + 1:>5} → {episode_num:<5}   "
                 f"Avg steps/ep: {avg_steps:>7.1f}   ║")
    lines.append(f"║  Fleet avg reward: {fleet_avg:>+8.3f}{' ' * (W - 32)}║")
    lines.append(f"╠{'═' * (W - 2)}╣")

    # column titles 
    lines.append(f"║  {'Agent':<10}  {'Avg Reward':>11}  {'Min':>8}  {'Max':>8}  ║")
    lines.append(f"║  {'─' * 10}  {'─' * 11}  {'─' * 8}  {'─' * 8}  ║")

    # per-agent rows
    for aid, (avg_r, min_r, max_r) in agent_stats.items():
        lines.append(f"║  {aid:<10}  {avg_r:>+11.3f}  {min_r:>+8.3f}  {max_r:>+8.3f}  ║")

    lines.append(f"╚{'═' * (W - 2)}╝")

    tqdm.write("\n".join(lines))


#This function is to run a single evaluation episode to collect data for plotting (Not required for training)
def run_one_episode(agent, make_env_fn, max_steps=5000, seed=123):
    
    eval_env = make_env_fn() #This is to create a single environment instance

    try:
        agent.set_training_mode(False)
    except Exception:
        pass

    if hasattr(agent, "expl_noise"): #This is to set the exploration noise to 0 so that actions are deterministic
        try:
            agent.expl_noise = 0.0
        except Exception:
            pass

    obs, info = eval_env.reset(seed=seed) #Reset environment with given seed
    agent_ids = list(obs.keys())

    #Creates empty lists to store the time series data
    rewards_ts    = {a: [] for a in agent_ids}
    gaps_ts       = {a: [] for a in agent_ids}
    speeds_ts     = {a: [] for a in agent_ids}
    accels_ts     = {a: [] for a in agent_ids}
    pos_ts        = {a: [] for a in agent_ids}
    time_gap_ts   = {a: [] for a in agent_ids}  # time headway (gap / ego speed) for each agent
    lead_speed_ts = []
    lead_pos_ts   = []


    for _ in range(max_steps):
        obs_batched = {a: np.asarray(obs[a], dtype=np.float32)[None, :] for a in obs} #Adds batch dimension becauase get_action expects bateched inputs

        try:
            action, _ = agent.get_action(obs=obs_batched, infos=info) #Calls get_action to obtain actions
        except TypeError:
            action, _ = agent.get_action(obs=obs_batched)

        action_single = {} #Batched actions for each agent, convert them to scalar values
        for a in action:
            arr = np.asarray(action[a])
            action_single[a] = arr[0].squeeze() if arr.ndim >= 1 else arr

        lead_speed_ts.append(float(eval_env.lead_vx)) #Records lead vehicles current speed and position
        lead_pos_ts.append(float(eval_env.lead_x))


        #Takes a step into the environment with the chosen action, returns all 5 informations
        obs, reward, termination, truncation, info = eval_env.step(action_single)

        #For each agent, append reward, gap from info, speeds from observation, accels from info, position and time headway
        for idx, a in enumerate(agent_ids):
            rewards_ts[a].append(float(reward.get(a, 0.0)))
            gaps_ts[a].append(float(info[a].get("dx", 0.0)))
            speeds_ts[a].append(float(obs[a][0] * eval_env.v_max))
            accels_ts[a].append(float(info[a].get("accel", 0.0)))
            pos_ts[a].append(float(eval_env.ego_x[idx]))
            time_gap_ts[a].append(float(info[a].get("time_gap", 0.0)))  # time headway from env info

        if any(termination.values()) or any(truncation.values()):
            break #End episode if any termination or truncation values are found

    eval_env.close() #closes environment and returns the time series data
    return rewards_ts, gaps_ts, speeds_ts, accels_ts, pos_ts, time_gap_ts, lead_speed_ts, lead_pos_ts


#Runs multiple evaluation episodes on test trajectories and returns per-agent average rewards
def run_test_evaluation(agent, make_test_env_fn, n_episodes=20, max_steps=5000):
    agent_ids = None
    all_episode_rewards = {}  # accumulates total reward per episode per agent

    for ep in range(n_episodes):
        rewards_ts, _, _, _, _, _, _, _ = run_one_episode( #This disregards all the other infos and only collects the rewards for each trajectory
            agent, make_test_env_fn, max_steps=max_steps, seed=200 + ep
        )
        if agent_ids is None:
            agent_ids = list(rewards_ts.keys())
            all_episode_rewards = {a: [] for a in agent_ids} #Only for the first episode, initialize the all_episode_rewards dict with empty lists for each agent

        for a in agent_ids:
            all_episode_rewards[a].append(float(np.mean(rewards_ts[a])) if rewards_ts[a] else float("nan")) #For each agent it caluclates the mean reward and appends it to a list

    return all_episode_rewards  # dict: agent_id -> list of mean rewards across test episodes


def plot_results(
    rewards_ts, gaps_ts, speeds_ts, accels_ts,
    pos_ts, time_gap_ts, lead_speed_ts, lead_pos_ts,
    episode_agent_rewards,  # per-agent list of per-episode average rewards recorded during training
    save_prefix="platoon",
):
    agent_ids = list(rewards_ts.keys()) #Extracts agent names
    n         = len(agent_ids) #Finds number of agent IDs
    colours   = plt.cm.tab10(np.linspace(0, 0.9, n)) #Assigns a colour for each ego
    T         = len(lead_speed_ts) #Number of timesteps recorded
    t         = np.arange(T) #Creates an array for the timesteps from 0 to T-1

    #Defines a list of four different plots, containing the data dict, y-axis labels, file namem, and whether to include lead in plot
    standard_plots = [
        (rewards_ts, "Reward",               "reward_vs_time.png",  False),
        (gaps_ts,    "Gap [m]",              "gap_vs_time.png",     False),
        (speeds_ts,  "Speed [m/s]",          "speed_vs_time.png",   True),
        (accels_ts,  "Acceleration [m/s^2]", "accel_vs_time.png",   False),
    ]

    for ts_dict, ylabel, fname, add_lead in standard_plots:
        fig, ax = plt.subplots(figsize=(11, 4))
        for a, col in zip(agent_ids, colours):
            ax.plot(np.asarray(ts_dict[a]), label=a, color=col, linewidth=1.0)
        if add_lead:
            ax.plot(
                t, np.asarray(lead_speed_ts[:T]),
                label="lead (dataset)", color="black",
                linewidth=2.0, linestyle="--", zorder=5,
            )
        ax.set_xlabel("Timestep")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} per Agent - 1 Evaluation Episode")
        ax.legend(loc="upper right", fontsize=7, ncol=2)
        ax.grid(alpha=0.3)
        path = f"{save_prefix}_{fname}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

    # Lead tracking plot
    fig, ax = plt.subplots(figsize=(11, 4))
    lead_arr = np.asarray(lead_speed_ts[:T])
    ego0_arr = np.asarray(speeds_ts["ego_0"][:T])
    err_arr  = ego0_arr - lead_arr

    ax.plot(t, lead_arr, label="lead (dataset)", color="black",
            linewidth=2.0, linestyle="--")
    ax.plot(t, ego0_arr, label="ego_0", color=colours[0], linewidth=1.3)
    ax.fill_between(t[:len(err_arr)], err_arr, alpha=0.18, color=colours[0],
                    label="speed error (ego_0 - lead)")
    ax.axhline(0, color="grey", linewidth=0.6, linestyle=":")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Speed [m/s]")
    ax.set_title("Lead Tracking: ego_0 vs Dataset Lead - 1 Evaluation Episode")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    path = f"{save_prefix}_lead_tracking.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # Time headway plot — shows each ego's time gap to the vehicle directly in front of it over the episode
    fig, ax = plt.subplots(figsize=(11, 4))
    for a, col in zip(agent_ids, colours):
        ax.plot(np.asarray(time_gap_ts[a]), label=a, color=col, linewidth=1.0)
    ax.axhline(1.5, color="red", linewidth=1.2, linestyle="--", label="desired time gap (1.5 s)")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Time Headway [s]")
    ax.set_title("Time Headway per Agent - 1 Evaluation Episode")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    path = f"{save_prefix}_time_headway.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # Reward per episode during training — smoothed with a rolling window to show learning trend per agent
    fig, ax = plt.subplots(figsize=(13, 5))
    smooth_window = max(1, MAX_EPISODES // 200)  # rolling average over ~0.5% of training for readability
    for a, col in zip(agent_ids, colours):
        raw = np.asarray(episode_agent_rewards[a], dtype=np.float32)
        if raw.size == 0:
            continue
        # compute rolling mean using cumsum trick
        kernel      = np.ones(smooth_window) / smooth_window
        smoothed    = np.convolve(raw, kernel, mode="valid")
        ep_axis_raw = np.arange(1, raw.size + 1)
        ep_axis_sm  = np.arange(smooth_window, raw.size + 1)
        ax.plot(ep_axis_raw, raw, alpha=0.15, color=col, linewidth=0.6)        # raw faint line
        ax.plot(ep_axis_sm,  smoothed, color=col, linewidth=1.4, label=a)      # smoothed bold line
    ax.set_xlabel("Episode")
    ax.set_ylabel("Avg Reward per Episode")
    ax.set_title(f"Per-Agent Training Reward over {MAX_EPISODES} Episodes (smoothed window={smooth_window})")
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    path = f"{save_prefix}_training_reward_curve.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")




def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}") #Usually this selects to use my GPU

    env = AsyncPettingZooVecEnv(
        [lambda _fn=make_env: _fn() for _ in range(NUM_ENVS)] #Creates a vectorized environment with 8 copies
    )
    env.reset() #initializes all the environments 

    agent_ids          = list(env.agents) #Gets the list of agents names
    observation_spaces = [env.single_observation_space(a) for a in agent_ids] #Retrieves observation space for each agent
    action_spaces      = [env.single_action_space(a)      for a in agent_ids] #Same for action space


    #Creates a neural network for each agent in agent_ids
    HIDDEN = [128, 128] 
    NET_CONFIG = {
        aid: {
            "encoder_config": {"hidden_size": HIDDEN, "activation": "ReLU"}, #Creates a neural netowrk with 128 neurons each and two hidden layers, uses ReLU activation
            "head_config":    {"hidden_size": [128]}, #Output layer of 128 neurons
        }
        for aid in agent_ids
    }

    #Creates a replay buffer which stores a memory of 200000 transitions, includes state, action, reward, next state and whether the episode is done or not
    memory = MultiAgentReplayBuffer(
        memory_size=200_000,
        field_names=["state", "action", "reward", "next_state", "done"],
        agent_ids=agent_ids,
        device=device,
    )

    #Initializes the maddpg algorithm with observation spaces, action spaces, agent ids, and hyperparameters
    maddpg = MADDPG(
        observation_spaces=observation_spaces,
        action_spaces=action_spaces,
        agent_ids=agent_ids,
        net_config=NET_CONFIG,
        vect_noise_dim=NUM_ENVS, #sets up exploration noise for each environment
        batch_size=256, #Sample 256 transitions when learning
        learn_step=50, #Learning update every 50 steps total between all the environments
        device=device,
    )
    maddpg.set_training_mode(True) #enables training mode (computes gradient, exploration noise becomes active)

    

    
    learning_delay = 2_000 #wait until buffer has at least 2000 samples before it starts to learn
    rollout_len    = 1_000 #number of steps to run before potentially resetting env

    pbar = tqdm(total=MAX_EPISODES, desc="Episodes") #progress bar that updates when episode ends
    completed_episode_scores = []
    total_completed_episodes = 0 #Coutner for completed episodes
    total_learn_updates = 0 #Coutner for how many times the learning networks were updated

    episode_agent_rewards = {a: [] for a in agent_ids} #Creates a list for each agent of how many rewards it earned per episode
    agent_scores          = {a: np.zeros(NUM_ENVS, dtype=np.float32) for a in agent_ids} #Total of each agents reward in current episode for each parallel env
    episode_steps         = [] #List of how manys steps each completed episode lasted
    env_step_counters     = np.zeros(NUM_ENVS, dtype=np.int32) #Counts steps taken in each env since reset


    #This allows the training to continue and keep resetting as long as it doesnt exceed the training limit of 20000 episodes
    while total_completed_episodes < MAX_EPISODES:
        obs, info = env.reset() #resets all environments and gets initial observations
        scores    = np.zeros(NUM_ENVS, dtype=np.float32) #Total  reward per environment 

        for _ in range(rollout_len):

            with _suppress_fd_stdout():
                action, raw_action = maddpg.get_action(obs=obs, infos=info) #Gets action from MADDPG agent for each agent in all the environments
                next_obs, reward, termination, truncation, info = env.step(action) #Takes step using those actions and returns next observation, reward, termination/truncation and info

            
            for a in agent_ids: #Loop over each agent in agent_ids
                r_vec = reward.get(a, np.zeros(NUM_ENVS)) #from reward dict returns by env.step() get reward for agent a in an array with each parallel env
                r_vec = np.where(np.isnan(r_vec), 0.0, r_vec)
                agent_scores[a] += r_vec #add this steps reward to the toal reward for this agent in each env


            #Creates a 1D array containing the total reward for each specific env for the episode
            rew_mat = np.array([
                np.where(np.isnan(reward.get(a, np.zeros(NUM_ENVS))), 
                         0.0, reward.get(a, np.zeros(NUM_ENVS)))
                for a in agent_ids
            ]).T
            scores += rew_mat.sum(axis=-1)

            done = {}
            for a in agent_ids:
                term  = _nan_to_bool(termination.get(a, True),  nan_value=True)
                trunc = _nan_to_bool(truncation.get(a, False),  nan_value=False)
                done[a] = term | trunc #This just states whether the episode is truncated or terminated and returns done for each agent

            memory.save_to_memory( #saves transition to replay buffer
                obs, raw_action, reward, next_obs, done,
                is_vectorised=True,
            )

            #This decides when to provide a learning update
            if len(memory) >= maddpg.batch_size and memory.counter > learning_delay:
                if maddpg.learn_step > NUM_ENVS:
                    k = max(1, maddpg.learn_step // NUM_ENVS)
                    if (maddpg.steps[-1] // NUM_ENVS) % k == 0:
                        maddpg.learn(memory.sample(maddpg.batch_size))
                        total_learn_updates += 1
                else:
                    for _ in range(max(1, NUM_ENVS // maddpg.learn_step)):
                        maddpg.learn(memory.sample(maddpg.batch_size))
                        total_learn_updates += 1

            obs = next_obs
            env_step_counters += 1

            done_mat = np.array([done[a] for a in agent_ids]).T
            reset_noise_idx = []

            for env_i, all_done in enumerate(done_mat.all(axis=1)):
                if not all_done:
                    continue

                total_completed_episodes += 1
                pbar.update(1)   # one episode completed
                completed_episode_scores.append(float(scores[env_i]))
                maddpg.scores.append(float(scores[env_i]))

                for a in agent_ids:
                    episode_agent_rewards[a].append(float(agent_scores[a][env_i]))
                    agent_scores[a][env_i] = 0.0

                episode_steps.append(int(env_step_counters[env_i]))
                env_step_counters[env_i] = 0
                scores[env_i]            = 0.0
                reset_noise_idx.append(env_i)

                log_episode_stats(
                    total_completed_episodes,
                    episode_agent_rewards,
                    episode_steps,
                    window=LOG_INTERVAL,
                )

                if total_completed_episodes >= MAX_EPISODES:
                    break

            if reset_noise_idx:
                maddpg.reset_action_noise(reset_noise_idx)

            maddpg.steps[-1] += NUM_ENVS

            if total_completed_episodes >= MAX_EPISODES:
                break

        # Update tqdm description with recent mean score
        if completed_episode_scores:
            recent_avgs = {
                a: float(np.mean(episode_agent_rewards[a][-10:]))
                for a in agent_ids if episode_agent_rewards[a]
            }
            agent_summary = "  ".join(
                f"{a}: {v:+.2f}" for a, v in recent_avgs.items()
            )
            pbar.set_description(
                f"score={np.mean(completed_episode_scores[-10:]):.2f} | {agent_summary}"
            )
        else:
            pbar.set_description("Collecting initial episodes...")

    pbar.close()
    env.close()

    print(f"\nTotal env steps:          {maddpg.steps[-1]}")
    print(f"Total learn updates:      {total_learn_updates}")
    print(f"Total completed episodes: {total_completed_episodes}")
    print(f"Replay buffer size:       {memory.counter}")


    maddpg.save_checkpoint("maddpg_platoon.pt")
    print("Checkpoint saved: maddpg_platoon.pt")

    print("\nRunning evaluation episode for plotting (test trajectories)...")
    (rewards_ts, gaps_ts, speeds_ts, accels_ts,
     pos_ts, time_gap_ts, lead_speed_ts, lead_pos_ts) = run_one_episode(
        maddpg, make_test_env, max_steps=5000, seed=123,
    )
    plot_results(
        rewards_ts, gaps_ts, speeds_ts, accels_ts,
        pos_ts, time_gap_ts, lead_speed_ts, lead_pos_ts,
        episode_agent_rewards,
        save_prefix="platoon",
    )

    # Run multiple test episodes on unseen trajectories and report average reward per agent
    print("\nEvaluating on test trajectories (20 episodes)...")
    test_rewards = run_test_evaluation(maddpg, make_test_env, n_episodes=20, max_steps=5000)
    print("\n── Test Evaluation Results (20 episodes, unseen trajectories) ──")
    for a, ep_rewards in test_rewards.items():
        valid = [r for r in ep_rewards if not np.isnan(r)]
        avg   = float(np.mean(valid)) if valid else float("nan")
        std   = float(np.std(valid))  if valid else float("nan")
        print(f"  {a:<10}  avg reward: {avg:>+8.4f}  std: {std:>7.4f}  (over {len(valid)} episodes)")


if __name__ == "__main__":
    main()