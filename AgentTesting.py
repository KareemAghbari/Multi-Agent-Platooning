import os
import matplotlib.pyplot as plt
import numpy as np
import torch

from agilerl.algorithms.maddpg import MADDPG

from MultiAgentPlatooningEnv import MultiAgentPlatooningEnv1


N_AGENTS = 6 #Number of agents — must match the saved model
CHECKPOINT_PATH = "maddpg_platoon.pt"


#Creates a test environment using the 20% held-out test car IDs
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


def run_test_episode(checkpoint_path: str, max_steps: int = 5000, seed: int = None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #Load the saved MADDPG checkpoint
    maddpg = MADDPG.load(checkpoint_path, device=device)
    maddpg.set_training_mode(False) #Disable training mode so no gradient updates or exploration noise
    print(f"Loaded checkpoint: {checkpoint_path}")

    #Use a random seed if none provided so each run picks a different test trajectory
    if seed is None:
        seed = int(np.random.randint(0, 10000))
    print(f"Using seed: {seed}")

    env = make_test_env()
    obs, info = env.reset(seed=seed)
    agent_ids = list(obs.keys())

    print(f"Running test episode on car ID: {env.current_car_id}  "
          f"(trajectory length: {env.max_steps} steps)")

    #Creates empty lists to store the time series data
    rewards_ts    = {a: [] for a in agent_ids}
    gaps_ts       = {a: [] for a in agent_ids}
    speeds_ts     = {a: [] for a in agent_ids}
    accels_ts     = {a: [] for a in agent_ids}
    time_gap_ts   = {a: [] for a in agent_ids}
    lead_speed_ts = []

    for step in range(max_steps):
        obs_batched = {a: np.asarray(obs[a], dtype=np.float32)[None, :] for a in obs} #Adds batch dimension because get_action expects batched inputs

        try:
            action, _ = maddpg.get_action(obs=obs_batched, infos=info) #Calls get_action to obtain actions
        except TypeError:
            action, _ = maddpg.get_action(obs=obs_batched)

        action_single = {} #Batched actions for each agent, convert them to scalar values
        for a in action:
            arr = np.asarray(action[a])
            action_single[a] = arr[0].squeeze() if arr.ndim >= 1 else arr

        lead_speed_ts.append(float(env.lead_vx)) #Records lead vehicles current speed

        #Takes a step into the environment with the chosen action, returns all 5 informations
        obs, reward, termination, truncation, info = env.step(action_single)

        #For each agent, append reward, gap, speed, accel and time headway
        for idx, a in enumerate(agent_ids):
            rewards_ts[a].append(float(reward.get(a, 0.0)))
            gaps_ts[a].append(float(info[a].get("dx", 0.0)))
            speeds_ts[a].append(float(obs[a][0] * env.v_max))
            accels_ts[a].append(float(info[a].get("accel", 0.0)))
            time_gap_ts[a].append(float(info[a].get("time_gap", 0.0)))

        if any(termination.values()) or any(truncation.values()):
            terminated_early = any(termination.values())
            print(f"Episode ended at step {step + 1} — "
                  f"{'collision (terminated)' if terminated_early else 'trajectory finished (truncated)'}")
            break

    env.close()

    #Print summary stats for each agent
    print(f"\n── Test Episode Summary (car ID: {env.current_car_id}) ──")
    for a in agent_ids:
        avg_r  = float(np.mean(rewards_ts[a]))
        avg_tg = float(np.mean(time_gap_ts[a]))
        avg_g  = float(np.mean(gaps_ts[a]))
        print(f"  {a:<10}  avg reward: {avg_r:>+7.4f}  "
              f"avg time headway: {avg_tg:>5.2f}s  avg gap: {avg_g:>6.2f}m")

    plot_test_episode(
        rewards_ts, gaps_ts, speeds_ts, accels_ts,
        time_gap_ts, lead_speed_ts, agent_ids,
        car_id=env.current_car_id,
    )

    #Run every test trajectory and print average reward per agent for each car ID
    evaluate_all_test_trajectories(maddpg, max_steps=max_steps)


#Runs one episode for every single test car ID and prints the average reward per agent for each trajectory
def evaluate_all_test_trajectories(maddpg, max_steps: int = 5000):

    #Create a temporary env just to read out the test car IDs then close it
    tmp_env   = make_test_env()
    test_ids  = tmp_env.test_ids.copy()
    agent_ids = tmp_env.possible_agents
    tmp_env.close()

    print(f"\n{'='*72}")
    print(f"  EVALUATING ALL {len(test_ids)} TEST TRAJECTORIES")
    print(f"{'='*72}")
    print(f"  {'Car ID':<10}  {'Steps':>6}  {'End':>12}  " +
          "  ".join(f"{a:>10}" for a in agent_ids))
    print(f"  {'─'*10}  {'─'*6}  {'─'*12}  " +
          "  ".join(f"{'─'*10}" for _ in agent_ids))

    #Stores the average reward for every agent across all test trajectories for the final summary
    all_avg_rewards    = {a: [] for a in agent_ids}
    trajectory_records = []  # stores (car_id, steps, end_reason, avg_rewards) for CSV saving

    for car_id in sorted(test_ids):

        #Create a fresh env and force it to use this specific car ID
        env = make_test_env()
        env.fixed_car_id = int(car_id)
        obs, info = env.reset(seed=0)

        rewards_ts         = {a: [] for a in agent_ids}
        steps_taken        = 0
        ended_by_collision = False

        for step in range(max_steps):
            obs_batched = {a: np.asarray(obs[a], dtype=np.float32)[None, :] for a in obs}

            try:
                action, _ = maddpg.get_action(obs=obs_batched, infos=info)
            except TypeError:
                action, _ = maddpg.get_action(obs=obs_batched)

            action_single = {}
            for a in action:
                arr = np.asarray(action[a])
                action_single[a] = arr[0].squeeze() if arr.ndim >= 1 else arr

            obs, reward, termination, truncation, info = env.step(action_single)

            for a in agent_ids:
                rewards_ts[a].append(float(reward.get(a, 0.0)))

            steps_taken += 1

            if any(termination.values()) or any(truncation.values()):
                ended_by_collision = any(termination.values())
                break

        env.close()

        #Compute average reward per agent for this trajectory
        avg_rewards = {a: float(np.mean(rewards_ts[a])) if rewards_ts[a] else float("nan")
                       for a in agent_ids}

        #Store for the overall summary at the end
        for a in agent_ids:
            if not np.isnan(avg_rewards[a]):
                all_avg_rewards[a].append(avg_rewards[a])

        end_reason = "collision" if ended_by_collision else "completed"
        trajectory_records.append((car_id, steps_taken, end_reason, avg_rewards))  # save for CSV
        print(f"  {car_id:<10}  {steps_taken:>6}  {end_reason:>12}  " +
              "  ".join(f"{avg_rewards[a]:>+10.4f}" for a in agent_ids))

    #Print overall summary across all test trajectories
    print(f"\n{'='*72}")
    print(f"  OVERALL TEST SUMMARY ({len(test_ids)} trajectories)")
    print(f"{'='*72}")
    print(f"  {'Agent':<10}  {'Mean Reward':>12}  {'Std':>8}  {'Min':>8}  {'Max':>8}")
    print(f"  {'─'*10}  {'─'*12}  {'─'*8}  {'─'*8}  {'─'*8}")
    for a in agent_ids:
        vals = all_avg_rewards[a]
        if vals:
            print(f"  {a:<10}  {np.mean(vals):>+12.4f}  {np.std(vals):>8.4f}  "
                  f"{np.min(vals):>+8.4f}  {np.max(vals):>+8.4f}")
    print(f"{'='*72}\n")

    #Save per-trajectory mean rewards to a CSV file — one row per car ID, one column per agent
    csv_path = "test_trajectory_rewards.csv"
    header   = "car_id,steps,end_reason," + ",".join(agent_ids)
    rows     = []
    for car_id, steps, end_reason, avg_r in trajectory_records:
        row = f"{car_id},{steps},{end_reason}," + ",".join(f"{avg_r[a]:.6f}" for a in agent_ids)
        rows.append(row)
    with open(csv_path, "w") as f:
        f.write(header + "\n")
        f.write("\n".join(rows) + "\n")
    print(f"  Saved per-trajectory rewards: {csv_path}")

    #Save a bar chart showing the mean reward per agent averaged across all test trajectories
    save_mean_reward_plot(all_avg_rewards, agent_ids, n_trajectories=len(test_ids))


#Saves a bar chart of mean reward per agent averaged across all test trajectories
def save_mean_reward_plot(all_avg_rewards: dict, agent_ids: list, n_trajectories: int):
    means = [float(np.mean(all_avg_rewards[a])) if all_avg_rewards[a] else float("nan") for a in agent_ids]
    stds  = [float(np.std(all_avg_rewards[a]))  if all_avg_rewards[a] else 0.0          for a in agent_ids]
    x     = np.arange(len(agent_ids))

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=plt.cm.tab10(np.linspace(0, 0.9, len(agent_ids))),
                  edgecolor="black", linewidth=0.6)

    #Label each bar with its mean value
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.005,
                f"{mean:+.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(agent_ids, fontsize=9)
    ax.set_xlabel("Agent")
    ax.set_ylabel("Mean Reward")
    ax.set_title(f"Mean Reward per Agent across All {n_trajectories} Test Trajectories (error bars = std)")
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.grid(axis="y", alpha=0.3)
    path = "test_mean_reward_all_trajectories.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved mean reward plot: {path}")


def plot_test_episode(
    rewards_ts, gaps_ts, speeds_ts, accels_ts,
    time_gap_ts, lead_speed_ts, agent_ids,
    car_id: int,
    save_prefix: str = "test",
):
    n       = len(agent_ids) #Finds number of agent IDs
    colours = plt.cm.tab10(np.linspace(0, 0.9, n)) #Assigns a colour for each ego
    T       = len(lead_speed_ts) #Number of timesteps recorded
    t       = np.arange(T) #Creates an array for the timesteps from 0 to T-1

    #Defines a list of four standard plots
    standard_plots = [
        (rewards_ts, "Reward",               "reward_vs_time.png",  False),
        (gaps_ts,    "Gap [m]",              "gap_vs_time.png",     False),
        (speeds_ts,  "Speed [m/s]",          "speed_vs_time.png",   True),
        (accels_ts,  "Acceleration [m/s^2]", "accel_vs_time.png",   False),
    ]

    for ts_dict, ylabel, fname, add_lead in standard_plots:
        fig, ax = plt.subplots(figsize=(11, 4))
        for a, col in zip(agent_ids, colours):
            arr = np.asarray(ts_dict[a])
            ax.plot(arr, label=a, color=col, linewidth=1.0)
        if add_lead:
            ax.plot(
                t, np.asarray(lead_speed_ts[:T]),
                label="lead (dataset)", color="black",
                linewidth=2.0, linestyle="--", zorder=5,
            )
        ax.set_xlabel("Timestep")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} per Agent — Test Episode (car ID {car_id})")
        ax.legend(loc="upper right", fontsize=7, ncol=2)
        ax.grid(alpha=0.3)
        path = f"{save_prefix}_{fname}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

    # Time headway plot — each agent's time gap to the vehicle directly in front with desired gap line
    fig, ax = plt.subplots(figsize=(11, 4))
    for a, col in zip(agent_ids, colours):
        ax.plot(np.asarray(time_gap_ts[a]), label=a, color=col, linewidth=1.0)
    ax.axhline(1.5, color="red", linewidth=1.2, linestyle="--", label="desired time gap (1.5 s)")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Time Headway [s]")
    ax.set_title(f"Time Headway per Agent — Test Episode (car ID {car_id})")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    path = f"{save_prefix}_time_headway.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


if __name__ == "__main__":
    run_test_episode(
        checkpoint_path=CHECKPOINT_PATH,
        max_steps=5000,
        seed=None,  # None = random test trajectory each run, or set a fixed integer to reproduce
    )