import os
import matplotlib.pyplot as plt
import numpy as np
import torch

from agilerl.algorithms.maddpg import MADDPG

from MultiAgentPlatooningEnv import MultiAgentPlatooningEnv1


N_AGENTS = 6
CHECKPOINT_PATH = "maddpg_delay_aware4.pt"

# Two scenarios runnable with the current 5-dim checkpoint.
# delay_only and combined are commented out — they require retraining with
# delay_steps=5 so the network input layer matches the 10-dim augmented obs.
# Uncomment them only after retraining with delay_steps=5 in AgentTraining.py.
SCENARIOS = {
  # "baseline":  {"delay_steps": 0, "packet_loss_prob": 0.00},
  #  "loss_only": {"delay_steps": 0, "packet_loss_prob": 0.05},
     "delay_only": {"delay_steps": 5, "packet_loss_prob": 0.00},  
     "combined":   {"delay_steps": 5, "packet_loss_prob": 0.05},  
}


def make_test_env(delay_steps: int = 5, packet_loss_prob: float = 0.0) -> MultiAgentPlatooningEnv1:
    return MultiAgentPlatooningEnv1(
        trajectory_file="VehicleSpeeds(NOMOTORCYCLES).txt",
        num_agents=N_AGENTS,
        dt=0.1,
        desired_time_gap=1.5,
        min_gap=2.0,
        max_gap=90.0,        v_max=30.0,
        a_max=3.0,
        d_max=3.0,
        vehicle_length=5.0,
        render_mode=None,
        mode="test",
        delay_steps=delay_steps,
        packet_loss_prob=packet_loss_prob,
    )


def run_test_episode(checkpoint_path: str, max_steps: int = 5000, seed: int = None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    maddpg = MADDPG.load(checkpoint_path, device=device)
    maddpg.set_training_mode(False)
    print(f"Loaded checkpoint: {checkpoint_path}")

    if seed is None:
        seed = int(np.random.randint(0, 10000))
    print(f"Using seed: {seed}")

    # Single episode plot always uses baseline (no delay, no loss)
    env = make_test_env(delay_steps=5, packet_loss_prob=0.0)
    obs, info = env.reset(seed=seed)
    agent_ids = list(obs.keys())

    print(f"Running test episode on car ID: {env.current_car_id}  "
          f"(trajectory length: {env.max_steps} steps)")

    rewards_ts    = {a: [] for a in agent_ids}
    gaps_ts       = {a: [] for a in agent_ids}
    speeds_ts     = {a: [] for a in agent_ids}
    accels_ts     = {a: [] for a in agent_ids}
    time_gap_ts   = {a: [] for a in agent_ids}
    lead_speed_ts = []

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

        if step >= 30:  # skip first 5 timesteps from all recordings
            lead_speed_ts.append(float(env.lead_vx))

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

    print(f"\n── Test Episode Summary (car ID: {env.current_car_id}) ──")
    for a in agent_ids:
        avg_r  = float(np.mean(rewards_ts[a])) if rewards_ts[a] else float("nan")
        avg_tg = float(np.mean(time_gap_ts[a])) if time_gap_ts[a] else float("nan")
        avg_g  = float(np.mean(gaps_ts[a])) if gaps_ts[a] else float("nan")
        print(f"  {a:<10}  avg reward: {avg_r:>+7.4f}  "
              f"avg time headway: {avg_tg:>5.2f}s  avg gap: {avg_g:>6.2f}m")

    plot_test_episode(
        rewards_ts, gaps_ts, speeds_ts, accels_ts,
        time_gap_ts, lead_speed_ts, agent_ids,
        car_id=env.current_car_id,
    )

    evaluate_all_test_trajectories(maddpg, max_steps=max_steps)


def evaluate_all_test_trajectories(maddpg, max_steps: int = 5000):
    # Runs every test car ID under each scenario and prints a comparison table.
    # This is the main way to measure how much delay and packet loss hurt performance.

    tmp_env   = make_test_env()
    test_ids  = tmp_env.test_ids.copy()
    agent_ids = tmp_env.possible_agents
    tmp_env.close()

    # Accumulate per-scenario results: scenario_name -> agent_id -> list of avg rewards
    scenario_results    = {name: {a: [] for a in agent_ids} for name in SCENARIOS}
    scenario_collisions = {name: 0 for name in SCENARIOS}  # count how many trajectories end in collision per scenario

    for scenario_name, scenario_kwargs in SCENARIOS.items():

        print(f"\n{'='*72}")
        print(f"  SCENARIO: {scenario_name}  "
              f"(delay_steps={scenario_kwargs['delay_steps']}, "
              f"packet_loss_prob={scenario_kwargs['packet_loss_prob']})")
        print(f"  EVALUATING ALL {len(test_ids)} TEST TRAJECTORIES")
        print(f"{'='*72}")
        print(f"  {'Car ID':<10}  {'Steps':>6}  {'End':>12}  " +
              "  ".join(f"{a:>10}" for a in agent_ids))
        print(f"  {'─'*10}  {'─'*6}  {'─'*12}  " +
              "  ".join(f"{'─'*10}" for _ in agent_ids))

        trajectory_records = []

        for car_id in sorted(test_ids):

            env = make_test_env(**scenario_kwargs)
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

                if step >= 30:  # skip first 30 timesteps from recordings
                    for a in agent_ids:
                        rewards_ts[a].append(float(reward.get(a, 0.0)))

                steps_taken += 1

                if any(termination.values()) or any(truncation.values()):
                    ended_by_collision = any(termination.values())
                    break

            env.close()

            avg_rewards = {a: float(np.mean(rewards_ts[a])) if rewards_ts[a] else float("nan")
                           for a in agent_ids}

            for a in agent_ids:
                if not np.isnan(avg_rewards[a]):
                    scenario_results[scenario_name][a].append(avg_rewards[a])

            if ended_by_collision:
                scenario_collisions[scenario_name] += 1

            end_reason = "collision" if ended_by_collision else "completed"
            trajectory_records.append((car_id, steps_taken, end_reason, avg_rewards))
            print(f"  {car_id:<10}  {steps_taken:>6}  {end_reason:>12}  " +
                  "  ".join(f"{avg_rewards[a]:>+10.4f}" for a in agent_ids))

        # Per-scenario summary
        print(f"\n  Summary for {scenario_name}:")
        print(f"  {'Agent':<10}  {'Mean Reward':>12}  {'Std':>8}  {'Min':>8}  {'Max':>8}")
        print(f"  {'─'*10}  {'─'*12}  {'─'*8}  {'─'*8}  {'─'*8}")
        for a in agent_ids:
            vals = scenario_results[scenario_name][a]
            if vals:
                print(f"  {a:<10}  {np.mean(vals):>+12.4f}  {np.std(vals):>8.4f}  "
                      f"{np.min(vals):>+8.4f}  {np.max(vals):>+8.4f}")

        # Save per-scenario CSV
        csv_path = f"test_trajectory_rewards_{scenario_name}.csv"
        header   = "car_id,steps,end_reason," + ",".join(agent_ids)
        rows     = []
        for car_id, steps, end_reason, avg_r in trajectory_records:
            row = f"{car_id},{steps},{end_reason}," + ",".join(f"{avg_r[a]:.6f}" for a in agent_ids)
            rows.append(row)
        with open(csv_path, "w") as f:
            f.write(header + "\n")
            f.write("\n".join(rows) + "\n")
        print(f"  Saved: {csv_path}")

    # Final cross-scenario comparison table — fleet average reward and collision count per scenario
    print(f"\n{'='*72}")
    print(f"  CROSS-SCENARIO COMPARISON")
    print(f"{'='*72}")
    print(f"  {'Scenario':<14}  {'Fleet Avg Reward':>18}  {'Collisions':>12}")
    print(f"  {'─'*14}  {'─'*18}  {'─'*12}")
    for scenario_name in SCENARIOS:
        all_vals  = [v for a in agent_ids for v in scenario_results[scenario_name][a]]
        fleet_avg = float(np.mean(all_vals)) if all_vals else float("nan")
        collisions = scenario_collisions[scenario_name]
        print(f"  {scenario_name:<14}  {fleet_avg:>+18.4f}  {collisions:>12}")
    print(f"{'='*72}\n")

    save_scenario_comparison_plot(scenario_results, agent_ids)


def save_scenario_comparison_plot(scenario_results: dict, agent_ids: list):
    # Bar chart comparing mean reward per agent across all scenarios side by side
    n_agents    = len(agent_ids)
    n_scenarios = len(SCENARIOS)
    x           = np.arange(n_agents)
    width       = 0.8 / n_scenarios  # divide bar width evenly among scenarios
    colours     = plt.cm.tab10(np.linspace(0, 0.6, n_scenarios))

    fig, ax = plt.subplots(figsize=(13, 5))

    for s_idx, (scenario_name, agent_rewards) in enumerate(scenario_results.items()):
        means = [float(np.mean(agent_rewards[a])) if agent_rewards[a] else float("nan")
                 for a in agent_ids]
        stds  = [float(np.std(agent_rewards[a]))  if agent_rewards[a] else 0.0
                 for a in agent_ids]
        offset = (s_idx - n_scenarios / 2 + 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=3,
               label=scenario_name, color=colours[s_idx],
               edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(agent_ids, fontsize=9)
    ax.set_xlabel("Agent")
    ax.set_ylabel("Mean Reward per Timestep")
    ax.set_title("Scenario Comparison: Baseline vs Packet Loss")
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    path = "test_scenario_comparison.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved scenario comparison plot: {path}")


def save_mean_reward_plot(all_avg_rewards: dict, agent_ids: list, n_trajectories: int):
    means = [float(np.mean(all_avg_rewards[a])) if all_avg_rewards[a] else float("nan") for a in agent_ids]
    stds  = [float(np.std(all_avg_rewards[a]))  if all_avg_rewards[a] else 0.0          for a in agent_ids]
    x     = np.arange(len(agent_ids))

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=plt.cm.tab10(np.linspace(0, 0.9, len(agent_ids))),
                  edgecolor="black", linewidth=0.6)

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
    n       = len(agent_ids)
    colours = plt.cm.tab10(np.linspace(0, 0.9, n))
    T       = len(lead_speed_ts)
    t       = np.arange(T)

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
        if ylabel == "Reward":
            ax.set_ylim(0.0, 0.5)  # reward is clipped to [-1, 1] but max achievable is 0.5 (gaussian peak)
        ax.set_xlabel("Timestep")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} per Agent — Test Episode (car ID {car_id})")
        ax.legend(loc="upper right", fontsize=7, ncol=2)
        ax.grid(alpha=0.3)
        path = f"{save_prefix}_{fname}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

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
        seed=None,
    )