import numpy as np
import matplotlib.pyplot as plt
import torch

from agilerl.algorithms.maddpg import MADDPG
from MultiAgentPlatooningEnv import MultiAgentPlatooningEnv1


CAR_ID          = 202          # Set this to whichever car ID you want to debug
CHECKPOINT_PATH = "maddpg_delay_aware2.pt"
DELAY_STEPS     = 5            # Must match what the checkpoint was trained with
PACKET_LOSS     = 0.05          # Set to 0.05 to also test packet loss on this trajectory
MAX_STEPS       = 5000


def make_env(car_id: int, delay_steps: int, packet_loss_prob: float) -> MultiAgentPlatooningEnv1:
    # Creates the test environment locked to a specific car ID so you always
    # run the exact same trajectory regardless of random sampling
    env = MultiAgentPlatooningEnv1(
        trajectory_file="VehicleSpeeds(NOMOTORCYCLES).txt",
        num_agents=6,
        dt=0.1,
        desired_time_gap=1.5,
        min_gap=2.0,
        max_gap=90.0,
        v_max=30.0,
        a_max=3.0,
        d_max=3.0,
        vehicle_length=5.0,
        render_mode=None,
        mode="test",
        delay_steps=delay_steps,
        packet_loss_prob=packet_loss_prob,
    )
    # Lock the environment to this specific car ID so it never samples a different one
    env.fixed_car_id = int(car_id)
    return env


def run_single_trajectory(car_id: int, checkpoint_path: str,
                           delay_steps: int, packet_loss_prob: float,
                           max_steps: int):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Load the trained policy
    maddpg = MADDPG.load(checkpoint_path, device=device)
    maddpg.set_training_mode(False)
    print(f"Loaded checkpoint: {checkpoint_path}")

    # Build the environment locked to the chosen car ID
    env = make_env(car_id, delay_steps, packet_loss_prob)
    obs, info = env.reset(seed=0)
    agent_ids = list(obs.keys())

    traj_len = env.max_steps
    print(f"\nCar ID:            {car_id}")
    print(f"Trajectory length: {traj_len} steps  ({traj_len * env.dt:.1f} seconds)")
    print(f"Lead start speed:  {env.lead_vx:.2f} m/s")
    print(f"Delay steps:       {delay_steps}  ({delay_steps * env.dt:.1f}s delay)")
    print(f"Packet loss:       {packet_loss_prob * 100:.0f}%")
    print(f"{'─' * 60}")

    # Storage for every metric we want to plot
    rewards_ts   = {a: [] for a in agent_ids}
    gaps_ts      = {a: [] for a in agent_ids}
    speeds_ts    = {a: [] for a in agent_ids}
    accels_ts    = {a: [] for a in agent_ids}
    time_gap_ts  = {a: [] for a in agent_ids}
    ttc_ts       = {a: [] for a in agent_ids}   # time to collision each step
    rel_v_ts     = {a: [] for a in agent_ids}   # relative velocity to vehicle in front
    lead_speed_ts = []

    collision_step = None   # records the step a collision occurred if any
    end_reason     = "trajectory completed"

    for step in range(max_steps):

        # Add batch dimension because get_action expects batched inputs
        obs_batched = {a: np.asarray(obs[a], dtype=np.float32)[None, :] for a in obs}

        try:
            action, _ = maddpg.get_action(obs=obs_batched, infos=info)
        except TypeError:
            action, _ = maddpg.get_action(obs=obs_batched)

        # Remove the batch dimension from each action before passing to env.step
        action_single = {}
        for a in action:
            arr = np.asarray(action[a])
            action_single[a] = arr[0].squeeze() if arr.ndim >= 1 else arr

        obs, reward, termination, truncation, info = env.step(action_single)

        # Record lead vehicle speed
        lead_speed_ts.append(float(env.lead_vx))

        # Record per-agent metrics
        for idx, a in enumerate(agent_ids):
            rewards_ts[a].append(float(reward.get(a, 0.0)))
            gaps_ts[a].append(float(info[a].get("dx", 0.0)))
            speeds_ts[a].append(float(obs[a][0] * env.v_max))
            accels_ts[a].append(float(info[a].get("accel", 0.0)))
            time_gap_ts[a].append(float(info[a].get("time_gap", 0.0)))
            rel_v_ts[a].append(float(info[a].get("rel_v", 0.0)))

            # Compute TTC manually: gap / closing speed. Only meaningful when closing (rel_v < 0)
            dx    = float(info[a].get("dx", 0.0))
            rel_v = float(info[a].get("rel_v", 0.0))
            if rel_v < -1e-3:
                # Closing — TTC = gap / closing speed (positive number, smaller = more dangerous)
                ttc = dx / abs(rel_v)
                ttc_ts[a].append(min(ttc, 10.0))   # cap at 10s for readability
            else:
                # Not closing — no danger, use 10s as a "safe" placeholder
                ttc_ts[a].append(10.0)

        # Check if episode ended
        if any(termination.values()) or any(truncation.values()):
            if any(termination.values()):
                collision_step = step + 1
                end_reason     = f"COLLISION at step {collision_step}"
            else:
                end_reason = f"trajectory completed at step {step + 1}"
            break

    env.close()
    steps_recorded = len(lead_speed_ts)

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"\nResult: {end_reason}")
    print(f"Steps recorded: {steps_recorded}")
    print(f"\n{'Agent':<10}  {'Avg Reward':>11}  {'Avg Gap':>9}  {'Avg Time Gap':>13}  {'Min Gap':>8}  {'Min TTC':>8}")
    print(f"{'─'*10}  {'─'*11}  {'─'*9}  {'─'*13}  {'─'*8}  {'─'*8}")
    for a in agent_ids:
        avg_r   = float(np.mean(rewards_ts[a]))   if rewards_ts[a]  else float("nan")
        avg_g   = float(np.mean(gaps_ts[a]))      if gaps_ts[a]     else float("nan")
        avg_tg  = float(np.mean(time_gap_ts[a]))  if time_gap_ts[a] else float("nan")
        min_g   = float(np.min(gaps_ts[a]))       if gaps_ts[a]     else float("nan")
        min_ttc = float(np.min(ttc_ts[a]))        if ttc_ts[a]      else float("nan")
        print(f"{a:<10}  {avg_r:>+11.4f}  {avg_g:>9.2f}m  {avg_tg:>12.3f}s  {min_g:>7.2f}m  {min_ttc:>7.2f}s")

    if collision_step:
        print(f"\n⚠  Collision occurred at step {collision_step}")
        print(f"   At that step the following gaps were:")
        # Print gap at the collision step for each agent
        for a in agent_ids:
            if len(gaps_ts[a]) >= collision_step:
                g = gaps_ts[a][collision_step - 1]
                print(f"   {a}: {g:.3f}m")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_single_trajectory(
        car_id        = car_id,
        agent_ids     = agent_ids,
        rewards_ts    = rewards_ts,
        gaps_ts       = gaps_ts,
        speeds_ts     = speeds_ts,
        accels_ts     = accels_ts,
        time_gap_ts   = time_gap_ts,
        ttc_ts        = ttc_ts,
        rel_v_ts      = rel_v_ts,
        lead_speed_ts = lead_speed_ts,
        collision_step= collision_step,
        delay_steps   = delay_steps,
    )


def plot_single_trajectory(
    car_id, agent_ids, rewards_ts, gaps_ts, speeds_ts,
    accels_ts, time_gap_ts, ttc_ts, rel_v_ts,
    lead_speed_ts, collision_step, delay_steps,
):
    n       = len(agent_ids)
    colours = plt.cm.tab10(np.linspace(0, 0.9, n))
    T       = len(lead_speed_ts)
    t       = np.arange(T)

    # Helper: draw a vertical red line at the collision step on any axis
    def mark_collision(ax):
        if collision_step and collision_step <= T:
            ax.axvline(collision_step, color="red", linewidth=1.5,
                       linestyle="--", label="collision")

    fig, axes = plt.subplots(6, 1, figsize=(13, 22), sharex=True)
    fig.suptitle(
        f"Single Trajectory Debug — Car ID {car_id}  "
        f"(delay={delay_steps} steps = {delay_steps * 0.1:.1f}s)",
        fontsize=13, fontweight="bold"
    )

    # ── Plot 1: Reward ────────────────────────────────────────────────────────
    ax = axes[0]
    for a, col in zip(agent_ids, colours):
        ax.plot(rewards_ts[a], label=a, color=col, linewidth=1.0)
    ax.axhline(0.5, color="grey", linewidth=0.8, linestyle=":", label="max reward (0.5)")
    mark_collision(ax)
    ax.set_ylabel("Reward")
    ax.set_title("Reward per Agent — drops indicate the agent is too far, too close, or accelerating hard")
    ax.legend(fontsize=7, ncol=4)
    ax.grid(alpha=0.3)

    # ── Plot 2: Gap ───────────────────────────────────────────────────────────
    ax = axes[1]
    for a, col in zip(agent_ids, colours):
        ax.plot(gaps_ts[a], label=a, color=col, linewidth=1.0)
    ax.axhline(2.0, color="red",    linewidth=1.2, linestyle="--", label="min_gap (2m)")
    ax.axhline(7.0, color="orange", linewidth=0.8, linestyle=":",  label="initial gap floor (7m)")
    mark_collision(ax)
    ax.set_ylabel("Gap [m]")
    ax.set_title("Bumper-to-Bumper Gap — below the red line triggers termination")
    ax.legend(fontsize=7, ncol=4)
    ax.grid(alpha=0.3)

    # ── Plot 3: Speed ─────────────────────────────────────────────────────────
    ax = axes[2]
    for a, col in zip(agent_ids, colours):
        ax.plot(speeds_ts[a], label=a, color=col, linewidth=1.0)
    ax.plot(t, np.asarray(lead_speed_ts[:T]), color="black",
            linewidth=2.0, linestyle="--", label="lead (dataset)", zorder=5)
    mark_collision(ax)
    ax.set_ylabel("Speed [m/s]")
    ax.set_title("Speed — egos should track the lead vehicle closely")
    ax.legend(fontsize=7, ncol=4)
    ax.grid(alpha=0.3)

    # ── Plot 4: Acceleration ──────────────────────────────────────────────────
    ax = axes[3]
    for a, col in zip(agent_ids, colours):
        ax.plot(accels_ts[a], label=a, color=col, linewidth=1.0)
    ax.axhline(0, color="grey", linewidth=0.6, linestyle=":")
    mark_collision(ax)
    ax.set_ylabel("Accel [m/s²]")
    ax.set_title("Executed Acceleration — large swings indicate the agent is overcorrecting")
    ax.legend(fontsize=7, ncol=4)
    ax.grid(alpha=0.3)

    # ── Plot 5: Time Headway ──────────────────────────────────────────────────
    ax = axes[4]
    for a, col in zip(agent_ids, colours):
        ax.plot(time_gap_ts[a], label=a, color=col, linewidth=1.0)
    ax.axhline(1.5, color="red", linewidth=1.2, linestyle="--", label="desired (1.5s)")
    ax.axhline(0.5, color="darkred", linewidth=0.8, linestyle=":", label="danger zone (<0.5s)")
    mark_collision(ax)
    ax.set_ylabel("Time Headway [s]")
    ax.set_title("Time Headway — should stay near 1.5s, below 0.5s is very dangerous")
    ax.legend(fontsize=7, ncol=4)
    ax.grid(alpha=0.3)

    # ── Plot 6: TTC ───────────────────────────────────────────────────────────
    ax = axes[5]
    for a, col in zip(agent_ids, colours):
        ax.plot(ttc_ts[a], label=a, color=col, linewidth=1.0)
    ax.axhline(1.5, color="red",    linewidth=1.2, linestyle="--", label="TTC safe threshold (1.5s)")
    ax.axhline(0.5, color="darkred",linewidth=0.8, linestyle=":",  label="critical TTC (<0.5s)")
    ax.set_ylim(0, 10.5)
    mark_collision(ax)
    ax.set_ylabel("TTC [s]")
    ax.set_xlabel("Timestep")
    ax.set_title("Time to Collision — below the red line means closing too fast, 10s = not closing at all")
    ax.legend(fontsize=7, ncol=4)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = f"debug_car_{car_id}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved: {path}")


if __name__ == "__main__":
    run_single_trajectory(
        car_id          = CAR_ID,
        checkpoint_path = CHECKPOINT_PATH,
        delay_steps     = DELAY_STEPS,
        packet_loss_prob= PACKET_LOSS,
        max_steps       = MAX_STEPS,
    )