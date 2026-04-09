import os
from collections import deque
from typing import Optional, Tuple

import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv

from Communication_Channel import ActionBuffer, PacketLossChannel


class MultiAgentPlatooningEnv1(ParallelEnv):
    

    metadata = {"name": "MultiAgentPlatooningEnv1", "render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        trajectory_file: str = "VehicleSpeeds(NOMOTORCYCLES).txt",
        num_agents: int = 2,
        car_id: Optional[int] = None,
        dt: float = 0.1,
        desired_time_gap: float = 1.5,
        min_gap: float = 2.0,
        max_gap: float = 90.0,
        v_max: float = 30.0,
        a_max: float = 3.0,
        d_max: float = 3.0,
        vehicle_length: float = 5.0,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        mode: str = "train",  # "train" uses 80% of car IDs, "test" uses the remaining 20%
        delay_steps: int = 0,          # k: number of timesteps of communication delay. 0 = no delay (baseline)
        packet_loss_prob: float = 0.0, # probability that a V2V observation packet is lost. 0.0 = no loss (baseline)
    ):
        super().__init__()

        self.render_mode = render_mode
        self.dt = float(dt)
        self.desired_time_gap = float(desired_time_gap)
        self.min_gap_dist = float(min_gap)
        self.max_gap_dist = float(max_gap)
        self.v_max = float(v_max)
        self.a_max = float(a_max)
        self.d_max = float(d_max)
        self.vehicle_length = float(vehicle_length)
        self.fixed_car_id = car_id
        self._n_agents = int(num_agents)
        self.mode = mode  #controls whether to sample from train or test car IDs
        self.delay_steps = int(delay_steps)
        self.packet_loss_prob = float(packet_loss_prob)

        self.possible_agents = [f"ego_{i}" for i in range(self._n_agents)] #Creates a list with ego cars
        self.agents: list[str] = []

        # Base 5-dim physical observation bounds
        base_obs_low  = np.array([0.0,  0.0, -0.2, -1.0,  0.0, -1.0], dtype=np.float32)
        base_obs_high = np.array([1.0,  1.0,  1.0,  1.0,  2.0,  1.0], dtype=np.float32)

        # If delay is active, append k extra dims for the pending action queue.
        # Each pending action is normalised to [-1, 1] by dividing by a_max.
        # If delay_steps=0 the observation stays 5-dim and behaves exactly as before.
        if self.delay_steps > 0:
            pending_low  = np.full(self.delay_steps, -1.0, dtype=np.float32) #Creates arrays for the pending action queue
            pending_high = np.full(self.delay_steps,  1.0, dtype=np.float32)
            obs_low  = np.concatenate([base_obs_low,  pending_low]) #Adds the pending queue action observations to the observation space
            obs_high = np.concatenate([base_obs_high, pending_high])
        else:
            obs_low  = base_obs_low
            obs_high = base_obs_high

        # 5-dim observation (or 5+k if delay is active)
        self._observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self._action_space = spaces.Box(
            low= np.array([-self.d_max], dtype=np.float32),
            high=np.array([ self.a_max], dtype=np.float32),
            dtype=np.float32,
        )

        self.action_spaces      = {a: self._action_space      for a in self.possible_agents} #Dicts for each agent
        self.observation_spaces = {a: self._observation_space for a in self.possible_agents}

        self._rng = np.random.default_rng(seed)
        self.all_car_ids, self.all_speeds, self.unique_car_ids = self._load_dataset(trajectory_file)

        # Split car IDs 80/20 into train and test sets using a fixed seed so the split is always the same
        rng_split = np.random.default_rng(42) #Creates random number generator with fixed seed 42
        shuffled  = self.unique_car_ids.copy() #Creates a copy of all the valid car ids
        rng_split.shuffle(shuffled) #Shuffles the car ids in place using the fixed seed, so the split is always the same across runs
        split_idx        = int(len(shuffled) * 0.8) #Calculates the index to split the car ids into 80% for training and 20% for testing
        self.train_ids   = shuffled[:split_idx]   # 80% of car IDs reserved for training
        self.test_ids    = shuffled[split_idx:]    # 20% of car IDs reserved for testing

        # Build one ActionBuffer per agent and one shared PacketLossChannel.
        # Created once here and only reset (zeroed) each episode in _reset_state(),
        # never recreated, so AgileRL always sees the same objects and correct obs shape.
        self.action_buffers = [
            ActionBuffer(self.delay_steps) for _ in range(self._n_agents)
        ]
        self.packet_channel = PacketLossChannel(self.packet_loss_prob, self._rng)

        # Internal state
        self._reset_state()

    @property
    def unwrapped(self):
        return self

    def observation_space(self, agent: str) -> spaces.Box:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Box:
        return self.action_spaces[agent]

    def _load_dataset(self, trajectory_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(base_dir, trajectory_file)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Trajectory file not found: {full_path}")

        car_ids: list[int] = []
        speeds: list[float] = []

        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.replace(",", " ").split()
                if len(parts) < 2:
                    continue
                try:
                    cid = int(float(parts[0]))
                except Exception:
                    continue
                speed_idx = 4 if len(parts) > 4 else 1
                try:
                    spd = float(parts[speed_idx])
                except Exception:
                    continue
                car_ids.append(cid)
                speeds.append(spd)

        car_ids_arr = np.asarray(car_ids, dtype=int)
        speeds_arr  = np.asarray(speeds,   dtype=np.float32)

        valid       = speeds_arr >= 3.0
        car_ids_arr = car_ids_arr[valid]
        speeds_arr  = speeds_arr[valid]

        unique_ids = np.unique(car_ids_arr)
        valid_ids  = [cid for cid in unique_ids
                      if np.unique(speeds_arr[car_ids_arr == cid]).size >= 200]
        valid_ids  = np.asarray(valid_ids, dtype=int)
        mask       = np.isin(car_ids_arr, valid_ids)

        return car_ids_arr[mask], speeds_arr[mask], valid_ids

    def _reset_state(self) -> None:
        n = self._n_agents
        self.lead_x:  float = 0.0
        self.lead_vx: float = 0.0
        self.ego_x   = np.zeros(n, dtype=np.float32)
        self.ego_vx  = np.zeros(n, dtype=np.float32)
        self.last_accels = np.zeros(n, dtype=np.float32)   # for external info
        self.prev_accels = np.zeros(n, dtype=np.float32)   # for jerk calculation
        self.prev_front_vx = np.zeros(n, dtype=np.float32)
        self.steps:      int = 0
        self.traj_index: int = 0
        self.max_steps:  int = 0
        self.current_car_id: Optional[int] = None
        self.lead_speeds: Optional[np.ndarray] = None
        # Reset action buffers and packet channel each episode so there are no
        # stale actions or stale cached observations carried over from the last trajectory
        for buf in self.action_buffers:
            buf.reset()
        self.packet_channel.reset()

    def _front_x(self, i: int) -> float:
        return float(self.lead_x) if i == 0 else float(self.ego_x[i - 1]) # Identifies which car is in front of chosen agent

    def _front_v(self, i: int) -> float:
        return float(self.lead_vx) if i == 0 else float(self.ego_vx[i - 1]) # Same but for speeds

    def _dx(self, i: int) -> float:
        return self._front_x(i) - (float(self.ego_x[i]) + self.vehicle_length) # Takes distance gap using bumper length

    def _rel_v(self, i: int) -> float:
        return self._front_v(i) - float(self.ego_vx[i]) #takes relative speed of agent in front of another

    def _time_gap(self, i: int) -> float:
        v_safe = max(float(self.ego_vx[i]), 1e-3)
        return self._dx(i) / v_safe

    def _obs_i(self, i: int) -> np.ndarray:
        ego_v   = float(self.ego_vx[i])
        front_v = self._front_v(i)
        dx      = self._dx(i)

        dx_norm     = float(np.clip(dx / self.max_gap_dist, -0.2, 1.0))
        rel_v_norm  = float(np.clip((front_v - ego_v) / self.v_max, -1.0, 1.0))
        speed_ratio = float(np.clip(ego_v / max(front_v, 1e-3), 0.0, 2.0))

        front_accel = (front_v - float(self.prev_front_vx[i])) / self.dt
        front_accel_norm = float(np.clip(front_accel / self.a_max, -1.0, 1.0))

        return np.array(
            [ego_v / self.v_max, front_v / self.v_max, dx_norm, rel_v_norm, speed_ratio, front_accel_norm],
            dtype=np.float32,
        )

    def _augment_obs(self, i: int, base_obs: np.ndarray) -> np.ndarray:
        # Forms the augmented state o_i = (o_obs, o_act) from the DAMARL paper.
        # o_obs is the 5-dim physical observation (possibly after packet loss fallback).
        # o_act is the flattened pending action queue, normalised to [-1, 1].
        # When delay_steps=0 there is no queue so base_obs is returned unchanged —
        # this means delay=0 is identical to the original baseline behaviour.
        if self.delay_steps == 0:
            return base_obs
        pending = self.action_buffers[i].get_pending_sequence()
        pending_norm = np.clip(pending / self.a_max, -1.0, 1.0)  # normalise to [-1, 1] to match obs scale
        return np.concatenate([base_obs, pending_norm])

    def reward(self, i: int, accel: float) -> float:
        
            dx = self._dx(i)
            if dx < self.min_gap_dist:
                return -1.0

            r = 0.0

            # far_margin   = 10.0

            # if dx < self.min_gap_dist + close_margin:
            #     depth = 1.0 - (dx - self.min_gap_dist) / close_margin #Higher depth means more negative reward
            #     r -= 0.8 * float(depth)
            # elif dx > self.max_gap_dist - far_margin:
            #     depth = (dx - (self.max_gap_dist - far_margin)) / far_margin #Same thing but if egos get too far behind
            #     r -= 0.8 * float(depth)
            # else:
            #     r += 0.1  # survival bonus

            #Soft proximity penalty that ramps up as gap approaches minimum safe distance
            close_margin = 3.0
            if dx < self.min_gap_dist + close_margin:
                depth = 1.0 - (dx - self.min_gap_dist) / close_margin
                r -= 0.3 * float(depth)

            #Time gap reward using gaussian equation
            tg = self._time_gap(i)
            r += 0.5 * float(np.exp(-((tg - self.desired_time_gap) / 1.0) ** 2))

            #If cars are far apart, negative reward for any speed matching since the egos wont learn to get to a closer distance
            desired_gap_dist = self.desired_time_gap * max(float(self.ego_vx[i]), 1e-3)
            if dx > desired_gap_dist * 1.2:
                ego_v   = float(self.ego_vx[i])
                front_v = self._front_v(i)
                speed_error = abs(ego_v - front_v) / self.v_max
                r -= 0.5 * float(speed_error)
                gap_error = (dx - desired_gap_dist) / self.max_gap_dist
                r -= 0.3 * float(gap_error)

            #Abrput accelerations or decelerations give penalty
            r -= 0.02 * (accel ** 2)

            #Jerk is the rate of change of acceleration, so if it changes too fast, negative reward penalty
            if self.steps > 0:
                jerk = (accel - self.prev_accels[i]) / self.dt
                r -= 0.001 * (jerk ** 2)

            #Calculates time to collision (distance / relative velocity) negative reward for if ttc is smaller than the safe amount of 2 seconds
            rel_v = self._rel_v(i)
            if rel_v < -1e-3:
                ttc = dx / abs(rel_v)
                ttc_safe = self.desired_time_gap
                if ttc < ttc_safe:
                    r -= 0.7 * (1.0 - ttc / ttc_safe)

            return float(np.clip(r, -1.0, 1.0))

    #Converts any arrays or list into a plain float value
    @staticmethod
    def _action_to_scalar(act) -> float:
        if act is None:
            return 0.0
        if isinstance(act, (float, int, np.floating, np.integer)):
            return float(act)
        arr = np.asarray(act, dtype=np.float32).reshape(-1)
        return float(arr[0])


    #Reset the environment
    def reset(self, seed: Optional[int] = None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        #List the agents in the env
        self._reset_state()
        self.agents = list(self.possible_agents)

        # Sample a car ID from the correct split depending on current mode
        if self.fixed_car_id is None:
            id_pool = self.train_ids if self.mode == "train" else self.test_ids #If mode is training, sample from training car ids, if testing sample from test car ids
            self.current_car_id = int(self._rng.choice(id_pool)) # Randomly chooses from the id pool, either training or testing and assigns as current car id for this episode
        else:
            self.current_car_id = int(self.fixed_car_id) #When the car id is assigned, just use that one for all episodes

        mask = self.all_car_ids == self.current_car_id
        self.lead_speeds = self.all_speeds[mask]
        self.max_steps   = int(self.lead_speeds.size)
        if self.max_steps <= 1:
            raise RuntimeError(
                f"Car ID {self.current_car_id} has too short a trajectory."
            )

        self.lead_x  = float(self._rng.uniform(30.0, 80.0))
        self.lead_vx = float(self.lead_speeds[0])

        n = self._n_agents
        self.ego_x  = np.zeros(n, dtype=np.float32)
        self.ego_vx = np.zeros(n, dtype=np.float32)

        front_x = self.lead_x
        front_v = self.lead_vx
        lead_v  = self.lead_vx  # save lead speed once so all egos initialise relative to the true lead, not compounding front_v

        for i in range(n):
            base_gap  = self.desired_time_gap * lead_v  #Ideal gap based on desired time headway multiplied by the front vehicles speed
            variation = float(self._rng.uniform(0.6, 1.4))  #Random multiplier between 60% and 140% to give +/- 40% variation
            gap       = float(np.clip(base_gap * variation, self.min_gap_dist + 5.0, self.max_gap_dist))  #Clip to keep gap within valid env bounds
            self.ego_x[i]  = float(front_x - gap - self.vehicle_length) #Calculates bumper to bumper length to find posititon of each ego
            self.ego_vx[i] = float(np.clip(lead_v * (0.90 + 0.2 * float(self._rng.random())), 0.0, self.v_max)) #Randomly choose speed of each ego between 85 and 100% of lead vehicle speed so egos never start faster than lead
            front_x = float(self.ego_x[i]) #Updates front position and velocity to calculate poisition and velocity of ego behind it
            front_v = float(self.ego_vx[i])

        for i in range(n):
            self.prev_front_vx[i] = self._front_v(i)

        # Build observations: get raw physical obs, apply packet loss fallback,
        # then augment with pending action queue. At reset the queue is all zeros
        # so o_act is a zero vector — no actions are pending yet.
        obs = {}
        for i, agent in enumerate(self.possible_agents):
            raw_obs    = self._obs_i(i)
            passed_obs = self.packet_channel.process(agent, raw_obs)
            obs[agent] = self._augment_obs(i, passed_obs)

        infos = {
            agent: {"car_id": self.current_car_id, "agent_index": i} #Info returns the card id, and agents. 
            for i, agent in enumerate(self.possible_agents)
        }
        return obs, infos #This is required in reset function by PettingZoo because AgileRL uses the observations as starting input for neural networks

    def step(self, actions: dict):
        if not self.agents:
            return {}, {}, {}, {}, {}

        n = self._n_agents

        for i in range(n):
            self.prev_front_vx[i] = self._front_v(i)

        # Advance lead first — traj_index increments before reading lead_speeds so
        # the lead_vx used in physics and reward() are consistent within this step
        self.traj_index = min(self.traj_index + 1, self.max_steps - 1)
        self.lead_vx = float(self.lead_speeds[self.traj_index])
        self.lead_x  = self.lead_x + self.lead_vx * self.dt

        # For each agent extract its action from dictionary, convert it to scalar and clip between max decel or accel, then store it in one array
        # If delay is active, push the clipped action into the buffer and execute the action decided k steps ago instead
        current_accels = np.zeros(n, dtype=np.float32)
        for i, agent in enumerate(self.possible_agents):
            a       = self._action_to_scalar(actions.get(agent, 0.0))
            clipped = float(np.clip(a, -self.d_max, self.a_max))
            if self.delay_steps > 0:
                executing = self.action_buffers[i].step(clipped)  # push new action in, get k-step-old action to execute
            else:
                executing = clipped  # no delay, execute immediately as before
            current_accels[i] = executing

        # Update ego states using current_accels
        self.ego_vx = np.clip(
            self.ego_vx + current_accels * self.dt, 0.0, self.v_max
        ).astype(np.float32)
        self.ego_x = (
            self.ego_x + self.ego_vx * self.dt + 0.5 * current_accels * (self.dt ** 2)
        ).astype(np.float32)

        #Creates observations and reward dicts for each agent and lets AgileRL store in replay buffer
        # Observations go through packet loss fallback then augmentation, same pipeline as reset()
        observations = {}
        for i, agent in enumerate(self.possible_agents):
            raw_obs              = self._obs_i(i)
            passed_obs           = self.packet_channel.process(agent, raw_obs)
            observations[agent]  = self._augment_obs(i, passed_obs)

        rewards = {
            agent: self.reward(i, float(current_accels[i]))  # uses executed accel (after delay), not the decided action
            for i, agent in enumerate(self.possible_agents)
        }

        # Update stored accelerations for next step
        self.prev_accels = current_accels.copy()
        self.last_accels = current_accels.copy()   

        # Termination conditions
        dists    = np.array([self._dx(i) for i in range(n)], dtype=np.float32)
        violated = bool(np.any(dists < self.min_gap_dist)) #If any of the distances are less than min gap, terminate 
        time_up  = bool(self.steps >= (self.max_steps - 1)) #If all traj data used then time is up

        terminations = {agent: violated for agent in self.possible_agents}
        truncations  = {agent: time_up  for agent in self.possible_agents}

        infos = {
            agent: {
                "car_id":      self.current_car_id,
                "agent_index": i,
                "dx":          float(self._dx(i)),
                "time_gap":    float(self._time_gap(i)),
                "rel_v":       float(self._rel_v(i)),
                "accel":       float(current_accels[i]),
            }
            for i, agent in enumerate(self.possible_agents)
        }

        self.steps     += 1

        if violated or time_up:
            self.agents = [] #Clears active agent list if episode ends

        return observations, rewards, terminations, truncations, infos #Used by AgileRL in replay buffer to store and uses rewards to update neural network weights

    def close(self):
        pass


Follow1DPlatoonEnv = MultiAgentPlatooningEnv1   # alias for backward compatibility