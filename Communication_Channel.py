from collections import deque
import numpy as np


class ActionBuffer:
    # Implements the DAMARL delayed action queue for one agent.
    # At every step the agent decides a new action which goes to the BACK of the
    # queue. The action at the FRONT of the queue (decided k steps ago) is what
    # actually gets executed in physics. This makes the delay learnable because
    # the full queue is appended to the observation so the policy can see every
    # committed-but-not-yet-executed action.

    def __init__(self, delay_steps: int):
        self.k = max(int(delay_steps), 1)  # minimum 1 so deque is never empty
        self.queue = deque([0.0] * self.k, maxlen=self.k)

    def reset(self):
        # Clears queue to all zeros at the start of each episode so the agent
        # doesn't carry over stale actions from the previous trajectory
        for j in range(self.k):
            self.queue[j] = 0.0

    def step(self, new_action: float) -> float:
        # Pushes the newly decided action onto the back of the queue.
        # The deque maxlen automatically pops the front (oldest) element,
        # which is the action that was decided k steps ago and executes now.
        # We read the front BEFORE appending so we return the executing action.
        executing = float(self.queue[0])
        self.queue.append(new_action)
        return executing

    def get_pending_sequence(self) -> np.ndarray:
        # Returns the full queue as a 1D array to be appended to the observation.
        # These are the o_act values in the augmented state o_i = (o_obs, o_act).
        return np.array(list(self.queue), dtype=np.float32)


class PacketLossChannel:
    # Simulates faulty V2V communication by randomly dropping observation packets.
    # When a packet is dropped, the last successfully received observation is
    # returned instead. Each agent has its own last-valid cache so a drop for
    # one agent doesn't affect others.
    # Note: packet loss applies to the physical observation only (what the agent
    # sees about the world). The action queue is local to each agent and cannot
    # be lost.

    def __init__(self, loss_prob: float, rng: np.random.Generator):
        self.loss_prob = float(loss_prob)
        self._rng = rng
        self._last_valid_obs: dict = {}  # agent_id -> last valid observation array

    def reset(self):
        # Clears stale cached observations at the start of each episode so agents
        # don't fall back on observations from the previous trajectory
        self._last_valid_obs = {}

    def process(self, agent_id: str, obs: np.ndarray) -> np.ndarray:
        # Either passes the observation through unchanged, or if a packet is
        # lost, returns the last valid observation for that agent.
        # On the very first step there is no cached obs, so the current one is
        # always used regardless of loss probability.
        if self.loss_prob > 0.0 and self._rng.random() < self.loss_prob:
            if agent_id in self._last_valid_obs:
                return self._last_valid_obs[agent_id]  # return stale obs
        # packet received successfully — update cache and return current obs
        self._last_valid_obs[agent_id] = obs.copy()
        return obs