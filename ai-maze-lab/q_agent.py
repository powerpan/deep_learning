import numpy as np

from config import (
    ACTION_COUNT,
    ALPHA,
    EPSILON_DECAY,
    EPSILON_MIN,
    EPSILON_START,
    GAMMA,
)


class QLearningAgent:
    """Tabular Q-learning agent for a row/col/key grid state space."""

    def __init__(
        self,
        rows,
        cols,
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon=EPSILON_START,
        epsilon_decay=EPSILON_DECAY,
        epsilon_min=EPSILON_MIN,
        seed=None,
    ):
        self.q_table = np.zeros((rows, cols, 2, ACTION_COUNT), dtype=np.float32)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.rng = np.random.default_rng(seed)

    def choose_action(self, state, explore=True):
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(ACTION_COUNT))
        return self.greedy_action(state)

    def greedy_action(self, state):
        row, col, has_key = self._unpack_state(state)
        q_values = self.q_table[row, col, has_key]
        best_actions = np.flatnonzero(q_values == np.max(q_values))
        return int(self.rng.choice(best_actions))

    def update(self, state, action, reward, next_state, done):
        row, col, has_key = self._unpack_state(state)
        next_row, next_col, next_has_key = self._unpack_state(next_state)

        old_value = self.q_table[row, col, has_key, action]
        future_value = 0.0 if done else float(np.max(self.q_table[next_row, next_col, next_has_key]))
        target = reward + self.gamma * future_value
        self.q_table[row, col, has_key, action] = old_value + self.alpha * (target - old_value)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return self.epsilon

    def _unpack_state(self, state):
        if len(state) == 2:
            row, col = state
            return row, col, 0
        row, col, has_key = state
        return row, col, int(has_key)
