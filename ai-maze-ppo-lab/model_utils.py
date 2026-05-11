from __future__ import annotations

import numpy as np


def load_trained_model(path: str):
    try:
        from sb3_contrib import RecurrentPPO
    except ImportError:
        RecurrentPPO = None

    if RecurrentPPO is not None:
        try:
            return RecurrentPPO.load(path), True
        except Exception:
            pass

    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise SystemExit(
            "Missing PPO dependencies. Run: pip install -r requirements.txt"
        ) from exc
    return PPO.load(path), False


def initial_recurrent_state(is_recurrent: bool):
    if not is_recurrent:
        return None, None
    return None, np.ones((1,), dtype=bool)


def predict_action(model, obs, deterministic: bool, is_recurrent: bool, state, episode_start):
    if is_recurrent:
        action, next_state = model.predict(
            obs,
            state=state,
            episode_start=episode_start,
            deterministic=deterministic,
        )
        return action, next_state, np.zeros((1,), dtype=bool)

    action, _ = model.predict(obs, deterministic=deterministic)
    return action, state, episode_start
