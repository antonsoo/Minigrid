from __future__ import annotations

import multiprocessing

import gymnasium as gym


def _reset_synth():
    env = gym.make("minigrid:BabyAI-SynthS5R2-v0")
    try:
        obs, _ = env.reset(seed=1741)
        assert env.observation_space.contains(obs)
        u = env.unwrapped
        front_cell = u.grid.get(*u.front_pos)
        assert front_cell is None or front_cell.type == "wall"
        assert u.grid.get(*u.agent_pos) is None
    finally:
        env.close()


def test_reset_retries_room_without_valid_agent_direction():
    # This seed fills the selected room except for its center, surrounded by
    # objects. Sampling empty positions succeeds, but every facing is rejected.
    process = multiprocessing.get_context("spawn").Process(target=_reset_synth)
    process.start()
    try:
        process.join(timeout=10)
        assert process.exitcode == 0, "reset did not finish successfully"
    finally:
        if process.is_alive():
            process.terminate()
        process.join()
