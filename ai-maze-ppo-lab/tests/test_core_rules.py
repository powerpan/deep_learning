import unittest

import numpy as np

from config import ACTIONS
from maze_env import MazePPOEnv
from random_maps import generate_random_key_door_map, is_solvable_key_door
from vision import CHANNEL_INDEX, encode_line_of_sight, observation_size


class MazeRulesTest(unittest.TestCase):
    def test_locked_door_blocks_without_key(self):
        env = MazePPOEnv(map_lines=["#####", "#SDK#", "#..E#", "#####"])
        env.reset()
        _, reward, terminated, truncated, info = env.step(3)
        self.assertEqual(info["position"], (1, 1))
        self.assertEqual(info["event"], "locked_door")
        self.assertEqual(reward, -2.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)

    def test_key_reward_only_once(self):
        env = MazePPOEnv(map_lines=["#####", "#SKE#", "#####"])
        env.reset()
        _, reward, *_ = env.step(3)
        self.assertAlmostEqual(reward, 9.9)
        env.step(2)
        _, reward, *_ = env.step(3)
        self.assertAlmostEqual(reward, -0.1)

    def test_exit_terminates_successfully(self):
        env = MazePPOEnv(map_lines=["#####", "#SE.#", "#####"])
        env.reset()
        _, reward, terminated, truncated, info = env.step(3)
        self.assertEqual(reward, 80.0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["success"])


class VisionTest(unittest.TestCase):
    def test_observation_shape_is_fixed(self):
        grid = [list("#######"), list("#S.K.E#"), list("#######")]
        obs = encode_line_of_sight(grid, (1, 1), False, False, 0, 100, view_range=7)
        self.assertEqual(obs.shape, (observation_size(7),))

    def test_wall_hides_tiles_behind_it(self):
        grid = [list("########"), list("#S#K..E#"), list("########")]
        obs = encode_line_of_sight(grid, (1, 1), False, False, 0, 100, view_range=3)
        rays = obs[:-2].reshape((len(ACTIONS), 3, -1))
        right_ray = rays[3]
        self.assertEqual(np.argmax(right_ray[0]), CHANNEL_INDEX["wall"])
        self.assertEqual(np.argmax(right_ray[1]), CHANNEL_INDEX["unknown"])


class RandomMapTest(unittest.TestCase):
    def test_generated_maps_are_solvable(self):
        generated = generate_random_key_door_map(rng=123)
        text = "\n".join(generated.lines)
        self.assertIn("S", text)
        self.assertIn("K", text)
        self.assertIn("D", text)
        self.assertIn("E", text)
        self.assertTrue(is_solvable_key_door(generated.lines))


if __name__ == "__main__":
    unittest.main()
