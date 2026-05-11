import unittest

import numpy as np

from config import ACTIONS
from maze_env import MazePPOEnv
from random_maps import (
    build_random_map_pool,
    find_tile,
    generate_random_exit_map,
    generate_random_key_door_map,
    is_solvable_key_door,
)
from vision import (
    CHANNEL_INDEX,
    LOCAL_GRID_CHANNEL_INDEX,
    encode_line_of_sight,
    encode_local_grid,
    local_grid_channels,
    observation_size,
)


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

    def test_exploration_reward_is_opt_in(self):
        env = MazePPOEnv(
            map_lines=["#####", "#S..#", "#..E#", "#####"],
            exploration_reward=True,
        )
        env.reset()
        _, reward, *_ = env.step(3)
        self.assertAlmostEqual(reward, -0.05)
        env.step(2)
        _, reward, *_ = env.step(3)
        self.assertAlmostEqual(reward, -0.18)
        env.step(2)
        _, reward, *_ = env.step(3)
        self.assertAlmostEqual(reward, -0.18)
        _, reward, *_ = env.step(2)
        self.assertAlmostEqual(reward, -0.18)

    def test_keyless_map_starts_with_key_condition_satisfied(self):
        env = MazePPOEnv(map_lines=["#####", "#SE.#", "#####"])
        _, info = env.reset()
        self.assertFalse(info["requires_key"])
        self.assertFalse(info["has_door"])
        self.assertTrue(info["has_key"])
        self.assertTrue(info["passed_door"])

    def test_large_map_gets_scaled_step_budget(self):
        rows = 21
        cols = 31
        lines = ["#" * cols]
        for row in range(1, rows - 1):
            if row == 1:
                lines.append("#S" + "." * (cols - 3) + "#")
            elif row == rows - 2:
                lines.append("#" + "." * (cols - 3) + "E#")
            else:
                lines.append("#" + "." * (cols - 2) + "#")
        lines.append("#" * cols)

        env = MazePPOEnv(map_lines=lines)
        env.reset()
        self.assertGreater(env.max_steps, 260)


class VisionTest(unittest.TestCase):
    def test_observation_shape_is_fixed(self):
        grid = [list("#######"), list("#S.K.E#"), list("#######")]
        obs = encode_line_of_sight(
            grid,
            (1, 1),
            False,
            False,
            0,
            100,
            view_range=3,
            view_width=3,
        )
        self.assertEqual(obs.shape, (observation_size(3, 3),))

    def test_wall_hides_tiles_behind_it(self):
        grid = [list("########"), list("#S#K..E#"), list("########")]
        obs = encode_line_of_sight(
            grid,
            (1, 1),
            False,
            False,
            0,
            100,
            view_range=3,
            view_width=3,
        )
        strips = obs[:-2].reshape((len(ACTIONS), 3, 3, -1))
        right_strip = strips[3]
        center_lane = 1
        self.assertEqual(np.argmax(right_strip[0, center_lane]), CHANNEL_INDEX["wall"])
        self.assertEqual(np.argmax(right_strip[1, center_lane]), CHANNEL_INDEX["unknown"])

    def test_directional_strip_has_side_vision(self):
        grid = [
            list("########"),
            list("#......#"),
            list("#.S.K.E#"),
            list("#......#"),
            list("########"),
        ]
        obs = encode_line_of_sight(
            grid,
            (2, 2),
            False,
            False,
            0,
            100,
            view_range=3,
            view_width=3,
        )
        strips = obs[:-2].reshape((len(ACTIONS), 3, 3, -1))
        right_strip = strips[3]
        self.assertEqual(np.argmax(right_strip[1, 1]), CHANNEL_INDEX["key"])
        self.assertEqual(np.argmax(right_strip[0, 0]), CHANNEL_INDEX["empty"])

    def test_local_grid_observation_is_image_shaped(self):
        grid = [list("#####"), list("#SKE#"), list("#####")]
        obs = encode_local_grid(
            grid,
            (1, 1),
            False,
            False,
            0,
            100,
            local_view_size=7,
        )
        self.assertEqual(obs.shape, (local_grid_channels(), 7, 7))
        self.assertEqual(obs.dtype, np.uint8)
        self.assertEqual(obs[LOCAL_GRID_CHANNEL_INDEX["tile"], 3, 3], 255)

    def test_env_grid_observation_tracks_last_action_and_repeats(self):
        env = MazePPOEnv(map_lines=["#####", "#S..#", "#..E#", "#####"])
        obs, _ = env.reset()
        self.assertEqual(obs.shape, env.observation_space.shape)
        obs, *_ = env.step(3)
        self.assertEqual(obs[LOCAL_GRID_CHANNEL_INDEX["last_action"], 3, 3], 255)
        obs, *_ = env.step(2)
        self.assertEqual(obs[LOCAL_GRID_CHANNEL_INDEX["last_action"], 3, 3], 191)
        self.assertEqual(obs[LOCAL_GRID_CHANNEL_INDEX["step_repeat"], 3, 3], 255)


class RandomMapTest(unittest.TestCase):
    def test_generated_maps_are_solvable(self):
        generated = generate_random_key_door_map(rng=123)
        text = "\n".join(generated.lines)
        self.assertIn("S", text)
        self.assertIn("K", text)
        self.assertIn("D", text)
        self.assertIn("E", text)
        self.assertTrue(is_solvable_key_door(generated.lines))

    def test_random_map_pool_uses_custom_size(self):
        pool = build_random_map_pool(
            2,
            seed=123,
            rows=13,
            cols=17,
            wall_density=0.18,
            trap_density=0.08,
            style="rooms",
            door_orientation="vertical",
            endpoint_mode="edges",
        )
        self.assertEqual(len(pool), 2)
        self.assertTrue(all(len(item.lines) == 13 for item in pool))
        self.assertTrue(all(len(item.lines[0]) == 17 for item in pool))
        self.assertTrue(all(is_solvable_key_door(item.lines) for item in pool))

    def test_random_maps_support_barrier_orientations(self):
        horizontal = generate_random_key_door_map(
            rng=7,
            door_orientation="horizontal",
            endpoint_mode="corners",
            wall_density=0.05,
        )
        vertical = generate_random_key_door_map(
            rng=8,
            door_orientation="vertical",
            endpoint_mode="corners",
            wall_density=0.05,
        )

        h_door = find_tile(horizontal.lines, "D")
        v_door = find_tile(vertical.lines, "D")
        self.assertIsNotNone(h_door)
        self.assertIsNotNone(v_door)
        assert h_door is not None
        assert v_door is not None

        h_row_walls = horizontal.lines[h_door[0]].count("#")
        h_col_walls = sum(line[h_door[1]] == "#" for line in horizontal.lines)
        v_row_walls = vertical.lines[v_door[0]].count("#")
        v_col_walls = sum(line[v_door[1]] == "#" for line in vertical.lines)

        self.assertGreater(h_row_walls, h_col_walls)
        self.assertGreater(v_col_walls, v_row_walls)
        self.assertTrue(is_solvable_key_door(horizontal.lines))
        self.assertTrue(is_solvable_key_door(vertical.lines))

    def test_random_map_styles_are_solvable(self):
        for index, style in enumerate(["open", "split", "rooms", "deadends", "maze"]):
            with self.subTest(style=style):
                generated = generate_random_key_door_map(
                    rng=100 + index,
                    rows=15,
                    cols=21,
                    style=style,
                    door_orientation="mixed",
                    endpoint_mode="mixed",
                    wall_density=0.14,
                    trap_density=0.05,
                )
                self.assertTrue(is_solvable_key_door(generated.lines))

    def test_random_exit_maps_have_no_key_or_door(self):
        generated = generate_random_exit_map(
            rng=321,
            rows=13,
            cols=17,
            style="rooms",
            wall_density=0.12,
            trap_density=0.04,
        )
        text = "\n".join(generated.lines)
        self.assertIn("S", text)
        self.assertIn("E", text)
        self.assertNotIn("K", text)
        self.assertNotIn("D", text)
        self.assertTrue(is_solvable_key_door(generated.lines))

    def test_random_pool_can_mix_simple_and_key_door_maps(self):
        pool = build_random_map_pool(8, seed=456, simple_map_probability=0.5)
        texts = ["\n".join(item.lines) for item in pool]
        self.assertTrue(any("K" not in text and "D" not in text for text in texts))
        self.assertTrue(any("K" in text and "D" in text for text in texts))


if __name__ == "__main__":
    unittest.main()
