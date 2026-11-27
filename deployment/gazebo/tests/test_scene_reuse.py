"""Repeated launch must retain the active world and avoid container-name conflicts."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reuse_scene


class SceneReuseTest(unittest.TestCase):
    def scene(self, root=None):
        return {"Mounts": [{"Destination": "/workspace", "Source": str(root or reuse_scene.ROOT)},
                           {"Destination": "/tmp/.X11-unix", "Source": "/tmp/.X11-unix"}],
                "Config": {"Env": ["DISPLAY=:1", "SIGMARL_ARENA_CONFIG=/workspace/config/jetauto_local_junction.json"]},
                "State": {"Running": True, "Paused": False}}

    def inspection(self, scene):
        return subprocess.CompletedProcess([], 0, json.dumps([scene]), "")

    @patch("reuse_scene.subprocess.run")
    def test_missing_container_falls_through_to_new_launch(self, run):
        run.return_value = subprocess.CompletedProcess([], 1, "", "Error: No such object: container")
        self.assertEqual(reuse_scene.reuse_scene("gui", "jetauto_local_junction.json"), reuse_scene.NEW_CONTAINER)

    @patch("reuse_scene.subprocess.run")
    def test_open_gui_is_reused_without_start_or_reset(self, run):
        run.side_effect = [self.inspection(self.scene()), subprocess.CompletedProcess([], 0, "123\n", "")]
        self.assertEqual(reuse_scene.reuse_scene("gui", "jetauto_local_junction.json"), 0)
        self.assertEqual(run.call_count, 2)
        self.assertEqual(run.call_args.args[0][-3:], ["pgrep", "-x", "gzclient"])

    @patch("reuse_scene.os.execvp")
    @patch("reuse_scene.subprocess.run")
    def test_closed_window_reconnects_to_same_server(self, run, execute):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "runtime/ros2").mkdir(parents=True)
            with patch.object(reuse_scene, "ROOT", root):
                run.side_effect = [self.inspection(self.scene(root)), subprocess.CompletedProcess([], 1, "", ""),
                                   subprocess.CompletedProcess([], 0, "0100cookie\n", ""), subprocess.CompletedProcess([], 0)]
                self.assertEqual(reuse_scene.reuse_scene("gui", "jetauto_local_junction.json"), 0)
            arguments = execute.call_args.args[1]
            self.assertEqual(arguments[:2], ["docker", "exec"])
            self.assertIn(reuse_scene.CONTAINER, arguments)
            self.assertEqual(arguments[-2:], ["gzclient", "--verbose"])

    @patch("reuse_scene.subprocess.run")
    def test_run_reuses_world_and_does_not_build(self, run):
        run.return_value = self.inspection(self.scene())
        self.assertEqual(reuse_scene.reuse_scene("run", "jetauto_local_junction.json"), 0)
        self.assertEqual(run.call_count, 1)
        with self.assertRaisesRegex(SystemExit, "构建前"):
            reuse_scene.reuse_scene("build", "jetauto_local_junction.json")

    @patch("reuse_scene.subprocess.run")
    def test_unrelated_workspace_or_different_profile_is_not_reused(self, run):
        run.return_value = self.inspection(self.scene(Path("/another/workspace")))
        with self.assertRaisesRegex(SystemExit, "其他工作目录"):
            reuse_scene.reuse_scene("gui", "jetauto_local_junction.json")
        run.return_value = self.inspection(self.scene())
        with self.assertRaisesRegex(SystemExit, "不同的场景"):
            reuse_scene.reuse_scene("gui", "jetauto_a1.json")

    @patch("reuse_scene.subprocess.run")
    def test_stopped_or_headless_container_gives_actionable_message(self, run):
        scene = self.scene()
        scene["State"]["Running"] = False
        run.return_value = self.inspection(scene)
        with self.assertRaisesRegex(SystemExit, "docker start -a"):
            reuse_scene.reuse_scene("gui", "jetauto_local_junction.json")
        scene = self.scene()
        scene["Mounts"] = scene["Mounts"][:1]
        run.side_effect = [self.inspection(scene), subprocess.CompletedProcess([], 1, "", "")]
        with self.assertRaisesRegex(SystemExit, "无窗口模式"):
            reuse_scene.reuse_scene("gui", "jetauto_local_junction.json")
