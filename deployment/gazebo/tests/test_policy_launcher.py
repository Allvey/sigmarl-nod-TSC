"""Simulation lifecycle failures must produce actionable launcher errors."""
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_policy import require_running_gazebo


class PolicyLauncherTest(unittest.TestCase):
    @patch("run_policy.subprocess.run")
    def test_deleted_container_explains_both_terminal_commands(self, run):
        for error in ("Error: No such object: sigmarl-jetauto-humble", "Error: No such container: sigmarl-jetauto-humble"):
            with self.subTest(error=error):
                run.return_value = subprocess.CompletedProcess([], 1, "", error)
                with self.assertRaises(SystemExit) as stopped:
                    require_running_gazebo()
                message = str(stopped.exception)
                self.assertIn("launch_jetauto.sh gui", message)
                self.assertIn("policy --duration 10", message)
                self.assertIn("保持该终端运行", message)

    @patch("run_policy.subprocess.run")
    def test_stopped_container_uses_existing_container(self, run):
        run.return_value = subprocess.CompletedProcess([], 0, "false\n", "")
        with self.assertRaises(SystemExit) as stopped:
            require_running_gazebo()
        self.assertIn("docker start -a sigmarl-jetauto-humble", str(stopped.exception))

    @patch("run_policy.subprocess.run")
    def test_docker_access_errors_are_not_reported_as_missing_container(self, run):
        run.return_value = subprocess.CompletedProcess([], 1, "", "permission denied connecting to Docker")
        with self.assertRaises(SystemExit) as stopped:
            require_running_gazebo()
        self.assertIn("permission denied", str(stopped.exception))
        self.assertNotIn("launch_jetauto.sh gui", str(stopped.exception))
        run.side_effect = subprocess.TimeoutExpired("docker", 10)
        with self.assertRaisesRegex(SystemExit, "超时"):
            require_running_gazebo()
        run.side_effect = FileNotFoundError()
        with self.assertRaisesRegex(SystemExit, "未找到 docker"):
            require_running_gazebo()

    @patch("run_policy.subprocess.run")
    def test_running_container_allows_policy_start(self, run):
        run.return_value = subprocess.CompletedProcess([], 0, "true\n", "")
        self.assertIsNone(require_running_gazebo())
