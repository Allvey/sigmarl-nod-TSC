# Copyright (c) 2024, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

import math

# Tensordict modules
from tensordict.tensordict import TensorDictBase, TensorDict
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor

from torchrl.collectors import SyncDataCollector

# Env
from torchrl.envs import TransformedEnv, StepCounter
from torchrl.envs.utils import (
    set_exploration_type,
    _terminated_or_truncated,
    step_mdp,
    _aggregate_end_of_traj,
    ExplorationType,
)
try:
    from torchrl.envs.utils import _convert_exploration_type
except ImportError:
    # TorchRL >= 0.10 removed this private compatibility helper.
    def _convert_exploration_type(exploration_mode=None, exploration_type=None):
        if exploration_mode is not None:
            return ExplorationType(exploration_mode)
        return exploration_type


from torchrl.envs.common import EnvBase
from torchrl.envs.libs.vmas import VmasEnv


# TorchRL Utils
from torchrl._utils import (
    _check_for_faulty_process,
    _ProcessNoWarn,
    accept_remote_rref_udf_invocation,
    prod,
    RL_WARNINGS,
    VERBOSE,
)

# Losses
from torchrl.objectives import ClipPPOLoss, ValueEstimators

from torchrl.data.utils import DEVICE_TYPING

from vmas.simulator.utils import (
    X,
    Y,
)

# Multi-agent network
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal

# Specs
from torchrl.data import UnboundedContinuousTensorSpec

# Utils
from matplotlib import pyplot as plt
from typing import Callable, Optional, Callable, Optional
from ctypes import byref

from matplotlib import pyplot as plt
import json
import os
import re

from typing import Any, Callable, Dict, Iterator, Optional, Sequence, Tuple, Union

DEFAULT_EXPLORATION_TYPE: ExplorationType = ExplorationType.RANDOM

import warnings

from utilities.constants import AGENTS
from utilities.topology_labels import (
    generate_soft_labels_full_graph,
    break_cycles_min_cost,
    enforce_transitivity,
    complete_total_order,
)


def get_model_name(parameters):
    # model_name = f"nags{parameters.n_agents}_it{parameters.n_iters}_fpb{parameters.frames_per_batch}_tfrms{parameters.total_frames}_neps{parameters.num_epochs}_mnbsz{parameters.minibatch_size}_lr{parameters.lr}_mgn{parameters.max_grad_norm}_clp{parameters.clip_epsilon}_gm{parameters.gamma}_lmbda{parameters.lmbda}_etp{parameters.entropy_eps}_mstp{parameters.max_steps}_nenvs{parameters.num_vmas_envs}"
    model_name = f"reward{parameters.episode_reward_mean_current:.2f}"

    return model_name


##################################################
## Custom Classes
##################################################
class TransformedEnvCustom(TransformedEnv):
    """
    Slightly modify the function `rollout`, `_rollout_stop_early`, and `_rollout_nonstop` to enable returning a frame list to save evaluation video
    """

    def rollout(
        self,
        max_steps: int,
        policy: Optional[Callable[[TensorDictBase], TensorDictBase]] = None,
        callback: Optional[Callable[[TensorDictBase], TensorDictBase]] = None,
        auto_reset: bool = True,
        auto_cast_to_device: bool = False,
        break_when_any_done: bool = True,
        return_contiguous: bool = True,
        tensordict: Optional[TensorDictBase] = None,
        out=None,
        is_save_simulation_video: bool = False,
        priority_module=None,
    ):
        """Executes a rollout in the environment.

        The function will stop as soon as one of the contained environments
        returns done=True.

        Args:
            max_steps (int): maximum number of steps to be executed. The actual number of steps can be smaller if
                the environment reaches a done state before max_steps have been executed.
            policy (callable, optional): callable to be called to compute the desired action. If no policy is provided,
                actions will be called using :obj:`env.rand_step()`
                default = None
            callback (callable, optional): function to be called at each iteration with the given TensorDict.
            auto_reset (bool, optional): if ``True``, resets automatically the environment
                if it is in a done state when the rollout is initiated.
                Default is ``True``.
            auto_cast_to_device (bool, optional): if ``True``, the device of the tensordict is automatically cast to the
                policy device before the policy is used. Default is ``False``.
            break_when_any_done (bool): breaks if any of the done state is True. If False, a reset() is
                called on the sub-envs that are done. Default is True.
            return_contiguous (bool): if False, a LazyStackedTensorDict will be returned. Default is True.
            tensordict (TensorDict, optional): if auto_reset is False, an initial
                tensordict must be provided.

        Returns:
            TensorDict object containing the resulting trajectory.

        The data returned will be marked with a "time" dimension name for the last
        dimension of the tensordict (at the ``env.ndim`` index).
        """
        try:
            policy_device = next(policy.parameters()).device
        except (StopIteration, AttributeError):
            policy_device = self.device

        env_device = self.device

        if auto_reset:
            if tensordict is not None:
                raise RuntimeError(
                    "tensordict cannot be provided when auto_reset is True"
                )
            tensordict = self.reset()
        elif tensordict is None:
            raise RuntimeError("tensordict must be provided when auto_reset is False")
        if policy is None:
            policy = self.rand_action

        kwargs = {
            "tensordict": tensordict,
            "auto_cast_to_device": auto_cast_to_device,
            "max_steps": max_steps,
            "policy": policy,
            "policy_device": policy_device,
            "env_device": env_device,
            "callback": callback,
            "is_save_simulation_video": is_save_simulation_video,
            "priority_module": priority_module,
        }
        if break_when_any_done:
            if is_save_simulation_video:
                tensordicts, frame_list = self._rollout_stop_early(
                    **kwargs
                )  # Modification
            else:
                tensordicts = self._rollout_stop_early(**kwargs)
        else:
            if is_save_simulation_video:
                tensordicts, frame_list = self._rollout_nonstop(
                    **kwargs
                )  # Modification
            else:
                tensordicts = self._rollout_nonstop(**kwargs)

        batch_size = self.batch_size if tensordict is None else tensordict.batch_size
        out_td = torch.stack(tensordicts, len(batch_size), out=out)
        if return_contiguous:
            out_td = out_td.contiguous()
        out_td.refine_names(..., "time")

        if is_save_simulation_video:
            return out_td, frame_list  # Modification
        else:
            return out_td

    def _rollout_stop_early(
        self,
        *,
        tensordict,
        auto_cast_to_device,
        max_steps,
        policy,
        policy_device,
        env_device,
        callback,
        is_save_simulation_video,
        priority_module,
    ):
        tensordicts = []

        if is_save_simulation_video:
            frame_list = []

        for i in range(max_steps):
            if auto_cast_to_device:
                tensordict = tensordict.to(policy_device, non_blocking=True)

            # Possibly predict the actions of surrounding agents using opponent modeling
            if (
                self.base_env.scenario.parameters.is_using_opponent_modeling
                and not self.base_env.scenario.parameters.is_testing_mode
            ):
                opponent_modeling(
                    tensordict=tensordict,
                    policy=policy,
                    n_nearing_agents_observed=self.base_env.scenario.parameters.n_nearing_agents_observed,
                    nearing_agents_indices=self.base_env.scenario.observations.nearing_agents_indices,
                    action_predictor=getattr(
                        self.base_env.scenario, "topology_action_predictor", None
                    ),
                    parameters=self.base_env.scenario.parameters,
                )

            if (
                self.base_env.scenario.parameters.is_using_prioritized_marl
                and priority_module
            ):
                tensordict = prioritized_ap_policy(
                    tensordict,
                    policy,
                    priority_module,
                    self.base_env.scenario.observations.nearing_agents_indices,
                    self.base_env.scenario.parameters.prioritization_method,
                )
            else:
                tensordict = policy(tensordict)

            if auto_cast_to_device:
                tensordict = tensordict.to(env_device, non_blocking=True)
            tensordict = self.step(tensordict)
            tensordicts.append(tensordict.clone(False))

            if i == max_steps - 1:
                # we don't truncated as one could potentially continue the run
                break
            tensordict = step_mdp(
                tensordict,
                keep_other=True,
                exclude_action=False,
                exclude_reward=True,
                reward_keys=self.reward_keys,
                action_keys=self.action_keys,
                done_keys=self.done_keys,
            )
            # done and truncated are in done_keys
            # We read if any key is done.
            any_done = _terminated_or_truncated(
                tensordict,
                full_done_spec=self.output_spec["full_done_spec"],
                key=None,
            )
            if any_done:
                break

            if callback is not None:
                if is_save_simulation_video:
                    frame = callback(self, tensordict)  # Modification
                    frame_list.append(frame)
                else:
                    callback(self, tensordict)

        if is_save_simulation_video:
            return tensordicts, frame_list  # Modification
        else:
            return tensordicts

    def _rollout_nonstop(
        self,
        *,
        tensordict,
        auto_cast_to_device,
        max_steps,
        policy,
        policy_device,
        env_device,
        callback,
        is_save_simulation_video,
        priority_module,
    ):
        tensordicts = []
        tensordict_ = tensordict

        if is_save_simulation_video:
            frame_list = []

        for i in range(max_steps):
            if auto_cast_to_device:
                tensordict_ = tensordict_.to(policy_device, non_blocking=True)

            # Possibly predict the actions of surrounding agents using opponent modeling
            if (
                self.base_env.scenario.parameters.is_using_opponent_modeling
                and not self.base_env.scenario.parameters.is_testing_mode
            ):
                opponent_modeling(
                    tensordict=tensordict_,
                    policy=policy,
                    n_nearing_agents_observed=self.base_env.scenario.parameters.n_nearing_agents_observed,
                    nearing_agents_indices=self.base_env.scenario.observations.nearing_agents_indices,
                    action_predictor=getattr(
                        self.base_env.scenario, "topology_action_predictor", None
                    ),
                    parameters=self.base_env.scenario.parameters,
                )

            if (
                self.base_env.scenario.parameters.is_using_prioritized_marl
                and priority_module
            ):
                tensordict_ = prioritized_ap_policy(
                    tensordict_,
                    policy,
                    priority_module,
                    self.base_env.scenario.observations.nearing_agents_indices,
                    self.base_env.scenario.parameters.prioritization_method,
                )
            else:
                tensordict_ = policy(tensordict_)
            if auto_cast_to_device:
                tensordict_ = tensordict_.to(env_device, non_blocking=True)
            prev_td = tensordict_
            tensordict, tensordict_ = self.step_and_maybe_reset(tensordict_)
            if (
                self.base_env.scenario.parameters.is_using_opponent_modeling
                and not self.base_env.scenario.parameters.is_testing_mode
            ):
                critic_obs = prev_td.get(
                    ("agents", "info", "critic_observation"), default=None
                )
                if critic_obs is not None:
                    tensordict[("agents", "info", "critic_observation")] = critic_obs
                opponent_modeling(
                    tensordict=tensordict_,
                    policy=policy,
                    n_nearing_agents_observed=self.base_env.scenario.parameters.n_nearing_agents_observed,
                    nearing_agents_indices=self.base_env.scenario.observations.nearing_agents_indices,
                    action_predictor=getattr(
                        self.base_env.scenario, "topology_action_predictor", None
                    ),
                    parameters=self.base_env.scenario.parameters,
                )
                critic_obs_next = tensordict_.get(
                    ("agents", "info", "critic_observation"), default=None
                )
                if critic_obs_next is not None:
                    tensordict[
                        ("next", "agents", "info", "critic_observation")
                    ] = critic_obs_next
            tensordicts.append(tensordict)
            if i == max_steps - 1:
                # we don't truncated as one could potentially continue the run
                break

            if callback is not None:
                if is_save_simulation_video:
                    frame = callback(self, tensordict)  # Modification
                    frame_list.append(frame)
                else:
                    callback(self, tensordict)

        if is_save_simulation_video:
            return tensordicts, frame_list  # Modification
        else:
            return tensordicts


class SyncDataCollectorCustom(SyncDataCollector):
    """
    Slightly modify the function `rollout` to enable opponent modeling and prioritized action propagation.
    """

    def helper_init(
        self,
        create_env_fn: Union[
            EnvBase, Sequence[Callable[[], EnvBase]]  # noqa: F821
        ],  # noqa: F821
        policy: Optional[
            Union[
                TensorDictModule,
                Callable[[TensorDictBase], TensorDictBase],
            ]
        ],
        *,
        frames_per_batch: int,
        total_frames: int,
        device: DEVICE_TYPING = None,
        storing_device: DEVICE_TYPING = None,
        create_env_kwargs: Union[Dict, None] = None,  # Changed from dict | None
        max_frames_per_traj: Union[int, None] = None,  # Changed from int | None
        init_random_frames: Union[int, None] = None,  # Changed from int | None
        reset_at_each_iter: bool = False,
        postproc: Optional[
            Callable[[TensorDictBase], TensorDictBase]
        ] = None,  # Changed from Callable[...] | None
        split_trajs: Union[bool, None] = None,  # Changed from bool | None
        exploration_type: ExplorationType = DEFAULT_EXPLORATION_TYPE,
        exploration_mode=None,
        return_same_td: bool = False,
        reset_when_done: bool = True,
        interruptor=None,
    ):
        from torchrl.envs.batched_envs import _BatchedEnv

        self.closed = True

        exploration_type = _convert_exploration_type(
            exploration_mode=exploration_mode, exploration_type=exploration_type
        )
        if create_env_kwargs is None:
            create_env_kwargs = {}
        if not isinstance(create_env_fn, EnvBase):
            env = create_env_fn(**create_env_kwargs)
        else:
            env = create_env_fn
            if create_env_kwargs:
                if not isinstance(env, _BatchedEnv):
                    raise RuntimeError(
                        "kwargs were passed to SyncDataCollector but they can't be set "
                        f"on environment of type {type(create_env_fn)}."
                    )
                env.update_kwargs(create_env_kwargs)

        if storing_device is None:
            if device is not None:
                storing_device = device
            elif policy is not None:
                try:
                    policy_device = next(policy.parameters()).device
                except (AttributeError, StopIteration):
                    policy_device = torch.device("cpu")
                storing_device = policy_device
            else:
                storing_device = torch.device("cpu")

        self.storing_device = torch.device(storing_device)
        self.env: EnvBase = env
        self.closed = False
        if not reset_when_done:
            raise ValueError("reset_when_done is deprectated.")
        self.reset_when_done = reset_when_done
        self.n_env = self.env.batch_size.numel()

        (self.policy, self.device, self.get_weights_fn,) = self._get_policy_and_device(
            policy=policy,
            device=device,
            observation_spec=self.env.observation_spec,
        )

        if isinstance(self.policy, nn.Module):
            self.policy_weights = TensorDict(dict(self.policy.named_parameters()), [])
            self.policy_weights.update(
                TensorDict(dict(self.policy.named_buffers()), [])
            )
        else:
            self.policy_weights = TensorDict({}, [])

        self.env: EnvBase = self.env.to(self.device)
        self.max_frames_per_traj = (
            int(max_frames_per_traj) if max_frames_per_traj is not None else 0
        )
        if self.max_frames_per_traj is not None and self.max_frames_per_traj > 0:
            # let's check that there is no StepCounter yet
            for key in self.env.output_spec.keys(True, True):
                if isinstance(key, str):
                    key = (key,)
                if "step_count" in key:
                    raise ValueError(
                        "A 'step_count' key is already present in the environment "
                        "and the 'max_frames_per_traj' argument may conflict with "
                        "a 'StepCounter' that has already been set. "
                        "Possible solutions: Set max_frames_per_traj to 0 or "
                        "remove the StepCounter limit from the environment transforms."
                    )
            env = self.env = TransformedEnv(
                self.env, StepCounter(max_steps=self.max_frames_per_traj)
            )

        if total_frames is None or total_frames < 0:
            total_frames = float("inf")
        else:
            remainder = total_frames % frames_per_batch
            if remainder != 0 and RL_WARNINGS:
                warnings.warn(
                    f"total_frames ({total_frames}) is not exactly divisible by frames_per_batch ({frames_per_batch})."
                    f"This means {frames_per_batch - remainder} additional frames will be collected."
                    "To silence this message, set the environment variable RL_WARNINGS to False."
                )
        self.total_frames = (
            int(total_frames) if total_frames != float("inf") else total_frames
        )
        self.reset_at_each_iter = reset_at_each_iter
        self.init_random_frames = (
            int(init_random_frames) if init_random_frames is not None else 0
        )
        if (
            init_random_frames is not None
            and init_random_frames % frames_per_batch != 0
            and RL_WARNINGS
        ):
            warnings.warn(
                f"init_random_frames ({init_random_frames}) is not exactly a multiple of frames_per_batch ({frames_per_batch}), "
                f" this results in more init_random_frames than requested"
                f" ({-(-init_random_frames // frames_per_batch) * frames_per_batch})."
                "To silence this message, set the environment variable RL_WARNINGS to False."
            )

        self.postproc = postproc
        if self.postproc is not None and hasattr(self.postproc, "to"):
            self.postproc.to(self.storing_device)
        if frames_per_batch % self.n_env != 0 and RL_WARNINGS:
            warnings.warn(
                f"frames_per_batch ({frames_per_batch}) is not exactly divisible by the number of batched environments ({self.n_env}), "
                f" this results in more frames_per_batch per iteration that requested"
                f" ({-(-frames_per_batch // self.n_env) * self.n_env})."
                "To silence this message, set the environment variable RL_WARNINGS to False."
            )
        self.requested_frames_per_batch = int(frames_per_batch)
        self.frames_per_batch = -(-frames_per_batch // self.n_env)
        self.exploration_type = (
            exploration_type if exploration_type else DEFAULT_EXPLORATION_TYPE
        )
        self.return_same_td = return_same_td

        self._tensordict = env.reset()
        traj_ids = torch.arange(self.n_env, device=env.device).view(self.env.batch_size)
        self._tensordict.set(
            ("collector", "traj_ids"),
            traj_ids,
        )

        with torch.no_grad():
            self._tensordict_out = self.env.fake_tensordict()
        # If the policy has a valid spec, we use it
        if (
            hasattr(self.policy, "spec")
            and self.policy.spec is not None
            and all(v is not None for v in self.policy.spec.values(True, True))
        ):
            if any(
                key not in self._tensordict_out.keys(isinstance(key, tuple))
                for key in self.policy.spec.keys(True, True)
            ):
                # if policy spec is non-empty, all the values are not None and the keys
                # match the out_keys we assume the user has given all relevant information
                # the policy could have more keys than the env:
                policy_spec = self.policy.spec
                if policy_spec.ndim < self._tensordict_out.ndim:
                    policy_spec = policy_spec.expand(self._tensordict_out.shape)
                for key, spec in policy_spec.items(True, True):
                    if key in self._tensordict_out.keys(isinstance(key, tuple)):
                        continue
                    self._tensordict_out.set(key, spec.zero())

        else:
            # otherwise, we perform a small number of steps with the policy to
            # determine the relevant keys with which to pre-populate _tensordict_out.
            # This is the safest thing to do if the spec has None fields or if there is
            # no spec at all.
            # See #505 for additional context.
            self._tensordict_out.update(self._tensordict)
            with torch.no_grad():
                self._tensordict_out = self.policy(self._tensordict_out.to(self.device))

        if (
            self.env.base_env.scenario.parameters.is_using_prioritized_marl
            and self.env.base_env.scenario.parameters.prioritization_method.lower()
            == "marl"
        ):
            # Create the TensorDict
            priority = TensorDict(
                {
                    "loc": torch.zeros(
                        self.env.base_env.scenario.parameters.num_vmas_envs,
                        self.env.base_env.scenario.parameters.n_agents,
                        1,
                        dtype=torch.float32,
                        device=self.env.base_env.scenario.parameters.device,
                    ),
                    "sample_log_prob": torch.zeros(
                        self.env.base_env.scenario.parameters.num_vmas_envs,
                        self.env.base_env.scenario.parameters.n_agents,
                        dtype=torch.float32,
                        device=self.env.base_env.scenario.parameters.device,
                    ),
                    "scale": torch.zeros(
                        self.env.base_env.scenario.parameters.num_vmas_envs,
                        self.env.base_env.scenario.parameters.n_agents,
                        1,
                        dtype=torch.float32,
                        device=self.env.base_env.scenario.parameters.device,
                    ),
                    "scores": torch.zeros(
                        self.env.base_env.scenario.parameters.num_vmas_envs,
                        self.env.base_env.scenario.parameters.n_agents,
                        1,
                        dtype=torch.float32,
                        device=self.env.base_env.scenario.parameters.device,
                    ),
                    "ordering": torch.zeros(
                        self.env.base_env.scenario.parameters.num_vmas_envs,
                        self.env.base_env.scenario.parameters.n_agents,
                        dtype=torch.int64,
                        device=self.env.base_env.scenario.parameters.device,
                    ),
                    "state_value": torch.zeros(
                        self.env.base_env.scenario.parameters.num_vmas_envs,
                        self.env.base_env.scenario.parameters.n_agents,
                        dtype=torch.float32,
                        device=self.env.base_env.scenario.parameters.device,
                    ),
                },
                batch_size=[self.env.base_env.scenario.parameters.num_vmas_envs],
            )

            self._tensordict_out["agents", "info"].set("priority", priority)

        self._tensordict_out = (
            self._tensordict_out.unsqueeze(-1)
            .expand(*env.batch_size, self.frames_per_batch)
            .clone()
            .zero_()
        )
        # in addition to outputs of the policy, we add traj_ids to
        # _tensordict_out which will be collected during rollout
        self._tensordict_out = self._tensordict_out.to(self.storing_device)
        self._tensordict_out.set(
            ("collector", "traj_ids"),
            torch.zeros(
                *self._tensordict_out.batch_size,
                dtype=torch.int64,
                device=self.storing_device,
            ),
        )

        self._tensordict_out.refine_names(..., "time")

        if split_trajs is None:
            split_trajs = False
        self.split_trajs = split_trajs
        self._exclude_private_keys = True
        self.interruptor = interruptor
        self._frames = 0
        self._iter = -1

    def __init__(
        self,
        env,
        policy,
        priority_module=None,
        **kwargs,
    ):
        self.priority_module = priority_module

        self.helper_init(env, policy, **kwargs)

    @torch.no_grad()
    def rollout(self) -> TensorDictBase:
        """Computes a rollout in the environment using the provided policy.

        Returns:
            TensorDictBase containing the computed rollout.

        """
        if self.reset_at_each_iter:
            self._tensordict.update(self.env.reset())

        # self._tensordict.fill_(("collector", "step_count"), 0)
        self._tensordict_out.fill_(("collector", "traj_ids"), -1)
        tensordicts = []
        with set_exploration_type(self.exploration_type):
            for t in range(self.frames_per_batch):
                if (
                    self.init_random_frames is not None
                    and self._frames < self.init_random_frames
                ):
                    self.env.rand_action(self._tensordict)

                    # <Modification starts>
                    # Possibly predict the actions of surrounding agents using opponent modeling
                elif (
                    self.env.base_env.scenario.parameters.is_using_opponent_modeling
                    and not self.env.base_env.scenario.parameters.is_testing_mode
                ):
                    opponent_modeling(
                        tensordict=self._tensordict,
                        policy=self.policy,
                        n_nearing_agents_observed=self.env.base_env.scenario.parameters.n_nearing_agents_observed,
                        nearing_agents_indices=self.env.base_env.scenario.observations.nearing_agents_indices,
                        action_predictor=getattr(
                            self.env.base_env.scenario,
                            "topology_action_predictor",
                            None,
                        ),
                        parameters=self.env.base_env.scenario.parameters,
                    )
                    self.policy(self._tensordict)
                # <Modification ends>
                elif (
                    self.env.base_env.scenario.parameters.is_using_prioritized_marl
                    and self.priority_module
                ):
                    prioritized_ap_policy(
                        tensordict=self._tensordict,
                        policy=self.policy,
                        priority_module=self.priority_module,
                        nearing_agents_indices=self.env.base_env.scenario.observations.nearing_agents_indices,
                        prioritization_method=self.env.base_env.scenario.parameters.prioritization_method,
                    )
                else:
                    self.policy(self._tensordict)

                tensordict, tensordict_ = self.env.step_and_maybe_reset(
                    self._tensordict
                )
                if (
                    self.env.base_env.scenario.parameters.is_using_opponent_modeling
                    and not self.env.base_env.scenario.parameters.is_testing_mode
                ):
                    critic_obs = self._tensordict.get(
                        ("agents", "info", "critic_observation"), default=None
                    )
                    if critic_obs is not None:
                        tensordict[
                            ("agents", "info", "critic_observation")
                        ] = critic_obs
                    opponent_modeling(
                        tensordict=tensordict_,
                        policy=self.policy,
                        n_nearing_agents_observed=self.env.base_env.scenario.parameters.n_nearing_agents_observed,
                        nearing_agents_indices=self.env.base_env.scenario.observations.nearing_agents_indices,
                        action_predictor=getattr(
                            self.env.base_env.scenario,
                            "topology_action_predictor",
                            None,
                        ),
                        parameters=self.env.base_env.scenario.parameters,
                    )
                    critic_obs_next = tensordict_.get(
                        ("agents", "info", "critic_observation"), default=None
                    )
                    if critic_obs_next is not None:
                        tensordict[
                            ("next", "agents", "info", "critic_observation")
                        ] = critic_obs_next
                self._tensordict = tensordict_.set(
                    "collector", tensordict.get("collector").clone(False)
                )
                tensordicts.append(
                    tensordict.to(self.storing_device, non_blocking=True)
                )

                self._update_traj_ids(tensordict)
                if (
                    self.interruptor is not None
                    and self.interruptor.collection_stopped()
                ):
                    try:
                        torch.stack(
                            tensordicts,
                            self._tensordict_out.ndim - 1,
                            out=self._tensordict_out[: t + 1],
                        )
                    except RuntimeError:
                        with self._tensordict_out.unlock_():
                            torch.stack(
                                tensordicts,
                                self._tensordict_out.ndim - 1,
                                out=self._tensordict_out[: t + 1],
                            )
                    break
            else:
                try:
                    self._tensordict_out = torch.stack(
                        tensordicts,
                        self._tensordict_out.ndim - 1,
                        out=self._tensordict_out,
                    )
                except RuntimeError:
                    with self._tensordict_out.unlock_():
                        self._tensordict_out = torch.stack(
                            tensordicts,
                            self._tensordict_out.ndim - 1,
                            out=self._tensordict_out,
                        )
        return self._tensordict_out


class Parameters:
    """
    This class stores parameters for training and testing.
    """

    def __init__(
        self,
        # General parameters
        n_agents: int = 4,  # Number of agents
        dt: float = 0.05,  # [s] sample time
        device: str = "cpu",  # Tensor device
        scenario_name: str = "road_traffic",  # Scenario name
        seed: int = None,
        # Training parameters
        n_iters: int = 250,  # Number of training iterations
        frames_per_batch: int = 4096,  # Number of team frames collected per training iteration
        # num_envs = frames_per_batch / max_steps
        # total_frames = frames_per_batch * n_iters
        # sub_batch_size = frames_per_batch // minibatch_size
        num_epochs: int = 60,  # Optimization steps per batch of data collected
        minibatch_size: int = 512,  # Size of the mini-batches in each optimization step (2**9 - 2**12?)
        lr: float = 2e-4,  # Learning rate
        lr_action_predictor: float = None,  # Learning rate for action predictor head; defaults to lr if None
        lr_min: float = 1e-5,  # Minimum learning rate (used for scheduling of learning rate)
        max_grad_norm: float = 1.0,  # Maximum norm for the gradients
        clip_epsilon: float = 0.2,  # Clip value for PPO loss
        gamma: float = 0.99,  # Discount factor from 0 to 1. A greater value corresponds to a better farsight
        lmbda: float = 0.9,  # lambda for generalised advantage estimation
        entropy_eps: float = 1e-4,  # Coefficient of the entropy term in the PPO loss
        topology_loss_weight: float = 0.5,  # Weight to integrate topology BCE into PPO total loss
        max_steps: int = 128,  # Episode steps before done
        total_frames: int = None,  # Total frame for one training, equals `frames_per_batch * n_iters`
        num_vmas_envs: int = None,  # Number of vectorized environments
        scenario_type: str = "intersection_1",  # One of {"CPM_entire", "CPM_mixed", "intersection_1", ...}. See SCENARIOS in utilities/constants.py for more scenarios.
        # "CPM_entire": Entire map of the CPM Lab
        # "CPM_mixed": Intersection, merge-in, and merge-out of the CPM Lab. Probability defined in `cpm_scenario_probabilities`
        # "intersection_1": Intersection with ID 1
        episode_reward_mean_current: float = 0.00,  # Achieved mean episode reward (total/n_agents)
        episode_reward_intermediate: float = -1e3,  # A arbitrary, small initial value
        is_prb: bool = False,  # Whether to enable prioritized replay buffer
        is_challenging_initial_state_buffer=False,  # Whether to enable challenging initial state buffer
        cpm_scenario_probabilities=[
            1.0,
            0.0,
            0.0,
        ],  # Probabilities of training agents in intersection, merge-in, or merge-out scenario
        n_steps_stored: int = 10,  # Store previous `n_steps_stored` steps of states
        # Observation
        n_points_short_term: int = 3,  # Number of points that build a short-term reference path
        # When enabled, prepend current position to short-term refs for topology
        is_append_current_pos_to_short_refs_for_topology: bool = True,
        is_partial_observation: bool = True,  # Whether to enable partial observation
        n_nearing_agents_observed: int = 2,  # Number of nearing agents to be observed (consider limited sensor range)
        n_topology_nearing_agents_observed: int = None,  # Number of nearing agents to be observed by topology network (can be larger than policy)
        # Parameters for ablation studies
        is_ego_view: bool = True,  # Ego view or bird view
        is_apply_mask: bool = True,  # Whether to mask distant agents
        is_observe_distance_to_agents: bool = True,  # Whether to observe the distance to other agents
        is_observe_distance_to_boundaries: bool = True,  # Whether to observe points on lanelet boundaries or observe the distance to labelet boundaries
        is_observe_distance_to_center_line: bool = True,  # Whether to observe the distance to reference path
        is_observe_vertices: bool = True,  # Whether to observe the vertices of other agents (or center point)
        is_add_noise: bool = True,  # Whether to add noise to observations
        is_observe_ref_path_other_agents: bool = False,  # Whether to observe the reference paths of other agents
        is_use_mtv_distance: bool = True,  # Whether to use MTV-based (Minimum Translation Vector) distance or c2c-based (center-to-center) distance.
        # Topology-based neighbor selection (policy-side)
        use_topology_neighbor_selection: bool = False,  # Enable using topology learner to select policy neighbors
        topology_selection_threshold: float = 0.5,  # Probability threshold for selecting neighbors
        # NOD phases 1-3: auxiliary only; no policy/reward/PPO behavior change
        is_using_nod_opinion: bool = True,
        nod_hidden_dim: int = 64,
        nod_lr: float = 1e-3,
        nod_tau: float = 0.25,
        nod_bifurcation_gain: float = 2.0,
        nod_observation_weight: float = 1.0,
        nod_z_epsilon: float = 1e-4,
        nod_root_iterations: int = 48,
        nod_root_tolerance: float = 1e-7,
        nod_edge_retention_steps: int = 2,
        nod_risk_temperature: float = 2.0,
        nod_risk_threshold: float = 1.25,
        nod_min_log_sigma: float = -2.5,
        nod_max_log_sigma: float = 0.0,
        nod_sensing_range: float = 0.8,
        nod_interaction_distance: float = 0.48,
        nod_conflict_radius: float = 0.08,
        nod_ttc_limit: float = 2.0,
        nod_counterfactual_horizon: int = 8,
        nod_safe_distance: float = 0.25,
        nod_label_slope: float = 12.0,
        nod_label_margin: float = 0.02,
        nod_nll_weight: float = 1.0,
        nod_calibration_weight: float = 1.0,
        nod_max_grad_norm: float = 1.0,
        nod_num_epochs: int = 4,
        nod_sequence_length: int = 32,
        nod_sequence_minibatch_size: int = 8,
        nod_history_mode: str = "gru",
        nod_high_risk_score: float = 1.25,
        nod_low_risk_score: float = 0.5,
        # Visu
        is_visualize_short_term_path: bool = True,  # Whether to visualize short-term reference paths
        is_visualize_lane_boundary: bool = False,  # Whether to visualize lane boundary
        is_real_time_rendering: bool = False,  # Simulation will be paused at each time step for a certain duration to enable real-time rendering
        is_visualize_extra_info: bool = True,  # Whether to render extra information such time and time step
        is_visualize_agent_id: bool = True,  # Whether to render agent index labels
        render_title: str = "",  # The title to be rendered
        render_pause_scale: float = 1.0,  # Real-time rendering pause multiplier (visual-only, does not affect physics)
        is_visualize_future_three_points: bool = False,  # Visualize only first 3 short-term future points per agent
        is_visualize_agent_trajectory: bool = False,
        agent_trajectory_len: int = 25,
        agent_trajectory_thickness_m: float = 0.06,
        agent_trajectory_thickness_m_beautify: float = 4.0,
        agent_trajectory_interp_points_per_segment: int = 4,
        agent_trajectory_interp_use_catmull_rom: bool = True,
        # Optional visualization of observed neighbors (for debugging)
        is_visualize_observed_neighbors: bool = False,
        visualize_observed_neighbors_agent_index: int = 0,
        # Save/Load
        is_save_intermediate_model: bool = True,  # Whether to save intermediate model (also called checkpoint) with the hightest episode reward
        is_load_model: bool = False,  # Whether to load saved model
        is_load_final_model: bool = False,  # Whether to load the final model (last iteration)
        model_name: str = None,
        where_to_save: str = "outputs/",  # Define where to save files such as intermediate models
        is_continue_train: bool = False,  # Whether to continue training after loading an offline model
        is_save_eval_results: bool = True,  # Whether to save evaluation results such as figures and evaluation outputs
        is_load_out_td: bool = False,  # Whether to load evaluation outputs
        is_testing_mode: bool = False,  # In testing mode, collisions do not terminate the current simulation
        is_save_simulation_video: bool = False,  # Whether to save simulation videos
        is_save_agent_speed: bool = False,
        agent_speed_log_path: str = None,
        agent_speed_log_interval: int = 1,
        is_using_opponent_modeling: bool = False,  # Whether to use opponent modeling to predict the actions of other agents
        is_using_prioritized_marl: bool = False,  # Whether to use prioritized MARL and action propagation.
        prioritization_method: str = "marl",  # Which method to use for generating priority ranks (options: {"marl", "random"}). Applicable only for prioritized MARL scenarios.
    ):

        self.n_agents = n_agents
        self.dt = dt

        self.device = device
        self.scenario_name = scenario_name
        self.seed = seed

        # Sampling
        self.n_iters = n_iters
        self.frames_per_batch = frames_per_batch

        if (frames_per_batch is not None) and (n_iters is not None):
            self.total_frames = frames_per_batch * n_iters

        # Training
        self.num_epochs = num_epochs
        self.minibatch_size = minibatch_size
        self.lr = lr
        # Action predictor learning rate defaults to main lr when not provided
        self.lr_action_predictor = (
            lr_action_predictor if lr_action_predictor is not None else lr
        )
        self.lr_min = lr_min
        self.max_grad_norm = max_grad_norm
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.lmbda = lmbda
        self.entropy_eps = entropy_eps
        self.topology_loss_weight = topology_loss_weight
        self.max_steps = max_steps

        self.scenario_type = scenario_type

        if (frames_per_batch is not None) and (max_steps is not None):
            self.num_vmas_envs = (
                frames_per_batch // max_steps
            )  # Number of vectorized envs. frames_per_batch should be divisible by this number,

        self.is_save_intermediate_model = is_save_intermediate_model
        self.is_load_model = is_load_model
        self.is_load_final_model = is_load_final_model

        self.episode_reward_mean_current = episode_reward_mean_current
        self.episode_reward_intermediate = episode_reward_intermediate
        self.where_to_save = where_to_save
        self.is_continue_train = is_continue_train

        self.n_points_short_term = n_points_short_term
        self.is_append_current_pos_to_short_refs_for_topology = (
            is_append_current_pos_to_short_refs_for_topology
        )
        # Observation
        self.is_partial_observation = is_partial_observation
        self.n_steps_stored = n_steps_stored
        self.n_nearing_agents_observed = n_nearing_agents_observed
        # If topology-specific neighbor count is not provided, default to policy's K
        self.n_topology_nearing_agents_observed = (
            n_topology_nearing_agents_observed
            if n_topology_nearing_agents_observed is not None
            else n_nearing_agents_observed
        )
        self.is_observe_distance_to_agents = is_observe_distance_to_agents

        self.is_testing_mode = is_testing_mode
        self.is_save_simulation_video = is_save_simulation_video
        self.is_save_agent_speed = is_save_agent_speed
        self.agent_speed_log_path = agent_speed_log_path
        self.agent_speed_log_interval = agent_speed_log_interval
        self.is_visualize_short_term_path = is_visualize_short_term_path
        self.is_visualize_lane_boundary = is_visualize_lane_boundary

        self.is_ego_view = is_ego_view
        self.is_apply_mask = is_apply_mask
        self.is_use_mtv_distance = is_use_mtv_distance
        # Topology-based neighbor selection controls
        self.use_topology_neighbor_selection = use_topology_neighbor_selection
        self.topology_selection_threshold = topology_selection_threshold
        self.is_using_nod_opinion = is_using_nod_opinion
        self.nod_hidden_dim = nod_hidden_dim
        self.nod_lr = nod_lr
        self.nod_tau = nod_tau
        self.nod_bifurcation_gain = nod_bifurcation_gain
        self.nod_observation_weight = nod_observation_weight
        self.nod_z_epsilon = nod_z_epsilon
        self.nod_root_iterations = nod_root_iterations
        self.nod_root_tolerance = nod_root_tolerance
        self.nod_edge_retention_steps = nod_edge_retention_steps
        self.nod_risk_temperature = nod_risk_temperature
        self.nod_risk_threshold = nod_risk_threshold
        self.nod_min_log_sigma = nod_min_log_sigma
        self.nod_max_log_sigma = nod_max_log_sigma
        self.nod_sensing_range = nod_sensing_range
        self.nod_interaction_distance = nod_interaction_distance
        self.nod_conflict_radius = nod_conflict_radius
        self.nod_ttc_limit = nod_ttc_limit
        self.nod_counterfactual_horizon = nod_counterfactual_horizon
        self.nod_safe_distance = nod_safe_distance
        self.nod_label_slope = nod_label_slope
        self.nod_label_margin = nod_label_margin
        self.nod_nll_weight = nod_nll_weight
        self.nod_calibration_weight = nod_calibration_weight
        self.nod_max_grad_norm = nod_max_grad_norm
        self.nod_num_epochs = nod_num_epochs
        self.nod_sequence_length = nod_sequence_length
        self.nod_sequence_minibatch_size = nod_sequence_minibatch_size
        self.nod_history_mode = nod_history_mode
        self.nod_high_risk_score = nod_high_risk_score
        self.nod_low_risk_score = nod_low_risk_score
        self.is_observe_distance_to_boundaries = is_observe_distance_to_boundaries
        self.is_observe_distance_to_center_line = is_observe_distance_to_center_line
        self.is_observe_vertices = is_observe_vertices
        self.is_add_noise = is_add_noise
        self.is_observe_ref_path_other_agents = is_observe_ref_path_other_agents

        self.is_save_eval_results = is_save_eval_results
        self.is_load_out_td = is_load_out_td

        self.is_real_time_rendering = is_real_time_rendering
        self.is_visualize_extra_info = is_visualize_extra_info
        self.is_visualize_agent_id = is_visualize_agent_id
        self.render_title = render_title
        self.render_pause_scale = render_pause_scale
        self.is_visualize_future_three_points = is_visualize_future_three_points
        self.is_visualize_agent_trajectory = is_visualize_agent_trajectory
        self.agent_trajectory_len = agent_trajectory_len
        self.agent_trajectory_thickness_m = agent_trajectory_thickness_m
        self.agent_trajectory_thickness_m_beautify = (
            agent_trajectory_thickness_m_beautify
        )
        self.agent_trajectory_interp_points_per_segment = (
            agent_trajectory_interp_points_per_segment
        )
        self.agent_trajectory_interp_use_catmull_rom = (
            agent_trajectory_interp_use_catmull_rom
        )
        self.is_visualize_observed_neighbors = is_visualize_observed_neighbors
        self.visualize_observed_neighbors_agent_index = (
            visualize_observed_neighbors_agent_index
        )

        self.is_prb = is_prb
        self.is_challenging_initial_state_buffer = is_challenging_initial_state_buffer

        self.cpm_scenario_probabilities = cpm_scenario_probabilities

        self.is_using_opponent_modeling = is_using_opponent_modeling
        self.is_using_prioritized_marl = is_using_prioritized_marl

        self.prioritization_method = prioritization_method

        if (model_name is None) and (scenario_name is not None):
            self.model_name = get_model_name(self)

    def to_dict(self):
        # Create a dictionary representation of the instance
        return self.__dict__

    @classmethod
    def from_dict(cls, dict_data):
        # Create an instance of the class from a dictionary
        # Filter out unknown keys to ensure forward compatibility with saved configs
        import inspect

        sig = inspect.signature(cls.__init__)
        valid_keys = set(sig.parameters.keys()) - {"self"}
        filtered = {k: v for k, v in dict_data.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def from_json(cls, config_file):
        with open(config_file, "r") as file:
            config = json.load(file)
            # Use from_dict to filter unknown keys for forward compatibility
            return cls.from_dict(config)


class SaveData:
    def __init__(
        self,
        parameters: Parameters,
        episode_reward_mean_list: [] = None,
        collision_agents_rate_list: [] = None,
        collision_lanelets_rate_list: [] = None,
        collision_total_rate_list: [] = None,
        nod_metrics_list: [] = None,
    ):
        self.parameters = parameters
        self.episode_reward_mean_list = episode_reward_mean_list
        self.collision_agents_rate_list = collision_agents_rate_list
        self.collision_lanelets_rate_list = collision_lanelets_rate_list
        self.collision_total_rate_list = collision_total_rate_list
        self.nod_metrics_list = nod_metrics_list

    def to_dict(self):
        return {
            "parameters": self.parameters.to_dict(),  # Convert Parameters instance to dict
            "episode_reward_mean_list": self.episode_reward_mean_list,
            "collision_agents_rate_list": self.collision_agents_rate_list,
            "collision_lanelets_rate_list": self.collision_lanelets_rate_list,
            "collision_total_rate_list": self.collision_total_rate_list,
            "nod_metrics_list": self.nod_metrics_list,
        }

    @classmethod
    def from_dict(cls, dict_data):
        parameters = Parameters.from_dict(
            dict_data["parameters"]
        )  # Convert dict back to Parameters instance
        return cls(
            parameters=parameters,
            episode_reward_mean_list=dict_data.get("episode_reward_mean_list", None),
            collision_agents_rate_list=dict_data.get("collision_agents_rate_list", None),
            collision_lanelets_rate_list=dict_data.get(
                "collision_lanelets_rate_list", None
            ),
            collision_total_rate_list=dict_data.get("collision_total_rate_list", None),
            nod_metrics_list=dict_data.get("nod_metrics_list", None),
        )


class PriorityModule:
    def __init__(self, env: TransformedEnvCustom = None, mappo: bool = True):
        """
        Initializes the PriorityModule, which is responsible for computing the priority ordering of agents
        and their scores using a neural network policy. It also sets up a PPO loss module with an actor-critic
        architecture and GAE (Generalized Advantage Estimation) for reinforcement learning optimization.

        Parameters:
        -----------
        env : TransformedEnvCustom
            The environment containing the observation specifications and other scenario parameters.
        mappo : bool, optional
            Flag to indicate whether to use centralised learning in the critic (MAPPO). Default is True.
        """

        self.env = env
        self.parameters = self.env.scenario.parameters

        # Tuple containing the prefix keys relevant to the priority variables
        self.prefix_key = ("agents", "info", "priority")

        observation_key = get_priority_observation_key()

        policy_net = torch.nn.Sequential(
            MultiAgentMLP(
                n_agent_inputs=env.observation_spec[observation_key].shape[-1],
                n_agent_outputs=2 * 1,  # 2 * n_actions_per_agents
                n_agents=self.parameters.n_agents,
                centralised=False,  # the policies are decentralised (ie each agent will act from its observation)
                share_params=True,
                device=self.parameters.device,
                depth=2,
                num_cells=256,
                activation_class=torch.nn.Tanh,
            ),
            NormalParamExtractor(),  # this will just separate the last dimension into two outputs: a loc and a non-negative scale
        )

        policy_module = TensorDictModule(
            policy_net,
            in_keys=[observation_key],
            out_keys=[self.prefix_key + ("loc",), self.prefix_key + ("scale",)],
        )

        policy = ProbabilisticActor(
            module=policy_module,
            spec=UnboundedContinuousTensorSpec(),
            in_keys=[self.prefix_key + ("loc",), self.prefix_key + ("scale",)],
            out_keys=[self.prefix_key + ("scores",)],
            distribution_class=TanhNormal,
            distribution_kwargs={},
            return_log_prob=True,
            log_prob_key=self.prefix_key + ("sample_log_prob",),
        )  # we'll need the log-prob for the PPO loss

        critic_net = MultiAgentMLP(
            n_agent_inputs=env.observation_spec[observation_key].shape[
                -1
            ],  # Number of observations
            n_agent_outputs=1,  # 1 value per agent
            n_agents=self.parameters.n_agents,
            centralised=mappo,  # If `centralised` is True (which may help overcome the non-stationary problem in MARL), each agent will use the inputs of all agents to compute its output (n_agent_inputs * n_agents will be the number of inputs for one agent). Otherwise, each agent will only use its data as input.
            share_params=True,  # If `share_params` is True, the same MLP will be used to make the forward pass for all agents (homogeneous policies). Otherwise, each agent will use a different MLP to process its input (heterogeneous policies).
            device=self.parameters.device,
            depth=2,
            num_cells=256,
            activation_class=torch.nn.Tanh,
        )

        critic = TensorDictModule(
            module=critic_net,
            in_keys=[
                observation_key
            ],  # Note that the critic in PPO only takes the same inputs (observations) as the actor
            out_keys=[self.prefix_key + ("state_value",)],
        )

        self.policy = policy
        self.critic = critic

        if self.parameters.prioritization_method.lower() == "marl":
            loss_module = ClipPPOLoss(
                actor=policy,
                critic=critic,
                clip_epsilon=self.parameters.clip_epsilon,
                entropy_coef=self.parameters.entropy_eps,
                normalize_advantage=False,  # Important to avoid normalizing across the agent dimension
            )

            # Comment out advantage and value_target keys to use the same advantage for both base and priority loss modules
            loss_module.set_keys(  # We have to tell the loss where to find the keys
                reward=env.reward_key,
                action=self.prefix_key + ("scores",),
                sample_log_prob=self.prefix_key + ("sample_log_prob",),
                value=self.prefix_key + ("state_value",),
                # These last 2 keys will be expanded to match the reward shape
                done=("agents", "done"),
                terminated=("agents", "terminated"),
                # advantage=self.prefix_key + ("advantage",),
                # value_target=self.prefix_key + ("value_target",),
            )

            loss_module.make_value_estimator(
                ValueEstimators.GAE,
                gamma=self.parameters.gamma,
                lmbda=self.parameters.lmbda,
            )  # We build GAE
            GAE = loss_module.value_estimator  # Generalized Advantage Estimation

            optim = torch.optim.Adam(loss_module.parameters(), self.parameters.lr)

            self.GAE = GAE
            self.loss_module = loss_module
            self.optim = optim

    def rank_agents(self, scores):
        """
        Ranks agents based on their priority scores.

        The method returns the indices of agents in descending order based on their scores.

        Parameters:
        -----------
        scores : Tensor
            A tensor containing the priority scores for each agent.

        Returns:
        --------
        ordered_indices : Tensor
            A tensor containing the indices of agents ordered by their priority scores in descending order.
        """
        # Remove the last dimension of size 1
        scores = scores.squeeze(-1)

        # Get the indices that would sort the tensor along the last dimension in descending order
        ordered_indices = torch.argsort(scores, dim=-1, descending=True)

        return ordered_indices

    def __call__(self, tensordict):
        """
        Computes the priority ordering of agents based on their scores and updates the tensordict.

        The method calls the priority actor to generate scores for each agent, ranks the agents
        based on those scores, and then updates the tensordict with the priority ordering.

        Parameters:
        -----------
        tensordict : TensorDict
            A dictionary-like object containing the data for the agents.

        Returns:
        --------
        tensordict : TensorDict
            The updated tensordict with the priority ordering of agents added.
        """

        # Call the priority actor and extract the scores key from the resulting tensordict
        scores = self.policy(tensordict)[self.prefix_key + ("scores",)]

        # Generate the priority ordering of agents
        priority_ordering = self.rank_agents(scores)

        tensordict[self.prefix_key + ("ordering",)] = priority_ordering

        # Return the tensordict with the priority ordering included
        return tensordict

    def compute_losses_and_optimize(self, data):
        """
        Computes the PPO loss (actor and critic losses) and performs backpropagation with gradient clipping.

        This method computes the combined loss (objective, critic, and entropy losses) from the loss module,
        checks for invalid gradients (NaN or infinite values), performs backpropagation, applies gradient clipping,
        and then steps the optimizer to update the model parameters.

        Parameters:
        -----------
        data : TensorDict
            A dictionary-like object containing the data for computing the losses.

        Returns:
        --------
        None
        """

        loss_vals = self.loss_module(data)

        loss_value = (
            loss_vals["loss_objective"]
            + loss_vals["loss_critic"]
            + loss_vals["loss_entropy"]
        )

        assert not loss_value.isnan().any()
        assert not loss_value.isinf().any()

        loss_value.backward()

        torch.nn.utils.clip_grad_norm_(
            self.loss_module.parameters(), self.parameters.max_grad_norm
        )  # Optional

        self.optim.step()
        self.optim.zero_grad()


##################################################
## Helper Functions
##################################################
def get_path_to_save_model(parameters: Parameters):
    parameters.model_name = get_model_name(parameters=parameters)

    PATH_POLICY = parameters.where_to_save + parameters.model_name + "_policy.pth"
    PATH_CRITIC = parameters.where_to_save + parameters.model_name + "_critic.pth"
    PATH_FIG = (
        parameters.where_to_save + parameters.model_name + "_training_process.pdf"
    )
    PATH_JSON = parameters.where_to_save + parameters.model_name + "_data.json"

    if (
        parameters.is_using_prioritized_marl
        and parameters.prioritization_method.lower() == "marl"
    ):
        PATH_PRIORITY_POLICY = (
            parameters.where_to_save + parameters.model_name + "_priority_policy.pth"
        )
        PATH_PRIORITY_CRITIC = (
            parameters.where_to_save + parameters.model_name + "_priority_critic.pth"
        )

        return (
            PATH_POLICY,
            PATH_CRITIC,
            PATH_PRIORITY_POLICY,
            PATH_PRIORITY_CRITIC,
            PATH_FIG,
            PATH_JSON,
        )

    return (
        PATH_POLICY,
        PATH_CRITIC,
        PATH_FIG,
        PATH_JSON,
    )


def delete_files_with_lower_mean_reward(parameters: Parameters):
    # Regular expression pattern to match and capture the float number
    pattern = r"reward(-?[0-9]*\.?[0-9]+)_"

    # Iterate over files in the directory
    for file_name in os.listdir(parameters.where_to_save):
        match = re.search(pattern, file_name)
        if match:
            # Get the achieved mean episode reward of the saved model
            episode_reward_mean = float(match.group(1))
            if episode_reward_mean < parameters.episode_reward_intermediate:
                # Delete the saved model if its performance is worse
                os.remove(os.path.join(parameters.where_to_save, file_name))


def find_the_highest_reward_among_all_models(path):
    """This function returns the highest reward of the models stored in folder `parameters.where_to_save`"""
    # Initialize variables to track the highest reward and corresponding model
    highest_reward = float("-inf")

    pattern = r"reward(-?[0-9]*\.?[0-9]+)_"
    # Iterate through the files in the directory
    for filename in os.listdir(path):
        match = re.search(pattern, filename)
        if match:
            # Extract the reward and convert it to float
            episode_reward_mean = float(match.group(1))

            # Check if this reward is higher than the current highest
            if episode_reward_mean > highest_reward:
                highest_reward = episode_reward_mean  # Update

    return highest_reward


def save(
    parameters: Parameters,
    save_data: SaveData,
    policy=None,
    critic=None,
    priority_policy=None,
    priority_critic=None,
    topology_model=None,
    topology_action_predictor=None,
    nod_checkpoint=None,
):
    # Get paths
    paths = get_path_to_save_model(parameters=parameters)

    if (
        parameters.is_using_prioritized_marl
        and parameters.prioritization_method.lower() == "marl"
    ):
        (
            PATH_POLICY,
            PATH_CRITIC,
            PATH_PRIORITY_POLICY,
            PATH_PRIORITY_CRITIC,
            PATH_FIG,
            PATH_JSON,
        ) = paths
    else:
        PATH_POLICY, PATH_CRITIC, PATH_FIG, PATH_JSON = paths

    # Save parameters and mean episode reward list
    json_object = json.dumps(save_data.to_dict(), indent=4)  # Serializing json
    with open(PATH_JSON, "w") as outfile:  # Writing to sample.json
        outfile.write(json_object)
    # Example to how to open the saved json file
    # with open('save_data.json', 'r') as file:
    #     data = json.load(file)
    #     loaded_parameters = Parameters.from_dict(data)

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(7.0, 6.0))

    axes[0].plot(save_data.episode_reward_mean_list)
    axes[0].set_xlabel("Iterations")
    axes[0].set_ylabel("Episode mean reward")

    collision_total_rate_list = getattr(save_data, "collision_total_rate_list", None)
    if collision_total_rate_list is not None:
        axes[1].plot(collision_total_rate_list)
        axes[1].set_xlabel("Iterations")
        axes[1].set_ylabel("Total collision rate")
    else:
        axes[1].set_visible(False)

    fig.tight_layout()
    fig.savefig(PATH_FIG, format="pdf", bbox_inches="tight")
    plt.close(fig)
    # plt.savefig(PATH_FIG, format="pdf")

    # Save models
    if (policy != None) & (critic != None):
        # Save current models
        torch.save(policy.state_dict(), PATH_POLICY)
        torch.save(critic.state_dict(), PATH_CRITIC)

        if (
            parameters.is_using_prioritized_marl
            and parameters.prioritization_method.lower() == "marl"
        ):
            if (priority_policy != None) & (priority_critic != None):
                # Save current models
                torch.save(priority_policy.state_dict(), PATH_PRIORITY_POLICY)
                torch.save(priority_critic.state_dict(), PATH_PRIORITY_CRITIC)

        # Save topology model if provided
        if topology_model is not None:
            PATH_TOPOLOGY = (
                parameters.where_to_save + parameters.model_name + "_topology.pth"
            )
            torch.save(topology_model.state_dict(), PATH_TOPOLOGY)

        if topology_action_predictor is not None:
            PATH_ACTION_PREDICTOR = (
                parameters.where_to_save
                + parameters.model_name
                + "_action_predictor.pth"
            )
            torch.save(topology_action_predictor.state_dict(), PATH_ACTION_PREDICTOR)

        # NOD is stored as a sidecar so existing policy/critic checkpoint files
        # and all external load commands remain backward compatible.
        if nod_checkpoint is not None:
            PATH_NOD = parameters.where_to_save + parameters.model_name + "_nod.pth"
            torch.save(nod_checkpoint, PATH_NOD)

        # Delete files with lower mean episode reward
        delete_files_with_lower_mean_reward(parameters=parameters)

    print(f"Saved model: {parameters.episode_reward_mean_current:.2f}.")


def compute_td_error(tensordict_data: TensorDict, gamma=0.9):
    """
    Computes TD error.

    Args:
        gamma: discount factor
    """
    current_state_values = tensordict_data["agents"]["state_value"]
    next_rewards = tensordict_data.get(("next", "agents", "reward"))
    next_state_values = tensordict_data.get(("next", "agents", "state_value"))
    done = tensordict_data.get(("next", "agents", "done"))

    # To mask out terminal states since TD error for terminal states should only consider the immediate reward without any future value
    not_done = ~done

    # See Eq. (2) of Section B EXPERIMENTAL DETAILS of paper https://doi.org/10.48550/arXiv.1511.05952
    td_error = (
        next_rewards + gamma * next_state_values * not_done - current_state_values
    )

    td_error = td_error.abs()  # Magnitude is more interesting than the actual TD error

    td_error_average_over_agents = td_error.mean(dim=-2)  # Cooperative agents

    # Normalize TD error to [0, 1] (priorities must be positive)
    td_min = td_error_average_over_agents.min()
    td_max = td_error_average_over_agents.max()
    td_error_range = td_max - td_min
    td_error_range = max(td_error_range, 1e-3)  # For numerical stability

    target_range = 10

    td_error_average_over_agents = (
        (td_error_average_over_agents - td_min) / td_error_range * target_range
    )
    td_error_average_over_agents = torch.clamp(
        td_error_average_over_agents, 1e-3, target_range
    )  # For numerical stability

    return td_error_average_over_agents


def opponent_modeling(
    tensordict,
    policy,
    n_nearing_agents_observed,
    nearing_agents_indices,
    noise_percentage: float = 0,
    action_predictor=None,
    parameters=None,
):
    """
    Opponent modeling for multi-agent training.

    Priority:
    - If a topology-based `action_predictor` is provided and the required info
      tensors exist in `tensordict`, use it to predict neighbor actions and
      append them to each ego agent's observation.
    - Otherwise, fall back to the original scheme: run the policy once to get
      tentative actions for all agents, then append neighbors' actions to ego
      observations.

    Reference: Raileanu et al., ICML 2018.
    """

    # Determine which observation field to update
    obs_key = (
        get_observation_key(parameters)
        if parameters is not None
        else ("agents", "observation")
    )

    # Choose sample (env and agent) to inspect
    env_idx = 0
    ego_idx = 0
    try:
        if parameters is not None and hasattr(
            parameters, "visualize_observed_neighbors_agent_index"
        ):
            ego_idx = int(parameters.visualize_observed_neighbors_agent_index)
    except Exception:
        pass

    if action_predictor is not None:
        ego_obs_all = tensordict.get(("agents", "info", "ego_observation"))
        neighbors_flat_all = tensordict.get(
            ("agents", "info", "neighbors_observation_flat"), default=None
        )
        relative_feats_all = tensordict.get(
            ("agents", "info", "relative_features"), default=None
        )

        if (
            ego_obs_all is not None
            and neighbors_flat_all is not None
            and relative_feats_all is not None
        ):
            B, N, D_ego = ego_obs_all.shape
            # Clamp ego_idx to valid range
            ego_idx = max(0, min(ego_idx, N - 1))
            K = int(n_nearing_agents_observed)
            D_nei = neighbors_flat_all.shape[-1] // K
            d_rel = relative_feats_all.shape[-1]

            ego_b = ego_obs_all.contiguous().view(B * N, D_ego)
            nei_b = neighbors_flat_all.contiguous().view(B * N, K, D_nei)
            rel_b = relative_feats_all.contiguous().view(B * N, K, d_rel)

            # Capture observation before modification for one sample
            obs_before_one = tensordict[obs_key][env_idx, ego_idx].clone()

            pred_actions = action_predictor(ego_b, nei_b, rel_b)  # [BN, K, A]
            A = pred_actions.shape[-1]

            # if noise_percentage != 0:
            #     noise_std_speed = AGENTS["max_speed"] * noise_percentage
            #     noise_std_steering = math.radians(AGENTS["max_steering"]) * noise_percentage
            #     # 仅支持动作维为2（速度、转向）的情况，与现有观测布局一致
            #     if A >= 2:
            #         noise_speed = torch.randn_like(pred_actions[..., 0]) * noise_std_speed
            #         noise_steer = torch.randn_like(pred_actions[..., 1]) * noise_std_steering
            #         pred_actions = pred_actions.clone()
            #         pred_actions[..., 0] = pred_actions[..., 0] + noise_speed
            #         pred_actions[..., 1] = pred_actions[..., 1] + noise_steer

            sel_idx = tensordict.get(
                ("agents", "info", "topology_selected_indices"), default=None
            )
            if sel_idx is None:
                sel_idx = tensordict.get(
                    ("agents", "info", "neighbors_indices"), default=None
                )
            if sel_idx is None and nearing_agents_indices is not None:
                sel_idx = nearing_agents_indices
            device = ego_obs_all.device
            # order = torch.stack([torch.randperm(N, device=device) for _ in range(B)])
            ref_local_all = tensordict.get(
                ("agents", "info", "ref_local"), default=None
            )
            Bn, Nn, D = ref_local_all.shape
            T = D // 2
            short_term_all = ref_local_all.view(Bn, Nn, T, 2).to(device)
            P_full = generate_soft_labels_full_graph(short_term_all)
            P_processed, _ = break_cycles_min_cost(P_full, eps_neutralize=0.02)
            P_trans = enforce_transitivity(
                P_processed, eps_neutralize=0.02, gamma=0.5, delta=1e-3
            )
            P_final = complete_total_order(
                P_trans, eps_neutralize=0.02, gamma=0.5, delta=1e-3
            )
            orders = []
            thr = 0.5 + 1e-6
            for b in range(Bn):
                P_b = P_final[b]
                indeg = [0] * Nn
                adj = [[] for _ in range(Nn)]
                for i in range(Nn):
                    for j in range(Nn):
                        if i == j:
                            continue
                        if float(P_b[i, j].item()) > thr:
                            indeg[j] += 1
                            adj[i].append(j)
                q = sorted([i for i in range(Nn) if indeg[i] == 0])
                ord_env = []
                while q:
                    u = q.pop(0)
                    ord_env.append(u)
                    for v in adj[u]:
                        indeg[v] -= 1
                        if indeg[v] == 0:
                            idxs = q + [v]
                            q = sorted(idxs)
                if len(ord_env) < Nn:
                    rest = [i for i in range(Nn) if i not in ord_env]
                    ord_env += sorted(rest)
                orders.append(torch.tensor(ord_env, device=device, dtype=torch.int64))
                order = torch.stack(orders, dim=0)
            pos = torch.zeros_like(order)
            # pos.scatter_(1, order, torch.arange(N, device=device).unsqueeze(0).expand(B, N))
            pos.scatter_(
                1,
                order,
                torch.arange(order.size(1), device=device)
                .unsqueeze(0)
                .expand(order.size(0), order.size(1)),
            )
            random_orders = torch.stack(
                [torch.randperm(Nn, device=device) for _ in range(Bn)], dim=0
            )
            tensordict[("agents", "info", "soft_label_priority_ordering")] = order
            tensordict[("agents", "info", "random_priority_ordering")] = random_orders
            # print(
            #     "[opponent_modeling] soft_label_order:",
            #     order[env_idx].detach().cpu().tolist(),
            #     "random_order:",
            #     random_orders[env_idx].detach().cpu().tolist(),
            # )
            if isinstance(sel_idx, torch.Tensor):
                sel_idx_long = sel_idx.to(torch.int64)
                valid_mask = sel_idx_long.ge(0)
                pos_exp = pos.unsqueeze(-1).expand(B, N, K)
                sel_idx_clamped = sel_idx_long.clamp(min=0)
                nei_pos = torch.gather(pos_exp, 1, sel_idx_clamped)
                ego_pos = pos.unsqueeze(-1).expand_as(nei_pos)
                gate = ego_pos.gt(nei_pos).to(pred_actions.dtype)
                gate = gate * valid_mask.to(pred_actions.dtype)
                pred_actions_4d = pred_actions.view(B, N, K, A)
                pred_actions_4d = pred_actions_4d * gate.unsqueeze(-1)
                pred_flat = pred_actions_4d.contiguous().view(B, N, K * A)
            else:
                pred_flat = pred_actions.contiguous().view(B, N, K * A)

            obs_base = tensordict[obs_key]
            obs_actor = obs_base.clone()
            obs_actor[..., -K * A :] = 0
            obs_critic = obs_base.clone()
            obs_critic[..., -K * A :] = pred_flat
            tensordict[obs_key] = obs_actor
            tensordict[("agents", "info", "critic_observation")] = obs_critic
            # used_predictor = True
            # Prepare debug info for a single sample
            bn_index = env_idx * N + ego_idx
            pred_one = pred_actions[bn_index].detach().cpu()
            obs_after_one = tensordict[obs_key][env_idx, ego_idx].detach().cpu()
            # print("[opponent_modeling] Using topology_action_predictor to populate neighbor actions.")
            # print(f"  Sample env={env_idx}, agent={ego_idx}, K={K}, A={A}")
            # print(f"  Before obs (len={obs_before_one.numel()}): {obs_before_one.detach().cpu().numpy()}")
            # print(f"  Neighbor actions predicted (shape=[{K},{A}]): {pred_one.numpy()}")
            # print(f"  After  obs (len={obs_after_one.numel()}): {obs_after_one.numpy()}")
        else:
            K = int(n_nearing_agents_observed)
            A = AGENTS["n_actions"]
            obs_base = tensordict[obs_key]
            obs_actor = obs_base.clone()
            obs_actor[..., -K * A :] = 0
            tensordict[obs_key] = obs_actor
            tensordict[("agents", "info", "critic_observation")] = obs_actor.clone()
            print(
                "[opponent_modeling] Missing info tensors for predictor; using zero tail for actor and critic."
            )

    return tensordict


def get_observation_key(parameters):
    return (
        ("agents", "observation")
        if not parameters.is_using_prioritized_marl
        else ("agents", "info", "base_observation")
    )


def get_priority_observation_key():
    return ("agents", "info", "priority_observation")


def prioritized_ap_policy(
    tensordict,
    policy,
    priority_module,
    nearing_agents_indices,
    prioritization_method,
    is_add_noise_to_actions: bool = False,
):
    """
    Implements prioritized action propagation (AP) for multiple agents.
    The function first generates a priority ordering using the provided priority module.
    Then, agents are processed in this priority order, where each agent computes its action
    and propagates it to lower-priority agents as part of their observation.

    Since the policy call generates actions for all agents in all environments at once,
    the function uses a mask to isolate the actions of the agent whose turn it is to act.
    These actions are progressively combined to form the full action tensor.

    Parameters:
    -----------
    tensordict : TensorDict
        A dictionary-like object that stores the data for all agents and environments.
    policy : Callable
        The policy function used to compute the actions for the agents.
    priority_module : Callable
        A module that computes the priority ordering for agents by wrapping the process
        of generating priority scores and ranking agents according to these scores.
    nearing_agents_indices : Tensor
        A tensor indicating the neighboring agents for each agent in each environment.

    Returns:
    --------
    tensordict : TensorDict
        The updated tensordict with combined actions and observations after prioritized action propagation.
    """
    base_observation_key = ("agents", "info", "base_observation")

    device = tensordict.device

    # Clone original observation
    original_obs = tensordict[base_observation_key].clone()

    # Infer parameters
    n_envs, n_agents, obs_dim, action_dim = (
        original_obs.shape[0],
        original_obs.shape[1],
        original_obs.shape[2],
        AGENTS["n_actions"],
    )

    if prioritization_method.lower() == "marl":
        # Generate priority ordering using the priority module
        priority_module(tensordict)
        # Extract priority ordering (shape: (n_envs, n_agents)) from tensordict
        priority_ordering = tensordict[priority_module.prefix_key + ("ordering",)]
    elif prioritization_method.lower() == "random":
        # Generate a random priority ordering
        priority_ordering = torch.stack(
            [torch.randperm(n_agents) for _ in range(n_envs)]
        )
    elif prioritization_method.lower() == "soft_label":
        ref_local_all = tensordict.get(("agents", "info", "ref_local"), default=None)
        if ref_local_all is not None:
            Bn, Nn, D = ref_local_all.shape
            T = D // 2
            short_term_all = ref_local_all.view(Bn, Nn, T, 2)
            P_full = generate_soft_labels_full_graph(short_term_all)
            P_processed, _ = break_cycles_min_cost(P_full, eps_neutralize=0.02)
            P_trans = enforce_transitivity(
                P_processed, eps_neutralize=0.02, gamma=0.5, delta=1e-3
            )
            P_final = complete_total_order(
                P_trans, eps_neutralize=0.02, gamma=0.5, delta=1e-3
            )
            thr = 0.5 + 1e-6
            ordering_list = []
            for b in range(Bn):
                P_b = P_final[b]
                indeg = [0] * Nn
                adj = [[] for _ in range(Nn)]
                for i in range(Nn):
                    for j in range(Nn):
                        if i == j:
                            continue
                        if float(P_b[i, j].item()) > thr:
                            indeg[j] += 1
                            adj[i].append(j)
                q = sorted([i for i in range(Nn) if indeg[i] == 0])
                ord_env = []
                while q:
                    u = q.pop(0)
                    ord_env.append(u)
                    for v in adj[u]:
                        indeg[v] -= 1
                        if indeg[v] == 0:
                            idxs = q + [v]
                            q = sorted(idxs)
                if len(ord_env) < Nn:
                    rest = [i for i in range(Nn) if i not in ord_env]
                    ord_env += sorted(rest)
                ordering_list.append(torch.tensor(ord_env, dtype=torch.int64))
            priority_ordering = torch.stack(ordering_list, dim=0)
        else:
            priority_ordering = torch.stack(
                [torch.randperm(n_agents) for _ in range(n_envs)]
            )

    # Temporary tensors to store intermediate observations and combined results
    temp_obs = torch.zeros(n_envs, n_agents, obs_dim)
    combined_action = torch.zeros(n_envs, n_agents, action_dim)
    combined_loc = torch.zeros(n_envs, n_agents, action_dim)
    combined_sample_log_prob = torch.zeros(n_envs, n_agents)
    combined_scale = torch.zeros(n_envs, n_agents, action_dim)
    combined_obs = torch.zeros(n_envs, n_agents, obs_dim)

    # Loop through each step in the priority ordering
    for turn in range(n_agents):
        # Reset the observation
        tensordict[("agents", "info")].set("base_observation", original_obs)

        # Get the list of agents for the current turn based on priority
        current_turn_agents = priority_ordering[:, turn]

        # Create environment indices (from 0 to n_envs - 1)
        envs = torch.arange(n_envs)

        # Create a mask indicating which agents are acting in each environment
        mask = torch.zeros(n_envs, n_agents, dtype=torch.bool)
        mask[envs, current_turn_agents] = True

        # Prepare input for the policy by modifying the observations
        for env in range(n_envs):
            agent_idx = current_turn_agents[env]
            obs = tensordict[base_observation_key][env, agent_idx].clone()

            # Get the indices of neighboring agents
            current_turn_agent_neighbors = nearing_agents_indices[env, agent_idx].to(
                torch.int64
            )

            # Collect actions of the current agent's neighbors
            actions_so_far = combined_action[env, current_turn_agent_neighbors].view(-1)

            # Propagate the collected actions into the current agent's observation
            if is_add_noise_to_actions:
                noise_percentage = 0.05
                std_noise_speed = AGENTS["max_speed"] * noise_percentage
                std_noise_steering = (
                    math.radians(AGENTS["max_steering"]) * noise_percentage
                )
                noise = torch.cat(
                    [
                        std_noise_speed * torch.randn(action_dim, device=device),
                        std_noise_steering * torch.randn(action_dim, device=device),
                    ],
                    dim=-1,
                )
                obs[-len(actions_so_far) :] = actions_so_far + noise
            else:
                obs[-len(actions_so_far) :] = actions_so_far

            # Store the updated observation for the current agent in temp_obs
            temp_obs[env, agent_idx] = obs

        # Update the base observation with temp_obs for policy execution
        tensordict[("agents", "info")].set("base_observation", temp_obs)

        # Call the policy to generate actions for the agents
        policy(tensordict)

        # Use the mask to place the data into the correct positions in the combined tensors
        combined_action[mask] = tensordict[("agents", "action")][mask]
        combined_loc[mask] = tensordict[("agents", "loc")][mask]
        combined_sample_log_prob[mask] = tensordict[("agents", "sample_log_prob")][mask]
        combined_scale[mask] = tensordict[("agents", "scale")][mask]
        combined_obs[mask] = tensordict[base_observation_key][mask]

    # Write the combined actions back to the tensordict
    tensordict[("agents", "action")] = combined_action
    tensordict[("agents", "loc")] = combined_loc
    tensordict[("agents", "sample_log_prob")] = combined_sample_log_prob
    tensordict[("agents", "scale")] = combined_scale
    tensordict[base_observation_key] = combined_obs

    return tensordict
