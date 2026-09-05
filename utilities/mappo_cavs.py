# Copyright (c) 2024, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://pytorch.org/rl/stable/tutorials/multiagent_ppo.html
import time

from termcolor import colored, cprint

# Torch
import torch

# Enable anomaly detection
# torch.autograd.set_detect_anomaly(True)

# Tensordict modules
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor

# Data collection
from utilities.helper_training import SyncDataCollectorCustom, PriorityModule
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data import TensorDictPrioritizedReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage

# Env
from torchrl.envs import RewardSum
from torchrl.envs.utils import (
    check_env_specs,
)

# Multi-agent network
from torchrl.modules import (
    MultiAgentMLP,
    ProbabilisticActor,
    SafeProbabilisticTensorDictSequential,
    TanhNormal,
)

# Loss
from torchrl.objectives import ClipPPOLoss, ValueEstimators

# Utils
from tqdm import tqdm

import os
import random
import numpy as np

try:
    import wandb
except Exception:
    wandb = None

import matplotlib.pyplot as plt

# Scientific plotting
import scienceplots  # Do not remove (https://github.com/garrettj403/SciencePlots)

plt.rcParams.update(
    {"figure.dpi": "100"}
)  # Avoid DPI problem (https://github.com/garrettj403/SciencePlots/issues/60)
plt.style.use(
    ["science", "ieee"]
)  # The science + ieee styles for IEEE papers (can also be one of 'ieee' and 'science' )
# print(plt.style.available) # List all available style

from torchrl.envs.libs.vmas import VmasEnv

# Import custom classes
from utilities.helper_training import (
    Parameters,
    SaveData,
    TransformedEnvCustom,
    get_path_to_save_model,
    find_the_highest_reward_among_all_models,
    save,
    compute_td_error,
    get_observation_key,
)

from scenarios.road_traffic import ScenarioRoadTraffic
from utilities.topology_module import TopologyManager
from utilities.nod_marl import (
    NODActorInputModule,
    NODOpinionManager,
    NOD_ACTOR_OBSERVATION_KEY,
)
from utilities.topology_labels import (
    generate_soft_labels_full_graph,
    break_cycles_min_cost,
    enforce_transitivity,
    complete_total_order,
)
import math

def _generate_seed() -> int:
    return int.from_bytes(os.urandom(8), byteorder="big", signed=False)


def _seed_everything(seed: int) -> int:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def _load_action_predictor_if_available(
    path_action_predictor: str,
    topology_manager: TopologyManager,
    ego_observation: torch.Tensor,
    neighbors_flat: torch.Tensor,
    relative_features: torch.Tensor,
    k_neighbors: int,
    parameters: Parameters,
):
    if not os.path.exists(path_action_predictor):
        parameters.is_topology_action_predictor_loaded = False
        print(
            colored(
                f"[WARN] No topology action predictor checkpoint found: {path_action_predictor}",
                "yellow",
            )
        )
        return False

    if topology_manager.action_predictor is None:
        parameters.is_topology_action_predictor_loaded = False
        print(colored("[WARN] Topology action predictor is not initialized.", "yellow"))
        return False

    sample_shape = neighbors_flat.shape[:-1]
    b_total = math.prod(sample_shape)
    d_ego = int(ego_observation.shape[-1])
    d_nei = int(neighbors_flat.shape[-1] // k_neighbors)
    d_rel = int(relative_features.shape[-1])
    ego_b = ego_observation.contiguous().view(b_total, d_ego)
    nei_b = neighbors_flat.contiguous().view(b_total, k_neighbors, d_nei)
    rel_b = relative_features.contiguous().view(b_total, k_neighbors, d_rel)

    with torch.no_grad():
        topology_manager.action_predictor(ego_b, nei_b, rel_b)

    sd = torch.load(path_action_predictor, map_location=parameters.device)
    topology_manager.action_predictor.load_state_dict(sd)
    topology_manager.action_predictor.eval()
    try:
        topology_manager.scenario.topology_action_predictor = (
            topology_manager.action_predictor
        )
    except Exception:
        pass
    print(
        colored(
            f"[INFO] Loaded topology action predictor: {path_action_predictor}",
            "blue",
        )
    )
    parameters.is_topology_action_predictor_loaded = True
    return True


def _load_nod_if_available(
    path_nod: str,
    nod_manager: NODOpinionManager,
    parameters: Parameters,
    *,
    load_optimizer: bool,
):
    """Load an optional NOD sidecar without affecting old checkpoints."""

    if not nod_manager.enabled or not os.path.exists(path_nod):
        return False
    checkpoint = torch.load(path_nod, map_location=parameters.device)
    loaded = nod_manager.load_checkpoint(
        checkpoint, load_optimizer=load_optimizer
    )
    if not loaded:
        print(
            colored(
                f"[WARN] {nod_manager.last_load_info}; starting NOD v2 fresh.",
                "yellow",
            )
        )
        return False
    print(colored(f"[INFO] Loaded NOD opinion model: {path_nod}", "blue"))
    return True


def _load_policy_checkpoint(
    path: str,
    policy,
    parameters: Parameters,
    *,
    actor_base_observation_dim: int,
    use_nod_actor: bool,
):
    """Load current Stage-4 or migrate a pre-Stage-4 actor checkpoint.

    Legacy Actor layers are matched by their ``agent_networks`` suffix. The
    original observation columns are retained in the enlarged first layer;
    the new opinion-message and previous-action columns keep their normal
    initialization. This changes no external checkpoint or command interface.
    """

    state_dict = torch.load(path, map_location=parameters.device)
    try:
        policy.load_state_dict(state_dict)
        return "loaded"
    except RuntimeError:
        if not use_nod_actor:
            raise

    current = policy.state_dict()
    migrated = dict(current)
    matched = 0
    expanded = 0
    for target_key, target_value in current.items():
        if (
            target_key in state_dict
            and state_dict[target_key].shape == target_value.shape
        ):
            migrated[target_key] = state_dict[target_key]
            matched += 1
            continue
        marker = "agent_networks."
        if marker not in target_key:
            continue
        suffix = target_key[target_key.index(marker) :]
        candidates = [
            value
            for source_key, value in state_dict.items()
            if marker in source_key
            and source_key[source_key.index(marker) :] == suffix
        ]
        if len(candidates) != 1:
            continue
        source_value = candidates[0]
        if source_value.shape == target_value.shape:
            migrated[target_key] = source_value
            matched += 1
        elif (
            source_value.ndim == 2
            and target_value.ndim == 2
            and source_value.shape[0] == target_value.shape[0]
            and source_value.shape[1] >= actor_base_observation_dim
            and target_value.shape[1] > source_value.shape[1]
        ):
            expanded_value = target_value.clone()
            expanded_value[:, :actor_base_observation_dim] = source_value[
                :, :actor_base_observation_dim
            ]
            migrated[target_key] = expanded_value
            matched += 1
            expanded += 1
    if matched == 0:
        policy.load_state_dict(state_dict)
    policy.load_state_dict(migrated)
    print(
        colored(
            "[INFO] Migrated a pre-Stage-4 policy checkpoint "
            f"({matched} Actor tensors reused, {expanded} input layer expanded).",
            "blue",
        )
    )
    return "migrated"


def mappo_cavs(parameters: Parameters):
    seed = getattr(parameters, "seed", None)
    if seed is None:
        seed = _generate_seed()
        parameters.seed = seed
    seed = _seed_everything(seed)
    print(colored("[INFO] Seed:", "black"), colored(f"{seed}", "blue"))

    scenario = ScenarioRoadTraffic()

    scenario.parameters = parameters

    # Using multi-threads to handle file writing
    # pool = ThreadPoolExecutor(128)

    env = VmasEnv(
        scenario=scenario,
        num_envs=parameters.num_vmas_envs,
        continuous_actions=True,  # VMAS supports both continuous and discrete actions
        max_steps=parameters.max_steps,
        device=parameters.device,
        # Scenario kwargs
        n_agents=parameters.n_agents,  # These are custom kwargs that change for each VMAS scenario, see the VMAS repo to know more.
    )

    save_data = SaveData(
        parameters=parameters,
        episode_reward_mean_list=[],
        collision_agents_rate_list=[],
        collision_lanelets_rate_list=[],
        collision_total_rate_list=[],
        nod_metrics_list=[],
    )

    env = TransformedEnvCustom(
        env,
        RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]),
    )

    check_env_specs(env)
    try:
        env.set_seed(seed)
    except Exception:
        pass

    observation_key = get_observation_key(parameters)
    critic_observation_key = (
        ("agents", "info", "critic_observation")
        if parameters.is_using_opponent_modeling
        else observation_key
    )

    topology_manager = TopologyManager(parameters=parameters, scenario=scenario)
    nod_manager = NODOpinionManager(
        parameters=parameters,
        relation_feature_dim=topology_manager.relation_dim,
        action_dim=topology_manager.action_dim,
    )
    scenario.topology_manager = topology_manager
    scenario.nod_manager = nod_manager
    use_nod_actor = bool(
        getattr(parameters, "is_using_nod_actor", True) and nod_manager.enabled
    )
    raw_actor_observation_dim = int(
        env.observation_spec[observation_key].shape[-1]
    )
    opponent_tail_dim = (
        int(parameters.n_nearing_agents_observed) * int(env.action_spec.shape[-1])
        if parameters.is_using_opponent_modeling
        else 0
    )
    actor_base_observation_dim = raw_actor_observation_dim - opponent_tail_dim
    if actor_base_observation_dim <= 0:
        raise ValueError("Actor base observation dimension must be positive")
    if use_nod_actor:
        actor_input_module = NODActorInputModule(
            observation_key=observation_key,
            base_observation_dim=actor_base_observation_dim,
            topology_manager=topology_manager,
            nod_manager=nod_manager,
            message_dim=int(getattr(parameters, "nod_message_dim", 32)),
            message_hidden_dim=int(
                getattr(parameters, "nod_message_hidden_dim", 64)
            ),
        ).to(parameters.device)
        actor_observation_key = NOD_ACTOR_OBSERVATION_KEY
        actor_observation_dim = actor_input_module.actor_input_dim
    else:
        actor_input_module = None
        actor_observation_key = observation_key
        actor_observation_dim = raw_actor_observation_dim

    policy_net = torch.nn.Sequential(
        MultiAgentMLP(
            n_agent_inputs=actor_observation_dim,
            n_agent_outputs=(2 * env.action_spec.shape[-1]),  # 2 * n_actions_per_agents
            n_agents=env.n_agents,
            centralised=False,  # the policies are decentralised (ie each agent will act from its observation)
            share_params=True,  # sharing parameters means that agents will all share the same policy, which will allow them to benefit from each other’s experiences, resulting in faster training. On the other hand, it will make them behaviorally homogenous, as they will share the same model
            device=parameters.device,
            depth=2,
            num_cells=256,
            activation_class=torch.nn.Tanh,
        ),
        NormalParamExtractor(),  # this will just separate the last dimension into two outputs: a `loc` and a non-negative `scale``, used as parameters for a normal distribution (mean and standard deviation)
    )

    # print("policy_net:", policy_net, "\n")

    policy_module = TensorDictModule(
        policy_net,
        in_keys=[actor_observation_key],
        out_keys=[
            ("agents", "loc"),
            ("agents", "scale"),
        ],  # represents the parameters of the policy distribution for each agent
    )

    # Use a probabilistic actor allows for exploration
    probabilistic_actor = ProbabilisticActor(
        module=policy_module,
        spec=env.unbatched_action_spec,
        in_keys=[("agents", "loc"), ("agents", "scale")],
        out_keys=[env.action_key],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "min": env.unbatched_action_spec[env.action_key].space.low,
            "max": env.unbatched_action_spec[env.action_key].space.high,
        },
        return_log_prob=True,
        log_prob_key=(
            "agents",
            "sample_log_prob",
        ),  # log probability favors numerical stability and gradient calculation
    )  # we'll need the log-prob for the PPO loss
    policy = (
        SafeProbabilisticTensorDictSequential(
            actor_input_module, *probabilistic_actor.module
        )
        if use_nod_actor
        else probabilistic_actor
    )

    mappo = True  # IPPO (Independent PPO) if False

    critic_net = MultiAgentMLP(
        n_agent_inputs=env.observation_spec[observation_key].shape[
            -1
        ],  # Number of observations
        n_agent_outputs=1,  # 1 value per agent
        n_agents=env.n_agents,
        centralised=mappo,  # If `centralised` is True (which may help overcome the non-stationary problem in MARL), each agent will use the inputs of all agents to compute its output (n_agent_inputs * n_agents will be the number of inputs for one agent). Otherwise, each agent will only use its data as input.
        share_params=True,  # If `share_params` is True, the same MLP will be used to make the forward pass for all agents (homogeneous policies). Otherwise, each agent will use a different MLP to process its input (heterogeneous policies).
        device=parameters.device,
        depth=2,
        num_cells=256,
        activation_class=torch.nn.Tanh,
    )

    critic = TensorDictModule(
        module=critic_net,
        in_keys=[critic_observation_key],
        out_keys=[("agents", "state_value")],
    )

    if (
        parameters.is_using_prioritized_marl
        and parameters.prioritization_method.lower() == "marl"
    ):
        priority_module = PriorityModule(env=env, mappo=mappo)
    else:
        priority_module = None

    policy_parameter_ids = {id(parameter) for parameter in policy.parameters()}
    nod_parameter_ids = {id(parameter) for parameter in nod_manager.model.parameters()}
    assert policy_parameter_ids.isdisjoint(nod_parameter_ids), (
        "The recurrent NOD model must remain detached from PPO"
    )

    # Check if the directory defined to store the model exists and create it if not
    if not os.path.exists(parameters.where_to_save):
        os.makedirs(parameters.where_to_save)
        print(
            colored(
                "[INFO] Created a new directory to save the trained model:", "black"
            ),
            colored(f"{parameters.where_to_save}", "blue"),
        )

    # Load an existing model or train a new model?
    if parameters.is_load_model:
        # Load the model with the highest reward in the folder `parameters.where_to_save`
        highest_reward = find_the_highest_reward_among_all_models(
            parameters.where_to_save
        )
        parameters.episode_reward_mean_current = highest_reward  # Update the parameter so that the right filename will be returned later on
        if highest_reward is not float("-inf"):
            if parameters.is_load_final_model:
                _load_policy_checkpoint(
                    parameters.where_to_save + "final_policy.pth",
                    policy,
                    parameters,
                    actor_base_observation_dim=actor_base_observation_dim,
                    use_nod_actor=use_nod_actor,
                )
                print(
                    colored(
                        "[INFO] Loaded the final model (instead of the intermediate model with the highest episode reward)",
                        "red",
                    )
                )

                if priority_module:
                    priority_module.policy.load_state_dict(
                        torch.load(
                            parameters.where_to_save + "final_priority_policy.pth"
                        )
                    )

                    print(
                        colored(
                            "[INFO] Loaded the final priority model (instead of the intermediate model with the highest episode reward)",
                            "red",
                        )
                    )

                # 加载最终拓扑模型权重（如果存在）
                PATH_TOPOLOGY_FINAL = parameters.where_to_save + "final_topology.pth"
                if os.path.exists(PATH_TOPOLOGY_FINAL):
                    # 通过一次极短 rollout 推断维度后实例化并加载
                    probe_out = env.rollout(
                        max_steps=1,
                        policy=policy,
                        priority_module=priority_module,
                        auto_cast_to_device=True,
                        break_when_any_done=False,
                    )
                    td_probe = (
                        probe_out[0] if isinstance(probe_out, tuple) else probe_out
                    )
                    ego_obs_p = td_probe.get(("agents", "info", "ego_observation"))
                    # Prefer topology-specific keys if available
                    nei_flat_p = td_probe.get(
                        ("agents", "info", "topology_neighbors_observation_flat"),
                        default=None,
                    )
                    rel_p = td_probe.get(
                        ("agents", "info", "topology_relative_features"), default=None
                    )
                    if (nei_flat_p is None) or (rel_p is None):
                        # Fallback to policy-sized neighbors if topology keys are absent
                        nei_flat_p = td_probe.get(
                            ("agents", "info", "neighbors_observation_flat")
                        )
                        rel_p = td_probe.get(("agents", "info", "relative_features"))
                    Kp = getattr(
                        parameters,
                        "n_topology_nearing_agents_observed",
                        parameters.n_nearing_agents_observed,
                    )
                    topology_manager.ensure_initialized(
                        ego_obs_p, nei_flat_p, rel_p, Kp
                    )
                    sd = torch.load(PATH_TOPOLOGY_FINAL, map_location=parameters.device)
                    topology_manager.learner.load_state_dict(sd)
                    topology_manager.learner.eval()
                    scenario.topology_learner = topology_manager.learner
                    _load_action_predictor_if_available(
                        parameters.where_to_save + "final_action_predictor.pth",
                        topology_manager,
                        ego_obs_p,
                        nei_flat_p,
                        rel_p,
                        Kp,
                        parameters,
                    )
                    print(
                        colored(
                            f"[INFO] Loaded final topology model: {PATH_TOPOLOGY_FINAL}",
                            "blue",
                        )
                    )
                _load_nod_if_available(
                    parameters.where_to_save + "final_nod.pth",
                    nod_manager,
                    parameters,
                    load_optimizer=parameters.is_continue_train,
                )

            else:
                # Get paths based on the parameter configuration
                paths = get_path_to_save_model(parameters=parameters)

                # Destructure paths based on whether prioritized MARL is enabled
                if priority_module:
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

                # Load the saved model state dictionaries for policy and critic
                _load_policy_checkpoint(
                    PATH_POLICY,
                    policy,
                    parameters,
                    actor_base_observation_dim=actor_base_observation_dim,
                    use_nod_actor=use_nod_actor,
                )
                print(
                    colored(
                        f"[INFO] Loaded the intermediate model {PATH_POLICY}  with the highest episode reward",
                        "blue",
                    )
                )
                _load_nod_if_available(
                    parameters.where_to_save + parameters.model_name + "_nod.pth",
                    nod_manager,
                    parameters,
                    load_optimizer=parameters.is_continue_train,
                )

                # Load priority policy and critic if prioritized (dual) MARL is enabled
                if priority_module:
                    priority_module.policy.load_state_dict(
                        torch.load(PATH_PRIORITY_POLICY)
                    )
                    print(
                        colored(
                            f"[INFO] Loaded the intermediate priority model {PATH_PRIORITY_POLICY} with the highest episode reward",
                            "blue",
                        )
                    )

                # 加载中途保存的拓扑模型权重（如果存在）
                PATH_TOPOLOGY = (
                    parameters.where_to_save + parameters.model_name + "_topology.pth"
                )
                if os.path.exists(PATH_TOPOLOGY):
                    try:
                        probe_out = env.rollout(
                            max_steps=1,
                            policy=policy,
                            priority_module=priority_module,
                            auto_cast_to_device=True,
                            break_when_any_done=False,
                        )
                        td_probe = (
                            probe_out[0] if isinstance(probe_out, tuple) else probe_out
                        )
                        ego_obs_p = td_probe.get(("agents", "info", "ego_observation"))
                        nei_flat_p = td_probe.get(
                            ("agents", "info", "topology_neighbors_observation_flat"),
                            default=None,
                        )
                        rel_p = td_probe.get(
                            ("agents", "info", "topology_relative_features"),
                            default=None,
                        )
                        if (nei_flat_p is None) or (rel_p is None):
                            nei_flat_p = td_probe.get(
                                ("agents", "info", "neighbors_observation_flat")
                            )
                            rel_p = td_probe.get(
                                ("agents", "info", "relative_features")
                            )
                        Kp = getattr(
                            parameters,
                            "n_topology_nearing_agents_observed",
                            parameters.n_nearing_agents_observed,
                        )
                        topology_manager.ensure_initialized(
                            ego_obs_p, nei_flat_p, rel_p, Kp
                        )
                        sd = torch.load(PATH_TOPOLOGY, map_location=parameters.device)
                        topology_manager.learner.load_state_dict(sd)
                        topology_manager.learner.eval()
                        scenario.topology_learner = topology_manager.learner
                        PATH_ACTION_PREDICTOR = (
                            parameters.where_to_save
                            + parameters.model_name
                            + "_action_predictor.pth"
                        )
                        _load_action_predictor_if_available(
                            PATH_ACTION_PREDICTOR,
                            topology_manager,
                            ego_obs_p,
                            nei_flat_p,
                            rel_p,
                            Kp,
                            parameters,
                        )
                        print(
                            colored(
                                f"[INFO] Loaded intermediate topology model: {PATH_TOPOLOGY}",
                                "blue",
                            )
                        )
                    except Exception as e:
                        print(
                            colored(
                                f"[WARN] Failed to load intermediate topology model: {e}",
                                "yellow",
                            )
                        )

        else:
            raise ValueError(
                "There is no model stored in '{parameters.where_to_save}', or the model names stored here are not following the right pattern."
            )

        if not parameters.is_continue_train:
            print(colored("[INFO] Training will not continue.", "blue"))

            nod_manager.reset_online_state()
            return env, policy, priority_module, parameters
        else:
            print(
                colored("[INFO] Training will continue with the loaded model.", "red")
            )
            critic.load_state_dict(torch.load(PATH_CRITIC))

            if priority_module:
                priority_module.critic.load_state_dict(torch.load(PATH_PRIORITY_CRITIC))

    # Loading probes and NOD parameter updates invalidate recurrent online
    # state. The next real rollout always starts from a coherent fresh state.
    nod_manager.reset_online_state()

    collector = SyncDataCollectorCustom(
        env,
        policy,
        priority_module=priority_module,
        device=parameters.device,
        storing_device=parameters.device,
        frames_per_batch=parameters.frames_per_batch,
        total_frames=parameters.total_frames,
    )

    if parameters.is_prb:
        replay_buffer = TensorDictPrioritizedReplayBuffer(
            alpha=0.7,
            beta=0.6,
            storage=LazyTensorStorage(
                parameters.frames_per_batch, device=parameters.device
            ),
            batch_size=parameters.minibatch_size,
            priority_key="td_error",
        )
    else:
        replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(
                parameters.frames_per_batch, device=parameters.device
            ),  # We store the frames_per_batch collected at each iteration
            sampler=SamplerWithoutReplacement(),
            batch_size=parameters.minibatch_size,  # We will sample minibatches of this size
        )

    loss_module = ClipPPOLoss(
        actor=policy,
        critic=critic,
        clip_epsilon=parameters.clip_epsilon,
        entropy_coef=parameters.entropy_eps,
        normalize_advantage=False,  # Important to avoid normalizing across the agent dimension
    )

    loss_module.set_keys(  # We have to tell the loss where to find the keys
        reward=env.reward_key,
        action=env.action_key,
        sample_log_prob=("agents", "sample_log_prob"),
        value=("agents", "state_value"),
        # These last 2 keys will be expanded to match the reward shape
        done=("agents", "done"),
        terminated=("agents", "terminated"),
    )

    loss_module.make_value_estimator(
        ValueEstimators.GAE, gamma=parameters.gamma, lmbda=parameters.lmbda
    )  # We build GAE
    GAE = loss_module.value_estimator  # Generalized Advantage Estimation

    optim = torch.optim.Adam(loss_module.parameters(), parameters.lr)

    pbar = tqdm(total=parameters.n_iters, desc="epi_rew_mean = 0")

    episode_reward_mean_list = []
    collision_agents_rate_list = []
    collision_lanelets_rate_list = []
    collision_total_rate_list = []
    last_nod_metrics = {}
    nod_metrics_list = []

    t_start = time.time()
    for tensordict_data in collector:
        tensordict_data.set(
            ("next", "agents", "done"),
            tensordict_data.get(("next", "done"))
            .unsqueeze(-1)
            .expand(tensordict_data.get_item_shape(("next", env.reward_key))),
        )
        tensordict_data.set(
            ("next", "agents", "terminated"),
            tensordict_data.get(("next", "terminated"))
            .unsqueeze(-1)
            .expand(tensordict_data.get_item_shape(("next", env.reward_key))),
        )

        # Ensure critic observation keys exist for current and next
        if parameters.is_using_opponent_modeling:
            cur_critic = tensordict_data.get(
                ("agents", "info", "critic_observation"), default=None
            )
            if cur_critic is None:
                tensordict_data[
                    ("agents", "info", "critic_observation")
                ] = tensordict_data.get(observation_key).clone()
            next_observation_key = (
                ("next", "agents", "info", "base_observation")
                if parameters.is_using_prioritized_marl
                else ("next", "agents", "observation")
            )
            next_critic = tensordict_data.get(
                ("next", "agents", "info", "critic_observation"), default=None
            )
            if next_critic is None:
                tensordict_data[
                    ("next", "agents", "info", "critic_observation")
                ] = tensordict_data.get(next_observation_key).clone()

        with torch.no_grad():
            GAE(
                tensordict_data,
                params=loss_module.critic_params,
                target_params=loss_module.target_critic_params,
            )  # Compute GAE and add it to the data

            if priority_module:
                priority_module.GAE(
                    tensordict_data,
                    params=priority_module.loss_module.critic_params,
                    target_params=priority_module.loss_module.target_critic_params,
                )

        # ---- 拓扑分支：生成 e_ij 标签并进行一次 BCE 验证训练 ----
        # 从 scenario.info() 中取出预置的结构化输入
        ego_obs = tensordict_data.get(("agents", "info", "ego_observation"))
        neighbors_flat = tensordict_data.get(
            ("agents", "info", "neighbors_observation_flat")
        )
        relative_feats = tensordict_data.get(("agents", "info", "relative_features"))

        ref_local_flat = tensordict_data.get(("agents", "info", "ref_local"))
        ref_neighbors_flat = tensordict_data.get(
            ("agents", "info", "ref_neighbors_local")
        )
        neighbors_distance = tensordict_data.get(
            ("agents", "info", "neighbors_distance")
        )
        neighbors_mask_distance = tensordict_data.get(
            ("agents", "info", "neighbors_mask_distance")
        )

        # Infer dimensions
        K = parameters.n_nearing_agents_observed
        D_ego = ego_obs.shape[-1]
        D_nei = neighbors_flat.shape[-1] // K if neighbors_flat is not None else 0
        d_rel = relative_feats.shape[-1]

        topology_manager.ensure_initialized(ego_obs, neighbors_flat, relative_feats, K)

        # 邻居观测重塑：按通用样本维（时间、环境、代理）进行 view，保证与 K 对齐
        if neighbors_flat is not None:
            sample_shape = neighbors_flat.shape[:-1]  # e.g., [T, E, A]
            neighbors_obs = neighbors_flat.contiguous().view(*sample_shape, K, D_nei)
        else:
            neighbors_obs = None

        # 展平所有样本维为单一批量维 B_total，便于送入解码器与 BCE
        sample_shape_ego = ego_obs.shape[:-1]
        B_total = math.prod(sample_shape_ego)

        ego_obs_b = ego_obs.contiguous().view(B_total, D_ego)
        neighbors_obs_b = (
            neighbors_obs.contiguous().view(B_total, K, D_nei)
            if neighbors_obs is not None
            else None
        )
        relative_feats_b = relative_feats.contiguous().view(B_total, K, d_rel)

        # 注意：拓扑 BCE 已在下方的 PPO 小批次训练中融合，这里不再单独反传/更新，避免重复训练。
        # 若需要在此处做全批量的拓扑验证，可在下方 wandb 日志汇总阶段进行。

        # Update sample priorities
        if parameters.is_prb:
            td_error = compute_td_error(tensordict_data, gamma=0.9)
            tensordict_data.set(
                ("td_error"), td_error
            )  # Adding TD error to the tensordict_data

            assert (
                tensordict_data["td_error"].min() >= 0
            ), "TD error must be greater than 0"

        data_view = tensordict_data.reshape(
            -1
        )  # Flatten the batch size to shuffle data
        replay_buffer.extend(data_view)
        # replay_buffer.update_tensordict_priority() # Not necessary, as priorities were updated automatically when calling `replay_buffer.extend()`

        last_loss_value = None
        # 迭代级别的打印开关：确保每个 iter 只输出一条记录
        printed_debug_this_iter = False
        for _ in range(parameters.num_epochs):
            # print("[DEBUG] for _ in range(parameters.num_epochs):")
            for _ in range(parameters.frames_per_batch // parameters.minibatch_size):
                # sample a batch of data
                mini_batch_data, info = replay_buffer.sample(return_info=True)

                loss_vals = loss_module(mini_batch_data)

                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )

                # ---- 将拓扑 BCE 融合进 PPO 总损失（按权重） ----
                topo_weight = float(getattr(parameters, "topology_loss_weight", 0.1))
                last_topology_bce_value = None

                if topo_weight > 0.0:
                    ego_obs_mb = mini_batch_data.get(
                        ("agents", "info", "ego_observation")
                    )
                    # Use topology-specific keys if available
                    neighbors_flat_mb = mini_batch_data.get(
                        ("agents", "info", "topology_neighbors_observation_flat"),
                        default=None,
                    )
                    relative_feats_mb = mini_batch_data.get(
                        ("agents", "info", "topology_relative_features"), default=None
                    )

                    ref_local_mb = mini_batch_data.get(("agents", "info", "ref_local"))
                    ref_neighbors_mb = mini_batch_data.get(
                        ("agents", "info", "topology_ref_neighbors_local"), default=None
                    )
                    neighbors_distance_mb = mini_batch_data.get(
                        ("agents", "info", "topology_neighbors_distance"), default=None
                    )
                    neighbors_mask_distance_mb = mini_batch_data.get(
                        ("agents", "info", "topology_neighbors_mask_distance"),
                        default=None,
                    )

                    # Fallback to policy-sized neighbors if topology keys are absent
                    if (neighbors_flat_mb is None) or (relative_feats_mb is None):
                        neighbors_flat_mb = mini_batch_data.get(
                            ("agents", "info", "neighbors_observation_flat")
                        )
                        relative_feats_mb = mini_batch_data.get(
                            ("agents", "info", "relative_features")
                        )
                        ref_neighbors_mb = mini_batch_data.get(
                            ("agents", "info", "ref_neighbors_local")
                        )
                        neighbors_distance_mb = mini_batch_data.get(
                            ("agents", "info", "neighbors_distance")
                        )
                        neighbors_mask_distance_mb = mini_batch_data.get(
                            ("agents", "info", "neighbors_mask_distance")
                        )

                    K_mb = getattr(
                        parameters,
                        "n_topology_nearing_agents_observed",
                        parameters.n_nearing_agents_observed,
                    )
                    D_ego_mb = ego_obs_mb.shape[-1]
                    D_nei_mb = (
                        neighbors_flat_mb.shape[-1] // K_mb
                        if neighbors_flat_mb is not None
                        else 0
                    )
                    d_rel_mb = relative_feats_mb.shape[-1]

                    topology_manager.ensure_initialized(
                        ego_obs_mb, neighbors_flat_mb, relative_feats_mb, K_mb
                    )

                    # 邻居观测重塑为 [B_total, K, D_nei]
                    sample_shape_mb = (
                        neighbors_flat_mb.shape[:-1]
                        if neighbors_flat_mb is not None
                        else ego_obs_mb.shape[:-1]
                    )
                    B_total_mb = math.prod(sample_shape_mb)

                    ego_b = ego_obs_mb.contiguous().view(B_total_mb, D_ego_mb)
                    nei_b = (
                        neighbors_flat_mb.contiguous().view(B_total_mb, K_mb, D_nei_mb)
                        if neighbors_flat_mb is not None
                        else None
                    )
                    rel_b = relative_feats_mb.contiguous().view(
                        B_total_mb, K_mb, d_rel_mb
                    )

                    e_labels_mb = topology_manager.generate_labels(
                        ref_local_mb.contiguous().view(B_total_mb, -1),
                        ref_neighbors_mb.contiguous().view(B_total_mb, -1),
                        neighbors_distance_mb.contiguous().view(B_total_mb, K_mb),
                        neighbors_mask_distance_mb.contiguous().view(B_total_mb, K_mb),
                        k_neighbors=K_mb,
                        n_points_short_term=(
                            parameters.n_points_short_term + 1
                            if getattr(
                                parameters,
                                "is_append_current_pos_to_short_refs_for_topology",
                                False,
                            )
                            else parameters.n_points_short_term
                        ),
                    )

                    if nei_b is not None:
                        bce_mb, edge_logits_mb = topology_manager.compute_bce(
                            ego_b, nei_b, rel_b, e_labels_mb
                        )
                        last_topology_bce_value = float(bce_mb.item())
                        # ---- 详细日志：一次迭代仅打印一次 ----
                        # ---- 打印：拓扑概率 / 真值标签 / 距离 / 是否为启发式邻居 ----
                        try:
                            # 仅在本 iter 打印一次：随机抽取展平样本
                            if not printed_debug_this_iter:
                                sample_shape_mb = neighbors_flat_mb.shape[:-1]
                                # 获取每环境智能体数 A（支持 [T,E,A] 或 [TE,A] 或退化）
                                A_mb = (
                                    int(sample_shape_mb[-1])
                                    if len(sample_shape_mb) >= 1
                                    else 1
                                )

                                # 简化为随机抽取展平样本索引；仅标注智能体编号
                                s0 = (
                                    random.randrange(int(B_total_mb))
                                    if int(B_total_mb) > 0
                                    else 0
                                )
                                agent_idx = int(s0 % max(1, int(A_mb)))

                                # 形状与样本索引信息
                                try:
                                    print(
                                        colored(
                                            f"[TOPO] edge_logits shape={list(edge_logits_mb.shape)}, B_total={int(B_total_mb)}, K={int(K_mb)}",
                                            "yellow",
                                        )
                                    )
                                    print(
                                        colored(
                                            f"[TOPO] neighbors_flat_mb shape={list(neighbors_flat_mb.shape)}",
                                            "yellow",
                                        )
                                    )
                                    print(
                                        colored(
                                            f"[TOPO] e_labels_mb shape={list(e_labels_mb.shape)}",
                                            "yellow",
                                        )
                                    )
                                    print(
                                        colored(
                                            f"[TOPO] sample index s0={int(s0)}, Agent={int(agent_idx)}",
                                            "yellow",
                                        )
                                    )
                                except Exception:
                                    pass

                                # 邻居概率（sigmoid）与标签/距离/掩码
                                edge_probs_mb = torch.sigmoid(
                                    edge_logits_mb
                                )  # [B_total, K]
                                probs0 = edge_probs_mb[s0]
                                labels0 = e_labels_mb[s0]
                                dist_b = neighbors_distance_mb.contiguous().view(
                                    B_total_mb, K_mb
                                )
                                dist0 = dist_b[s0]
                                mask_b = neighbors_mask_distance_mb.contiguous().view(
                                    B_total_mb, K_mb
                                )
                                mask0 = mask_b[s0]

                                # 打印该样本的原始 logit / 概率 / 标签 / 距离 / 掩码 / 邻居ID
                                try:
                                    sample_logits_0 = (
                                        edge_logits_mb[s0].detach().flatten().tolist()
                                    )
                                    sample_probs_0 = probs0.detach().flatten().tolist()
                                    sample_labels_0 = (
                                        labels0.detach().flatten().tolist()
                                    )
                                    sample_dist_0 = dist0.detach().flatten().tolist()
                                    sample_mask_0 = mask0.detach().flatten().tolist()
                                    print(
                                        colored(
                                            f"[TOPO] sample logits: {sample_logits_0}",
                                            "yellow",
                                        )
                                    )
                                    print(
                                        colored(
                                            f"[TOPO] sample probs:  {sample_probs_0}",
                                            "yellow",
                                        )
                                    )
                                    print(
                                        colored(
                                            f"[TOPO] sample labels: {sample_labels_0}",
                                            "yellow",
                                        )
                                    )
                                    print(
                                        colored(
                                            f"[TOPO] sample dist:   {sample_dist_0}",
                                            "yellow",
                                        )
                                    )
                                    print(
                                        colored(
                                            f"[TOPO] sample mask:   {sample_mask_0}",
                                            "yellow",
                                        )
                                    )
                                except Exception:
                                    pass

                                # 邻居编号（拓扑优先，其次策略）
                                topo_ids_mb = mini_batch_data.get(
                                    ("agents", "info", "topology_neighbors_indices"),
                                    default=None,
                                )
                                if topo_ids_mb is None:
                                    topo_ids_mb = mini_batch_data.get(
                                        ("agents", "info", "neighbors_indices"),
                                        default=None,
                                    )
                                ids_b = (
                                    topo_ids_mb.contiguous().view(B_total_mb, K_mb)
                                    if topo_ids_mb is not None
                                    else None
                                )
                                ids0 = ids_b[s0] if ids_b is not None else None
                                try:
                                    if ids0 is not None:
                                        sample_ids_0 = ids0.detach().flatten().tolist()
                                        print(
                                            colored(
                                                f"[TOPO] sample ids:   {sample_ids_0}",
                                                "yellow",
                                            )
                                        )
                                except Exception:
                                    pass

                                # 策略网络启发式邻居编号集合（用于标注 heuristic）
                                policy_ids_mb = mini_batch_data.get(
                                    ("agents", "info", "neighbors_indices"),
                                    default=None,
                                )
                                K_policy = int(
                                    getattr(parameters, "n_nearing_agents_observed", 2)
                                    or 2
                                )
                                policy_ids_b = (
                                    policy_ids_mb.contiguous().view(
                                        B_total_mb, K_policy
                                    )
                                    if policy_ids_mb is not None
                                    else None
                                )
                                policy_ids0 = (
                                    set(policy_ids_b[s0].tolist())
                                    if policy_ids_b is not None
                                    else set()
                                )

                                # 汇总统计与头部：仅打印智能体编号
                                try:
                                    num_masked = int(mask0.sum().item())
                                    num_pos = int(labels0.sum().item())
                                    print(
                                        colored(
                                            f"[TOPO] summary: pos_labels={num_pos}/{int(K_mb)}, masked={num_masked}/{int(K_mb)}",
                                            "yellow",
                                        )
                                    )
                                except Exception:
                                    pass
                                hdr_agent = colored(
                                    f"Agent {agent_idx}", "cyan", attrs=["bold"]
                                )
                                print(f"{hdr_agent}")

                                # 遍历并打印所有邻居（仅过滤自车；保留掩码并显式标注）
                                for j in range(K_mb):
                                    d_val = float(dist0[j].item())
                                    lbl_val = int(labels0[j].item())
                                    p_val = float(probs0[j].item())
                                    id_val = (
                                        int(ids0[j].item()) if ids0 is not None else j
                                    )

                                    # 仅在存在真实邻居ID时按当前样本的自车编号过滤；否则不做过滤
                                    ego_id_current = int(agent_idx)
                                    is_self = (ids0 is not None) and (
                                        id_val == ego_id_current
                                    )
                                    if is_self:
                                        continue

                                    is_masked = bool(mask0[j].item())
                                    is_heuristic = id_val in policy_ids0

                                    # 简化彩色字段：仅为 label 着色，其余默认颜色
                                    prefix = f"Neighbor-{j+1}:"
                                    id_s = f"id={id_val}"
                                    prob_s = f"prob={p_val:.3f}"
                                    label_color = "green" if lbl_val == 1 else "red"
                                    label_s = colored(
                                        f"label={lbl_val}", label_color, attrs=["bold"]
                                    )
                                    dist_s = f"dist={d_val:.3f}"
                                    heur_s = f"heuristic={is_heuristic}"
                                    masked_s = f"masked={is_masked}"

                                    print(
                                        f"{prefix} {id_s}, {prob_s}, {label_s}, {dist_s}, {heur_s}, {masked_s}"
                                    )

                                # 追加：打印动作预测网络对该智能体的邻居动作预测与真值
                                if (topology_manager.action_predictor is not None) and (
                                    nei_b is not None
                                ):
                                    # 前向预测（不影响梯度，仅用于日志）
                                    with torch.no_grad():
                                        pred_ap_mb = topology_manager.action_predictor(
                                            ego_b, nei_b, rel_b
                                        )
                                        pred0_ap = pred_ap_mb[s0]  # [K_mb, A]

                                    labels_b_ap_mb = (
                                        topology_manager.generate_action_labels(
                                            mini_batch_data
                                        )
                                    )
                                    labels0_ap = labels_b_ap_mb[s0]

                                    # 打印每个邻居的预测与真值
                                    print(
                                        colored(
                                            "[AP] neighbor action predictions (pred vs gt):",
                                            "cyan",
                                        )
                                    )
                                    pos_world_norm = getattr(
                                        scenario.normalizers, "pos_world", 1.0
                                    )
                                    use_current_idx = bool(
                                        getattr(
                                            parameters,
                                            "is_append_current_pos_to_short_refs_for_topology",
                                            False,
                                        )
                                    )
                                    nei_pts0_world = None
                                    try:
                                        if ref_neighbors_mb is not None:
                                            T_pts_ap = int(
                                                getattr(
                                                    parameters, "n_points_short_term", 3
                                                )
                                            ) + (1 if use_current_idx else 0)
                                            nei_ref_b_ap = (
                                                ref_neighbors_mb.contiguous().view(
                                                    int(B_total_mb),
                                                    int(K_mb),
                                                    T_pts_ap,
                                                    2,
                                                )
                                            )
                                            if not isinstance(
                                                pos_world_norm, torch.Tensor
                                            ):
                                                pos_world_norm_t = torch.tensor(
                                                    pos_world_norm,
                                                    device=nei_ref_b_ap.device,
                                                    dtype=nei_ref_b_ap.dtype,
                                                )
                                            else:
                                                pos_world_norm_t = pos_world_norm.to(
                                                    nei_ref_b_ap.device,
                                                    dtype=nei_ref_b_ap.dtype,
                                                )
                                            nei_ref_b_world_ap = (
                                                nei_ref_b_ap * pos_world_norm_t
                                            )
                                            nei_pts0_world = nei_ref_b_world_ap[s0]
                                    except Exception:
                                        nei_pts0_world = None

                                    for j in range(int(K_mb)):
                                        id_val = (
                                            int(ids0[j].item())
                                            if ids0 is not None
                                            else j
                                        )
                                        ego_id_current = int(agent_idx)
                                        if (ids0 is not None) and (
                                            id_val == ego_id_current
                                        ):
                                            continue
                                        pred_j = pred0_ap[j].detach().flatten().tolist()
                                        gt_j = labels0_ap[j].detach().flatten().tolist()
                                        pred_s = (
                                            f"pred=[{pred_j[0]:.3f},{pred_j[1]:.3f}]"
                                            if len(pred_j) >= 2
                                            else f"pred={pred_j}"
                                        )
                                        gt_s = (
                                            f"gt=[{gt_j[0]:.3f},{gt_j[1]:.3f}]"
                                            if len(gt_j) >= 2
                                            else f"gt={gt_j}"
                                        )
                                        pos_s = ""
                                        try:
                                            if nei_pts0_world is not None:
                                                idx_cur = 0 if use_current_idx else 0
                                                px = float(
                                                    nei_pts0_world[j, idx_cur, 0].item()
                                                )
                                                py = float(
                                                    nei_pts0_world[j, idx_cur, 1].item()
                                                )
                                                pos_s = f", pos=({px:.3f},{py:.3f})"
                                        except Exception:
                                            pos_s = ""
                                        print(
                                            f"Neighbor-{j+1} id={id_val}: {pred_s}, {gt_s}{pos_s}"
                                        )

                                # 标记为已打印，避免本 iter 再次输出
                                printed_debug_this_iter = True

                                # 追加：打印拓扑选择结果与启发式集合对比（若可用）
                                try:
                                    sel_ids_mb = mini_batch_data.get(
                                        ("agents", "info", "topology_selected_indices"),
                                        default=None,
                                    )
                                    sel_probs_mb = mini_batch_data.get(
                                        ("agents", "info", "topology_selected_probs"),
                                        default=None,
                                    )
                                    if (sel_ids_mb is not None) and (
                                        sel_probs_mb is not None
                                    ):
                                        K_policy = int(
                                            getattr(
                                                parameters,
                                                "n_nearing_agents_observed",
                                                2,
                                            )
                                            or 2
                                        )
                                        thr = float(
                                            getattr(
                                                parameters,
                                                "topology_selection_threshold",
                                                0.5,
                                            )
                                        )
                                        sel_ids_b = sel_ids_mb.contiguous().view(
                                            B_total_mb, K_policy
                                        )
                                        sel_probs_b = sel_probs_mb.contiguous().view(
                                            B_total_mb, K_policy
                                        )
                                        sel_ids0 = (
                                            sel_ids_b[s0].detach().flatten().tolist()
                                        )
                                        sel_probs0 = (
                                            sel_probs_b[s0].detach().flatten().tolist()
                                        )
                                        pairs0 = [
                                            f"{int(i)}:{float(p):.2f}"
                                            for i, p in zip(sel_ids0, sel_probs0)
                                            if int(i) >= 0
                                        ]
                                        print(
                                            colored(
                                                f"[TOPO] selected_by_topology(thr={thr:.2f}, K={K_policy}): {pairs0}",
                                                "magenta",
                                            )
                                        )
                                        print(
                                            colored(
                                                f"[TOPO] heuristic_set(K={K_policy}): {sorted(list(policy_ids0))}",
                                                "magenta",
                                            )
                                        )
                                except Exception:
                                    pass

                                # 追加：打印用于生成拓扑标签的参考轨迹世界坐标（两位小数）
                                try:
                                    T_pts = int(
                                        getattr(parameters, "n_points_short_term", 3)
                                    ) + (
                                        1
                                        if bool(
                                            getattr(
                                                parameters,
                                                "is_append_current_pos_to_short_refs_for_topology",
                                                False,
                                            )
                                        )
                                        else 0
                                    )
                                    ego_ref_b = ref_local_mb.contiguous().view(
                                        B_total_mb, T_pts, 2
                                    )
                                    nei_ref_b = ref_neighbors_mb.contiguous().view(
                                        B_total_mb, K_mb, T_pts, 2
                                    )

                                    # 反归一化为世界坐标
                                    pos_world_norm = getattr(
                                        scenario.normalizers, "pos_world", 1.0
                                    )
                                    if not isinstance(pos_world_norm, torch.Tensor):
                                        pos_world_norm = torch.tensor(
                                            pos_world_norm,
                                            device=ego_ref_b.device,
                                            dtype=ego_ref_b.dtype,
                                        )
                                    else:
                                        pos_world_norm = pos_world_norm.to(
                                            ego_ref_b.device, dtype=ego_ref_b.dtype
                                        )
                                    ego_ref_b_world = ego_ref_b * pos_world_norm
                                    nei_ref_b_world = nei_ref_b * pos_world_norm

                                    def fmt_pts_row(t_row):
                                        return (
                                            "["
                                            + ",".join(
                                                [
                                                    f"({float(t_row[k,0].item()):.2f},{float(t_row[k,1].item()):.2f})"
                                                    for k in range(t_row.size(0))
                                                ]
                                            )
                                            + "]"
                                        )

                                    ego_pts0_world = ego_ref_b_world[s0]
                                    print(
                                        colored(
                                            f"[TOPO] refs(ego, world): {fmt_pts_row(ego_pts0_world)}",
                                            "yellow",
                                        )
                                    )
                                    for j in range(int(K_mb)):
                                        id_j = (
                                            int(ids0[j].item())
                                            if ids0 is not None
                                            else j
                                        )
                                        pts_j_world = nei_ref_b_world[s0, j]
                                    print(
                                        colored(
                                            f"[TOPO] refs(nei id={id_j}, world): {fmt_pts_row(pts_j_world)}",
                                            "yellow",
                                        )
                                    )
                                except Exception:
                                    pass
                                try:
                                    st_all = getattr(
                                        scenario.ref_paths_agent_related,
                                        "short_term",
                                        None,
                                    )
                                    if st_all is not None:
                                        P_full = generate_soft_labels_full_graph(
                                            st_all.to(parameters.device)
                                        )
                                        (
                                            P_processed,
                                            removed_edges,
                                        ) = break_cycles_min_cost(
                                            P_full, eps_neutralize=0.02
                                        )
                                        P_trans = enforce_transitivity(
                                            P_processed,
                                            eps_neutralize=0.02,
                                            gamma=0.5,
                                            delta=1e-3,
                                        )
                                        P_final = complete_total_order(
                                            P_trans,
                                            eps_neutralize=0.02,
                                            gamma=0.5,
                                            delta=1e-3,
                                        )
                                        P0 = P_final[0]
                                        A_full = int(P0.shape[0])
                                        rows = []
                                        for i_row in range(A_full):
                                            row_vals = [
                                                f"{float(P0[i_row, j].item()):.3f}"
                                                for j in range(A_full)
                                            ]
                                            rows.append("[" + ",".join(row_vals) + "]")
                                        print(
                                            colored(
                                                "[TOPO-SOFT] p_ij matrix (env=0, after transitivity+total-order):",
                                                "cyan",
                                            )
                                        )
                                        for r in rows:
                                            print(r)
                                        labels = [
                                            chr(ord("A") + i) if i < 26 else f"N{i}"
                                            for i in range(A_full)
                                        ]
                                        rels = []
                                        for i in range(A_full):
                                            for j in range(A_full):
                                                if i == j:
                                                    continue
                                                pij = float(P0[i, j].item())
                                                if pij > 0.5:
                                                    rels.append((i, j, pij))
                                        print(
                                            colored(
                                                "[TOPO-SOFT] relations (i < j):", "cyan"
                                            )
                                        )
                                        for (i, j, w) in sorted(
                                            rels, key=lambda x: (-x[2], x[0], x[1])
                                        ):
                                            print(
                                                f"{labels[i]} < {labels[j]} (w={w:.3f})"
                                            )
                                        adj_less = [[] for _ in range(A_full)]
                                        indeg = [0] * A_full
                                        und_adj = [[] for _ in range(A_full)]
                                        for (i, j, w) in rels:
                                            adj_less[i].append((j, w))
                                            indeg[j] += 1
                                            und_adj[i].append(j)
                                            und_adj[j].append(i)
                                        seen = [False] * A_full
                                        comps = []
                                        for s in range(A_full):
                                            if not seen[s]:
                                                q = [s]
                                                seen[s] = True
                                                comp = []
                                                while q:
                                                    u = q.pop()
                                                    comp.append(u)
                                                    for v in und_adj[u]:
                                                        if not seen[v]:
                                                            seen[v] = True
                                                            q.append(v)
                                                comps.append(sorted(comp))
                                        print(
                                            colored(
                                                "[TOPO-SOFT] priority chains:", "cyan"
                                            )
                                        )
                                        for comp in comps:
                                            indeg_c = {u: 0 for u in comp}
                                            for u in comp:
                                                for (v, _) in adj_less[u]:
                                                    if v in indeg_c:
                                                        indeg_c[v] += 1
                                            q = [u for u in comp if indeg_c[u] == 0]
                                            order = []
                                            indeg_cur = indeg_c.copy()
                                            while q:
                                                u = q.pop(0)
                                                order.append(u)
                                                for (v, _) in adj_less[u]:
                                                    if v in indeg_cur:
                                                        indeg_cur[v] -= 1
                                                        if indeg_cur[v] == 0:
                                                            q.append(v)
                                            if len(order) == 0:
                                                continue
                                            chain_str = " < ".join(
                                                [labels[u] for u in order]
                                            )
                                            print(chain_str)
                                        has_cycle = False
                                        adj_dir = [[] for _ in range(A_full)]
                                        for (i, j, w) in rels:
                                            adj_dir[i].append(j)
                                        vis = [0] * A_full
                                        st = [0] * A_full

                                        def _dfs(u):
                                            vis[u] = 1
                                            st[u] = 1
                                            for v in adj_dir[u]:
                                                if vis[v] == 0:
                                                    if _dfs(v):
                                                        return True
                                                elif st[v] == 1:
                                                    return True
                                            st[u] = 0
                                            return False

                                        for n in range(A_full):
                                            if vis[n] == 0:
                                                if _dfs(n):
                                                    has_cycle = True
                                                    break
                                        print(
                                            colored(
                                                f"[TOPO-SOFT] has_cycle={has_cycle}",
                                                "cyan",
                                            )
                                        )
                                        if len(removed_edges) > 0:
                                            print(
                                                colored(
                                                    "[TOPO-SOFT] removed_edges (min-cost):",
                                                    "cyan",
                                                )
                                            )
                                            re0 = [
                                                e for e in removed_edges if e[0] == 0
                                            ]
                                            for (_, u, v, w) in re0:
                                                print(
                                                    f"{u} -> {v} (w={w:.3f}) neutralized"
                                                )
                                except Exception:
                                    pass
                        except Exception:
                            pass
                        # 合并损失
                        combined_loss = loss_value + topo_weight * bce_mb
                    else:
                        combined_loss = loss_value
                else:
                    combined_loss = loss_value

                assert not combined_loss.isnan().any()
                assert not combined_loss.isinf().any()

                # 反传：同时更新 PPO 与拓扑分支参数
                optim.zero_grad()
                topology_manager.zero_grad()
                combined_loss.backward()

                # Track last loss value for logging（包含拓扑项）
                last_loss_value = combined_loss.detach().mean().item()

                torch.nn.utils.clip_grad_norm_(
                    loss_module.parameters(), parameters.max_grad_norm
                )  # Optional

                optim.step()
                topology_manager.step(topo_weight)
                optim.zero_grad()

                # ---- 独立训练：邻居动作预测分支 ----
                last_action_pred_loss_value = None
                if (
                    topology_manager.action_predictor is not None
                    and topology_manager.action_optim is not None
                ):
                    last_action_pred_loss_value = (
                        topology_manager.train_action_predictor(mini_batch_data)
                    )

                if priority_module:
                    priority_module.compute_losses_and_optimize(mini_batch_data)

                if parameters.is_prb:
                    # Recalculate loss
                    with torch.no_grad():
                        GAE(
                            mini_batch_data,
                            params=loss_module.critic_params,
                            target_params=loss_module.target_critic_params,
                        )
                        if parameters.is_using_prioritized_marl:
                            priority_module.GAE(
                                tensordict_data,
                                params=priority_module.loss_module.critic_params,
                                target_params=priority_module.loss_module.target_critic_params,
                            )
                    # Recalculate the TD errors of the sampled minibatch with updated model weights and update priorities in the buffer
                    new_td_errors = compute_td_error(mini_batch_data, gamma=0.9)
                    mini_batch_data.set("td_error", new_td_errors)
                    replay_buffer.update_tensordict_priority(mini_batch_data)
        # Train NOD only after topology and action-head updates. NOD consumes
        # detached, id-aligned relation features; PPO trains only the separate
        # stateless message aggregator from collection-time cached context.
        last_nod_metrics = nod_manager.train_on_rollout(
            tensordict_data, topology_manager=topology_manager
        )
        nod_manager.reset_online_state()
        if use_nod_actor:
            actor_message = tensordict_data.get(
                ("agents", "info", "nod_actor_message"), default=None
            )
            actor_edge_mask = tensordict_data.get(
                ("agents", "info", "nod_actor_edge_mask"), default=None
            )
            actor_attention = tensordict_data.get(
                ("agents", "info", "nod_actor_message_attention"), default=None
            )
            context_ready = tensordict_data.get(
                ("agents", "info", "nod_actor_context_ready"), default=None
            )
            if actor_message is not None:
                last_nod_metrics["actor_message_l2_mean"] = float(
                    torch.linalg.vector_norm(actor_message.detach(), dim=-1).mean()
                )
            if actor_edge_mask is not None:
                last_nod_metrics["actor_active_edge_ratio"] = float(
                    actor_edge_mask.detach().float().mean()
                )
            if actor_attention is not None and actor_edge_mask is not None:
                valid_receivers = actor_edge_mask.detach().bool().any(
                    dim=-1
                )
                entropy = -(
                    actor_attention.detach().clamp_min(1e-8)
                    * actor_attention.detach().clamp_min(1e-8).log()
                ).sum(dim=-1)
                last_nod_metrics["actor_attention_entropy_mean"] = (
                    float(entropy[valid_receivers].mean())
                    if bool(valid_receivers.any())
                    else 0.0
                )
            if context_ready is not None:
                last_nod_metrics["actor_context_ready_ratio"] = float(
                    context_ready.detach().float().mean()
                )
        nod_metrics_list.append(dict(last_nod_metrics))

        collector.update_policy_weights_()  # Updates the policy weights if the policy of the data collector and the trained policy live on different devices

        # Logging
        done = tensordict_data.get(("next", "agents", "done"))
        episode_reward_mean_raw = (
            tensordict_data.get(("next", "agents", "episode_reward"))[done]
            .mean()
            .item()
        )
        episode_reward_mean = round(episode_reward_mean_raw, 2)
        episode_reward_mean_list.append(episode_reward_mean_raw)

        def _safe_get(td, key_path):
            try:
                return td.get(key_path)
            except Exception:
                return None

        coll_agents = _safe_get(
            tensordict_data,
            ("next", "agents", "info", "is_collision_with_agents"),
        )
        if coll_agents is None:
            coll_agents = _safe_get(
                tensordict_data, ("agents", "info", "is_collision_with_agents")
            )

        coll_lane = _safe_get(
            tensordict_data,
            ("next", "agents", "info", "is_collision_with_lanelets"),
        )
        if coll_lane is None:
            coll_lane = _safe_get(
                tensordict_data, ("agents", "info", "is_collision_with_lanelets")
            )

        def _rate(tensor_bool):
            try:
                if tensor_bool is None:
                    return 0.0
                return tensor_bool.to(torch.float32).reshape(-1).mean().item()
            except Exception:
                return 0.0

        collision_agents_rate = _rate(coll_agents)
        collision_lanelets_rate = _rate(coll_lane)
        collision_total_rate = min(1.0, collision_agents_rate + collision_lanelets_rate)

        collision_agents_rate_list.append(collision_agents_rate)
        collision_lanelets_rate_list.append(collision_lanelets_rate)
        collision_total_rate_list.append(collision_total_rate)

        pbar.set_description(
            f"Episode mean reward = {episode_reward_mean:.2f} | collision = {collision_total_rate:.4f}",
            refresh=False,
        )

        # env.scenario.iter = pbar.n # A way to pass the information from the training algorithm to the environment

        if parameters.is_save_intermediate_model:
            # Update the current mean episode reward
            parameters.episode_reward_mean_current = episode_reward_mean
            save_data.episode_reward_mean_list = episode_reward_mean_list
            save_data.collision_agents_rate_list = collision_agents_rate_list
            save_data.collision_lanelets_rate_list = collision_lanelets_rate_list
            save_data.collision_total_rate_list = collision_total_rate_list
            save_data.nod_metrics_list = nod_metrics_list

            if episode_reward_mean > parameters.episode_reward_intermediate:
                # Save the model if it improves the mean episode reward sufficiently enough
                parameters.episode_reward_intermediate = episode_reward_mean

                if (
                    parameters.is_using_prioritized_marl
                    and parameters.prioritization_method.lower() == "marl"
                ):
                    save(
                        parameters=parameters,
                        save_data=save_data,
                        policy=policy,
                        critic=critic,
                        priority_policy=priority_module.policy,
                        priority_critic=priority_module.critic,
                        topology_model=topology_manager.learner,
                        topology_action_predictor=topology_manager.action_predictor,
                        nod_checkpoint=nod_manager.checkpoint_state()
                        if nod_manager.enabled
                        else None,
                    )
                else:
                    save(
                        parameters=parameters,
                        save_data=save_data,
                        policy=policy,
                        critic=critic,
                        topology_model=topology_manager.learner,
                        topology_action_predictor=topology_manager.action_predictor,
                        nod_checkpoint=nod_manager.checkpoint_state()
                        if nod_manager.enabled
                        else None,
                    )
            else:
                # Save only the mean episode reward list and parameters
                parameters.episode_reward_mean_current = (
                    parameters.episode_reward_intermediate
                )
                save(
                    parameters=parameters,
                    save_data=save_data,
                    policy=None,
                    critic=None,
                    priority_policy=None,
                    priority_critic=None,
                    topology_model=None,
                    topology_action_predictor=None,
                    nod_checkpoint=None,
                )

        # Learning rate schedule
        for param_group in optim.param_groups:
            # Linear decay to lr_min
            lr_decay = (parameters.lr - parameters.lr_min) * (
                1 - (pbar.n / parameters.n_iters)
            )
            param_group["lr"] = parameters.lr_min + lr_decay
            if pbar.n % 10 == 0:
                print(f"Learning rate updated to {param_group['lr']}.")

        # Compute collision metrics and upload key metrics to wandb
        if wandb is not None and getattr(wandb, "run", None) is None:
            # Initialize wandb late if not already initialized
            try:
                wandb.init(
                    project=os.getenv("WANDB_PROJECT", "sigmarl-traffic"),
                    name=os.getenv("WANDB_RUN_NAME", "mappo-cavs"),
                )
            except Exception:
                pass

        if wandb is not None and getattr(wandb, "run", None) is not None:
            log_payload = {
                "reward/episode_mean": episode_reward_mean,
                "collision/agents_rate": collision_agents_rate,
                "collision/lanelets_rate": collision_lanelets_rate,
                "collision/total_rate": collision_total_rate,
            }

            # Log current learning rate
            try:
                current_lr = float(optim.param_groups[0]["lr"])  # main optimizer LR
                log_payload["optim/lr"] = current_lr
            except Exception:
                pass

            if last_loss_value is not None:
                log_payload["loss/total"] = last_loss_value
            # 附加：拓扑 BCE 与权重
            if (
                "last_topology_bce_value" in locals()
                and last_topology_bce_value is not None
            ):
                log_payload["loss/topology_bce"] = last_topology_bce_value
                log_payload["loss/topology_weight"] = float(
                    getattr(parameters, "topology_loss_weight", 0.1)
                )
            # 附加：动作预测分支损失与学习率
            if (
                "last_action_pred_loss_value" in locals()
                and last_action_pred_loss_value is not None
            ):
                log_payload["loss/action_pred_mse"] = last_action_pred_loss_value
                if topology_manager.action_optim is not None:
                    log_payload["optim_action/lr"] = float(
                        topology_manager.action_optim.param_groups[0]["lr"]
                    )
            for metric_name, metric_value in last_nod_metrics.items():
                log_payload[f"nod/{metric_name}"] = metric_value
            wandb.log(log_payload, step=pbar.n)

        pbar.update()

    # Save the final model
    torch.save(policy.state_dict(), parameters.where_to_save + "final_policy.pth")
    torch.save(critic.state_dict(), parameters.where_to_save + "final_critic.pth")
    if nod_manager.enabled:
        torch.save(
            nod_manager.checkpoint_state(),
            parameters.where_to_save + "final_nod.pth",
        )
    # Save final topology model if available
    if "topology_manager" in locals() and topology_manager.learner is not None:
        try:
            torch.save(
                topology_manager.learner.state_dict(),
                parameters.where_to_save + "final_topology.pth",
            )
            if topology_manager.action_predictor is not None:
                torch.save(
                    topology_manager.action_predictor.state_dict(),
                    parameters.where_to_save + "final_action_predictor.pth",
                )
        except Exception:
            pass

    if (
        parameters.is_using_prioritized_marl
        and parameters.prioritization_method.lower() == "marl"
    ):
        torch.save(
            priority_module.policy.state_dict(),
            parameters.where_to_save + "final_priority_policy.pth",
        )
        torch.save(
            priority_module.critic.state_dict(),
            parameters.where_to_save + "final_priority_critic.pth",
        )

    print(
        colored("[INFO] All files have been saved under:", "black"),
        colored(f"{parameters.where_to_save}", "red"),
    )
    # plt.show()

    training_duration = (time.time() - t_start) / 3600  # seconds to hours
    print(colored(f"[INFO] Training duration: {training_duration:.2f} hours.", "blue"))

    # Finish wandb run if active
    if wandb is not None and getattr(wandb, "run", None):
        wandb.finish()

    return env, policy, priority_module, parameters


if __name__ == "__main__":
    config_file = "config.json"
    parameters = Parameters.from_json(config_file)
    env, policy, priority_module, parameters = mappo_cavs(parameters=parameters)
