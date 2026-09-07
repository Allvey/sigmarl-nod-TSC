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
from utilities.nod_marl import (
    NODActorInputModule,
    NODOpinionManager,
    NOD_ACTOR_OBSERVATION_KEY,
)
from utilities.nod_marl.safety import SafetyCriticManager
from utilities.nod_marl.deadlock import DeadlockCriticManager

class BoundedNormalParamExtractor(NormalParamExtractor):
    """Keep the original scale mapping and floor, with a fixed upper bound."""

    def forward(self, tensor):
        loc, scale = super().forward(tensor)
        return loc, scale.clamp_max(1.0)


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
                f"[WARN] {nod_manager.last_load_info}; starting NOD fresh.",
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
        # If strict loading failed, the checkpoint predates the current
        # 72-dimensional physical-context contract. Reusing only part of an
        # older aggregator would couple random first-layer features to stale
        # downstream weights, so the whole aggregator keeps its initialization.
        if ".aggregator." in target_key:
            continue
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
        safety_metrics_list=[],
        deadlock_metrics_list=[],
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
    critic_observation_key = observation_key

    nod_manager = NODOpinionManager(parameters=parameters)
    scenario.nod_manager = nod_manager
    use_nod_actor = bool(
        getattr(parameters, "is_using_nod_actor", True) and nod_manager.enabled
    )
    raw_actor_observation_dim = int(
        env.observation_spec[observation_key].shape[-1]
    )
    actor_base_observation_dim = raw_actor_observation_dim
    if actor_base_observation_dim <= 0:
        raise ValueError("Actor base observation dimension must be positive")
    if use_nod_actor:
        actor_input_module = NODActorInputModule(
            observation_key=observation_key,
            base_observation_dim=actor_base_observation_dim,
            nod_manager=nod_manager,
            action_dim=int(env.action_spec.shape[-1]),
            message_dim=int(getattr(parameters, "nod_message_dim", 32)),
            message_hidden_dim=int(
                getattr(parameters, "nod_message_hidden_dim", 64)
            ),
            message_scale=float(getattr(parameters, "nod_message_scale", 0.1)),
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
        BoundedNormalParamExtractor(),
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
    safety_manager = SafetyCriticManager(
        parameters, raw_actor_observation_dim, env.n_agents,
        int(env.action_spec.shape[-1]), observation_key,
    )
    scenario.safety_manager = safety_manager
    deadlock_manager = DeadlockCriticManager(
        parameters, raw_actor_observation_dim, env.n_agents,
        int(env.action_spec.shape[-1]), observation_key,
    )
    scenario.deadlock_manager = deadlock_manager
    assert policy_parameter_ids.isdisjoint({id(p) for p in deadlock_manager.model.parameters()})
    assert policy_parameter_ids.isdisjoint({id(p) for p in safety_manager.model.parameters()})
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

        else:
            raise ValueError(
                "There is no model stored in '{parameters.where_to_save}', or the model names stored here are not following the right pattern."
            )

        safety_prefix = "final" if parameters.is_load_final_model else parameters.model_name
        safety_manager.load_if_available(
            os.path.join(parameters.where_to_save, safety_prefix + "_safety_critic.pth"),
            load_optimizer=parameters.is_continue_train,
        )
        deadlock_manager.load_if_available(
            os.path.join(parameters.where_to_save, safety_prefix + "_deadlock_critic.pth"),
            load_optimizer=parameters.is_continue_train,
        )

        if not parameters.is_continue_train:
            print(colored("[INFO] Training will not continue.", "blue"))

            nod_manager.reset_online_state()
            return env, policy, priority_module, parameters
        else:
            print(
                colored("[INFO] Training will continue with the loaded model.", "red")
            )
            critic_path = (os.path.join(parameters.where_to_save, "final_critic.pth")
                           if parameters.is_load_final_model else PATH_CRITIC)
            critic.load_state_dict(torch.load(critic_path))

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

    trainable_parameters = [
        parameter for parameter in loss_module.parameters() if parameter.requires_grad
    ]
    message_parameter_ids = (
        {
            id(parameter)
            for parameter in actor_input_module.aggregator.parameters()
            if parameter.requires_grad
        }
        if use_nod_actor
        else set()
    )
    message_parameters = [
        parameter
        for parameter in trainable_parameters
        if id(parameter) in message_parameter_ids
    ]
    if use_nod_actor and len(message_parameters) != len(message_parameter_ids):
        raise RuntimeError(
            "The NOD message aggregator is not fully registered in the PPO optimizer"
        )
    task_parameters = [
        parameter
        for parameter in trainable_parameters
        if id(parameter) not in message_parameter_ids
    ]
    if message_parameters:
        message_lr = float(getattr(parameters, "nod_message_lr", 5e-5))
        lr_floor_ratio = float(parameters.lr_min) / max(float(parameters.lr), 1e-12)
        optim = torch.optim.Adam(
            [
                {
                    "params": task_parameters,
                    "lr": float(parameters.lr),
                    "initial_lr": float(parameters.lr),
                    "minimum_lr": float(parameters.lr_min),
                },
                {
                    "params": message_parameters,
                    "lr": message_lr,
                    "initial_lr": message_lr,
                    "minimum_lr": message_lr * lr_floor_ratio,
                },
            ]
        )
    else:
        optim = torch.optim.Adam(
            [
                {
                    "params": task_parameters,
                    "lr": float(parameters.lr),
                    "initial_lr": float(parameters.lr),
                    "minimum_lr": float(parameters.lr_min),
                }
            ]
        )

    pbar = tqdm(total=parameters.n_iters, desc="epi_rew_mean = 0")

    episode_reward_mean_list = []
    collision_agents_rate_list = []
    collision_lanelets_rate_list = []
    collision_total_rate_list = []
    last_nod_metrics = {}
    nod_metrics_list = []
    safety_metrics_list = []
    deadlock_metrics_list = []

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

                combined_loss = loss_value

                assert not combined_loss.isnan().any()
                assert not combined_loss.isinf().any()

                # PPO updates the Actor, message aggregator and task Critic.
                optim.zero_grad()
                combined_loss.backward()

                # Track the PPO task loss for logging.
                last_loss_value = combined_loss.detach().mean().item()

                torch.nn.utils.clip_grad_norm_(
                    loss_module.parameters(), parameters.max_grad_norm
                )  # Optional

                optim.step()
                optim.zero_grad()

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
        # NOD learns directly from ordered physical pair features. PPO trains
        # only the stateless message aggregator from cached online context.
        nod_update_interval = max(
            1, int(getattr(parameters, "nod_update_interval", 10))
        )
        should_update_nod = (pbar.n % nod_update_interval) == 0
        if should_update_nod:
            last_nod_metrics = nod_manager.train_on_rollout(tensordict_data)
            last_nod_metrics["update_skipped"] = 0.0
        else:
            last_nod_metrics = dict(last_nod_metrics)
            last_nod_metrics["optimizer_updates"] = 0.0
            last_nod_metrics["update_skipped"] = 1.0
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

        # Safety fits its own ordered targets, with no gradients or RNG shared
        # with PPO/NOD and no contribution to the Actor objective.
        safety_metrics = safety_manager.train_on_rollout(tensordict_data)
        safety_metrics_list.append(safety_metrics)
        deadlock_metrics = deadlock_manager.train_on_rollout(tensordict_data)
        deadlock_metrics_list.append(deadlock_metrics)

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
            save_data.safety_metrics_list = safety_metrics_list
            save_data.deadlock_metrics_list = deadlock_metrics_list

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
                        nod_checkpoint=nod_manager.checkpoint_state()
                        if nod_manager.enabled
                        else None,
                        safety_checkpoint=safety_manager.checkpoint_state()
                        if safety_manager.enabled else None,
                        deadlock_checkpoint=deadlock_manager.checkpoint_state()
                        if deadlock_manager.enabled else None,
                    )
                else:
                    save(
                        parameters=parameters,
                        save_data=save_data,
                        policy=policy,
                        critic=critic,
                        nod_checkpoint=nod_manager.checkpoint_state()
                        if nod_manager.enabled
                        else None,
                        safety_checkpoint=safety_manager.checkpoint_state()
                        if safety_manager.enabled else None,
                        deadlock_checkpoint=deadlock_manager.checkpoint_state()
                        if deadlock_manager.enabled else None,
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
                    nod_checkpoint=None,
                )

        # Learning rate schedule
        for param_group in optim.param_groups:
            # Keep the message encoder on its own smaller learning-rate scale.
            progress_remaining = max(
                0.0, 1.0 - (float(pbar.n) / max(1, parameters.n_iters))
            )
            initial_lr = float(param_group["initial_lr"])
            minimum_lr = float(param_group["minimum_lr"])
            param_group["lr"] = minimum_lr + (
                initial_lr - minimum_lr
            ) * progress_remaining
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
            for metric_name, metric_value in last_nod_metrics.items():
                log_payload[f"nod/{metric_name}"] = metric_value
            for metric_name, metric_value in safety_metrics.items():
                log_payload[f"safety/{metric_name}"] = metric_value
            for metric_name, metric_value in deadlock_metrics.items():
                log_payload[f"deadlock/{metric_name}"] = metric_value
            wandb.log(log_payload, step=pbar.n)

        pbar.update()

    # Save the final model
    torch.save(policy.state_dict(), parameters.where_to_save + "final_policy.pth")
    torch.save(critic.state_dict(), parameters.where_to_save + "final_critic.pth")
    if safety_manager.enabled:
        torch.save(safety_manager.checkpoint_state(),
                   parameters.where_to_save + "final_safety_critic.pth")
    if deadlock_manager.enabled:
        torch.save(deadlock_manager.checkpoint_state(),
                   parameters.where_to_save + "final_deadlock_critic.pth")
    if nod_manager.enabled:
        torch.save(
            nod_manager.checkpoint_state(),
            parameters.where_to_save + "final_nod.pth",
        )
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
