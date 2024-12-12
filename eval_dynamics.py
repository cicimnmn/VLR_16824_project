#!/usr/bin/env python3

import argparse
import pathlib
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import tqdm

from stap import agents, envs
from stap.dynamics import Dynamics, TableEnvDynamics
from stap.dynamics.utils import geodesic_loss
from stap.dynamics.utils import load as load_dynamics
from stap.utils import random, tensors
from stap.utils.transformation_utils import rotation_6d_to_matrix


@tensors.numpy_wrap
def query_policy_actor(policy: agents.RLAgent, observation: torch.Tensor, policy_args: Optional[Any]) -> torch.Tensor:
    """Numpy wrapper to query the policy actor."""
    return policy.actor.predict(policy.encoder.encode(observation.to(policy.device), policy_args))


@tensors.numpy_wrap
def query_policy_critic(policy: agents.RLAgent, observation: torch.Tensor, policy_args: Optional[Any]) -> torch.Tensor:
    """Numpy wrapper to query the policy critic."""
    action = policy.actor.predict(policy.encoder.encode(observation.to(policy.device), policy_args))
    return policy.critic.predict(policy.encoder.encode(observation.to(policy.device), policy_args), action)


def observation_str(env: envs.Env, observation: np.ndarray) -> str:
    """Converts observations to a pretty string."""
    if isinstance(env, envs.pybullet.TableEnv):
        return str(env.object_states())

    return str(observation)


def action_str(env: envs.Env, action: np.ndarray) -> str:
    """Converts actions to a pretty string."""
    if isinstance(env, envs.pybullet.TableEnv):
        primitive = env.get_primitive()
        assert isinstance(primitive, envs.pybullet.table.primitives.Primitive)
        return str(primitive.Action(action))

    return str(action)


def evaluate_episode(
    dynamics: Dynamics,
    env: envs.Env,
    seed: Optional[int] = None,
    verbose: bool = False,
    debug: bool = False,
    record: bool = False,
) -> Tuple[List[float], Dict[str, Any]]:
    """Evaluates the policy on one episode."""
    observation, reset_info = env.reset(seed=seed)
    if verbose:
        print("primitive:", env.get_primitive())
        print("reset_info:", reset_info)

    if record:
        env.record_start()

    rewards = []
    obs_diffs = []
    done = False
    while not done:
        observation, reset_info = env.reset(seed=seed)
        primitive = env.get_primitive()
        action = primitive.sample()
        new_observation, reward, terminated, truncated, step_info = env.step(action)
        if reward < 1.0:
            continue
        obs_tensor = torch.from_numpy(observation[np.newaxis, ...]).to(dynamics.device)
        action_tensor = torch.from_numpy(action[np.newaxis, ...]).to(dynamics.device)
        predicted_new_observation = dynamics.forward_eval(obs_tensor, action_tensor, primitive)
        observation_difference = predicted_new_observation.cpu().detach().numpy() - new_observation
        observation_difference_norm = np.linalg.norm(observation_difference)
        obj_1_rot_pred = rotation_6d_to_matrix(predicted_new_observation[0:1, 1:2, 3:9])
        obj_1_rot_observed = rotation_6d_to_matrix(
            torch.from_numpy(new_observation[1, 3:9][np.newaxis, np.newaxis, :]).to(dynamics.device)
        )
        rotational_loss = geodesic_loss(obj_1_rot_pred, obj_1_rot_observed)[0, 0].cpu().detach().numpy()
        if verbose:
            print("step_info:", step_info)
            print(f"reward: {reward}, terminated: {terminated}, truncated: {truncated}")
            # print magnitude of difference in observations
            print(
                "observation difference_norm:",
                observation_difference_norm,
            )
            print("rotational_loss:", rotational_loss)
            obs_diffs.append(observation_difference_norm)
        observation = new_observation
        rewards.append(reward)
        done = terminated or truncated

    if record:
        env.record_stop()

    info = {
        "obs_diff": obs_diffs,
    }
    return rewards, info


def evaluate_episodes(
    env: envs.Env,
    dynamics: Dynamics,
    num_episodes: int,
    path: pathlib.Path,
    verbose: bool,
) -> None:
    """Evaluates policy for the given number of episodes."""
    num_successes = 0
    pbar = tqdm.tqdm(
        range(num_episodes),
        desc=f"Evaluate {env.name}",
        dynamic_ncols=True,
    )
    rewards_lst = []
    obs_diffs = []
    for i in pbar:
        # Evaluate episode.
        rewards, info = evaluate_episode(dynamics, env, verbose=verbose, debug=False, record=True)
        obs_diff = info["obs_diff"]
        rewards_lst.append(rewards)
        obs_diffs.append(obs_diff)
        success = sum(rewards) > 0.0
        num_successes += success
        pbar.set_postfix({"rewards": rewards, "successes": f"{num_successes} / {num_episodes}"})

        # Save recording.
        suffix = "" if success else "_fail"
        env.record_save(path / env.name / f"eval_{i}{suffix}.gif", reset=True)

        if isinstance(env, envs.pybullet.TableEnv):
            # Save reset seed.
            with open(path / env.name / f"results_{i}.npz", "wb") as f:
                save_dict = {
                    "seed": env._seed,
                }
                np.savez_compressed(f, **save_dict)  # type: ignore
    if verbose:
        rewards_lst = np.array(rewards_lst)
        obs_diffs = np.array(obs_diffs)
        print(f"obs_diffs: {obs_diffs}")
        print(f"obs_diffs avg: {np.mean(obs_diffs)}")
        print(f"obs_diffs std: {np.std(obs_diffs)}")
        # stats for successes
        success_obs_diffs = obs_diffs[rewards_lst > 0]
        print(f"success_obs_diffs: {success_obs_diffs}")
        print(f"success_obs_diffs avg: {np.mean(success_obs_diffs)}")
        print(f"success_obs_diffs std: {np.std(success_obs_diffs)}")
        # stats for failures
        fail_obs_diffs = obs_diffs[rewards_lst <= 0]
        print(f"fail_obs_diffs: {fail_obs_diffs}")
        print(f"fail_obs_diffs avg: {np.mean(fail_obs_diffs)}")
        print(f"fail_obs_diffs std: {np.std(fail_obs_diffs)}")


def evaluate_dynamics(
    dynamics_checkpoint: Union[str, pathlib.Path],
    env_config: Optional[Union[str, pathlib.Path]] = None,
    debug_results: Optional[str] = None,
    path: Optional[Union[str, pathlib.Path]] = None,
    num_episodes: int = 100,
    seed: Optional[int] = 0,
    gui: Optional[bool] = None,
    verbose: bool = True,
    device: str = "auto",
) -> None:
    """Evaluates the policy either by loading an episode from `debug_results` or
    generating `num_eval_episodes` episodes.
    """
    if path is None and debug_results is None:
        raise ValueError("Either path or load_results must be specified")
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Load env.
    env_kwargs: Dict[str, Any] = {}
    if gui is not None:
        env_kwargs["gui"] = bool(gui)
    if env_config is None:
        # Try to load eval env.
        raise NotImplementedError("env_config must be specified")
    env = envs.load(env_config, **env_kwargs)
    # Load policy.
    dynamics = load_dynamics(checkpoint=dynamics_checkpoint, env=env, device=device)
    # DEBUG
    if isinstance(dynamics, TableEnvDynamics):
        dynamics._hand_crafted = True
        dynamics.plan_mode()

    if debug_results is not None:
        # Load reset seed.
        with open(debug_results, "rb") as f:
            seed = int(np.load(f, allow_pickle=True)["seed"])
        evaluate_episode(dynamics, env, seed=seed, verbose=verbose, debug=True, record=False)
    elif path is not None:
        # Evaluate episodes.
        evaluate_episodes(env, dynamics, num_episodes, pathlib.Path(path), verbose)


def main(args: argparse.Namespace) -> None:
    evaluate_dynamics(**vars(args))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", help="Env config")
    parser.add_argument("--dynamics-checkpoint", help="Dynamics checkpoint")
    parser.add_argument("--debug-results", type=str, help="Path to results_i.npz file.")
    parser.add_argument("--path", help="Path for output plots")
    parser.add_argument("--num-episodes", type=int, default=100, help="Number of episodes to evaluate")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--gui", type=int, help="Show pybullet gui")
    parser.add_argument("--verbose", type=int, default=1, help="Print debug messages")
    parser.add_argument("--device", default="auto", help="Torch device")
    args = parser.parse_args()

    main(args)
