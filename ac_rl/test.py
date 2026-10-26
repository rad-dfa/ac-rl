import os
import jax
import yaml
import argparse
import itertools
import jax.numpy as jnp
from functools import partial
from rad_embeddings import Encoder, EncoderModule
from flax.traverse_util import flatten_dict
from dfa_gym import TokenEnv, DFAWrapper
import flax.serialization as serialization
from rad_embeddings.encoder import canonical_sampler
from naming import make_sampler, token_run_name, checkpoint_path
from train import ActorCritic
from ppo import batchify
from collections import Counter
from dfax import data2sampler


if __name__ == "__main__":

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(description="Train TokenEnv policy")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        required=True,
        help="List of seeds for testing (must match trained checkpoints)"
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=5,
        help="Number of DFA states (default: 5)"
    )
    parser.add_argument(
        "--n-tokens",
        type=int,
        default=10,
        help="Number tokens (default: 10)"
    )
    parser.add_argument(
        "--sampler",
        type=canonical_sampler,
        default="RAD",
        help="DFA sampler type: Reach (R), ReachAvoid (RA), ReachAvoidDerived (RAD), Mixture (MIX) (default: RAD)"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="storage",
        help="--save-dir of the runs; checkpoints are read from {model-dir}/models (default: storage)"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of samples (default: 10)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=-1,
        help="Batch size (default: n)"
    )
    parser.add_argument(
        "--no-rad",
        action="store_true",
        help="Don't use pretrained RAD embeddings"
    )
    parser.add_argument(
        "--binary-reward",
        action="store_true",
        help="Policies were trained with --binary-reward"
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Output results on a comma seperated line"
    )
    args = parser.parse_args()

    batch_size = args.batch_size
    if batch_size == -1:
        batch_size = args.n

    # As in train.py, DFA sizes are sampled uniformly (p = None)
    sampler = make_sampler(args.sampler, args.max_size, args.n_tokens, None)

    n_seeds = len(args.seeds)
    success_rate_list = jnp.zeros((n_seeds,))
    avg_len_list = jnp.zeros((n_seeds,))
    avg_reward_list = jnp.zeros((n_seeds,))
    avg_disc_return_list = jnp.zeros((n_seeds,))

    for i, seed in enumerate(args.seeds):

        key = jax.random.PRNGKey(seed + 100)

        token_env = TokenEnv(
            n_agents=1,
            max_steps_in_episode=100,
            fixed_map_seed=args.seeds[i]
        )

        env = DFAWrapper(
            env=token_env,
            gamma=None,
            sampler=sampler,
            binary_reward=args.binary_reward
        )

        if args.no_rad:
            encoder = EncoderModule(
                max_size=env.sampler.max_size
            )
        else:
            # Must match train.py: the frozen encoder isn't stored in the checkpoint.
            encoder = Encoder(
                max_size=env.sampler.max_size,
                n_tokens=token_env.n_tokens,
                seed=args.seeds[i],
                binary_reward=args.binary_reward,
                sampler=args.sampler,
                p=sampler.p,
            )

        network = ActorCritic(
            action_dim=env.action_space(env.agents[0]).n,
            encoder=encoder,
            n_agents=env.num_agents,
            n_tokens=env.n_tkns
        )

        key, subkey = jax.random.split(key)
        init_x = env.observation_space(env.agents[0]).sample(subkey)
        key, subkey = jax.random.split(key)
        params = network.init(subkey, init_x)

        run_name = token_run_name(dict(vars(args), seed=seed, p=sampler.p, dynamic_alphabet=False))
        with open(checkpoint_path(args.model_dir, run_name), "rb") as f:
            params = serialization.from_bytes(params, f.read())

        @partial(jax.jit, static_argnums=(0, 1))
        def run_episode(env, network, params, key):
            key, subkey = jax.random.split(key)
            obs, state = env.reset(subkey)

            carry = {
                "key": key,
                "obs": obs,
                "state": state,
                "done": False,
                "success": 0,
                "ep_len": 0,
                "ep_reward": 0.0,
                "ep_discounted_return": 0.0,
                "discount": 1.0,
            }

            def cond_fn(carry):
                return ~carry["done"]

            def body_fn(carry):
                key, subkey = jax.random.split(carry["key"])
                pi, value = network.apply(params, batchify(carry["obs"], env.agents))
                actions = pi.sample(seed=subkey)
                actions = {agent: actions[i] for i, agent in enumerate(env.agents)}
                key, subkey = jax.random.split(key)
                obs, state, rewards, dones, infos = env.step(subkey, carry["state"], actions)

                done = dones["__all__"]
                rewards_arr = jnp.array([rewards[a] for a in env.agents])
                success = carry["success"] + jnp.all(rewards_arr > 0) * done
                ep_reward = carry["ep_reward"] + jnp.mean(rewards_arr)
                ep_discounted_return = carry["ep_discounted_return"] + carry["discount"] * jnp.mean(rewards_arr)
                discount = carry["discount"] * (env.gamma if env.gamma is not None else 1.0)
                ep_len = carry["ep_len"] + 1

                return {
                    "key": key,
                    "obs": obs,
                    "state": state,
                    "done": done,
                    "success": success,
                    "ep_len": ep_len,
                    "ep_reward": ep_reward,
                    "ep_discounted_return": ep_discounted_return,
                    "discount": discount,
                }

            final_carry = jax.lax.while_loop(cond_fn, body_fn, carry)
            return (
                final_carry["success"],
                final_carry["ep_len"],
                final_carry["ep_reward"],
                final_carry["ep_discounted_return"]
            )

        @partial(jax.jit, static_argnums=(0, 1))
        def run_episodes(env, network, params, keys):
            return jax.vmap(run_episode, (None, None, None, 0))(env, network, params, keys)

        def batched_vmap_run(env, network, params, keys, batch_size):
            n = keys.shape[0]
            results = []
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch_keys = keys[start:end]
                batch_results = run_episodes(env, network, params, batch_keys)
                results.append(batch_results)
            results = jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *results)
            return results

        keys = jax.random.split(key, args.n)
        results = batched_vmap_run(env, network, params, keys, batch_size=batch_size)

        success_counts, ep_lens, ep_rewards, ep_disc_returns = results

        success_rate = jnp.mean(success_counts)
        avg_len = jnp.mean(ep_lens)
        avg_reward = jnp.mean(ep_rewards)
        avg_disc_return = jnp.mean(ep_disc_returns)

        success_rate_list = success_rate_list.at[i].set(success_rate)
        avg_len_list = avg_len_list.at[i].set(avg_len)
        avg_reward_list = avg_reward_list.at[i].set(avg_reward)
        avg_disc_return_list = avg_disc_return_list.at[i].set(avg_disc_return)

    success_rate_mean = jnp.mean(success_rate_list)
    success_rate_std = jnp.std(success_rate_list)

    avg_len_mean = jnp.mean(avg_len_list)
    avg_len_std = jnp.std(avg_len_list)

    avg_reward_mean = jnp.mean(avg_reward_list)
    avg_reward_std = jnp.std(avg_reward_list)

    avg_disc_return_mean = jnp.mean(avg_disc_return_list)
    avg_disc_return_std = jnp.std(avg_disc_return_list)

    if args.csv:
        print(f"{'no_rad' if args.no_rad else 'rad'}, {args.sampler}, {success_rate_mean} +/- {success_rate_std}, {avg_len_mean} +/- {avg_len_std}, {avg_reward_mean} +/- {avg_reward_std}, {avg_disc_return_mean} +/- {avg_disc_return_std}")
    else:
        print(f"Test completed for {n_seeds} seeds.")
        print(f"Success rate: {success_rate_mean:.2f} +/- {success_rate_std:.2f}")
        print(f"Average episode length: {avg_len_mean:.2f} +/- {avg_len_std:.2f}")
        print(f"Average episode reward: {avg_reward_mean:.2f} +/- {avg_reward_std:.2f}")
        print(f"Average episode discounted return: {avg_disc_return_mean:.2f} +/- {avg_disc_return_std:.2f}")

