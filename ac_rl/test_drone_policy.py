import os
import jax
import argparse
import jax.numpy as jnp
from ppo import batchify
from train_drone_policy import ActorCritic
import flax.serialization as serialization
from rad_embeddings import Encoder, EncoderModule
from dfa_gym import DroneEnv, DFAWrapper, animate_drone_trace
from dfax.samplers import ReachSampler, ReachAvoidSampler, RADSampler

# Matches train_drone_policy.py's config["GAMMA"] (used by LogWrapper for disc_return).
GAMMA = 0.99


def make_parser(description, n_default=100, gif_flag=True):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the saved policy params (.msgpack)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed the policy was trained with; also selects the pretrained RAD encoder (default: 42)"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="storage",
        help="Directory for saving gifs, written to {save-dir}/gifs (default: storage)"
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="RAD",
        help="DFA sampler type: Reach (R), ReachAvoid (RA), ReachAvoidDerived (RAD) (default: RAD)"
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=5,
        help="Number of DFA states (default: 5)"
    )
    parser.add_argument(
        "--no-rad",
        action="store_true",
        help="Don't use pretrained RAD embeddings."
    )
    parser.add_argument(
        "--binary-reward",
        action="store_true",
        help="Use binary reward"
    )
    parser.add_argument("--safe", action="store_true", help="Policy was trained with --safe (PPO-Lagrangian)")
    parser.add_argument("--delta", type=float, default=0.05, help="--delta the --safe policy was trained with (default: 0.05)")
    parser.add_argument("--lambda-lr", type=float, default=0.05, help="--lambda-lr the --safe policy was trained with (default: 0.05)")
    parser.add_argument("--lambda-warmup", type=float, default=0, help="--lambda-warmup the --safe policy was trained with (default: 0)")
    parser.add_argument("--lambda-init", type=float, default=0, help="--lambda-init the --safe policy was trained with (default: 0)")
    parser.add_argument("--x-low", type=float, default=-1.0, help="Geofence lower x bound (default: -1.0)")
    parser.add_argument("--x-high", type=float, default=1.0, help="Geofence upper x bound (default: 1.0)")
    parser.add_argument("--y-low", type=float, default=-1.0, help="Geofence lower y bound (default: -1.0)")
    parser.add_argument("--y-high", type=float, default=1.0, help="Geofence upper y bound (default: 1.0)")
    parser.add_argument("--z-low", type=float, default=-1.0, help="Geofence lower z bound (default: -1.0)")
    parser.add_argument("--z-high", type=float, default=1.0, help="Geofence upper z bound (default: 1.0)")
    parser.add_argument("--max-speed", type=float, default=1.0, help="Max drone speed per axis (default: 1.0)")
    parser.add_argument("--dt", type=float, default=0.1, help="Simulation timestep (default: 0.1)")
    parser.add_argument(
        "--use-displacement-action",
        action="store_true",
        help="Actions are position deltas instead of velocity commands"
    )
    parser.add_argument(
        "--max-steps-in-episode",
        type=int,
        default=500,
        help="Episode horizon (default: 500)"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=n_default,
        help=f"Number of test episodes (default: {n_default})"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=-1,
        help="Number of episodes vmapped at once (default: n)"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Act with the policy mean instead of sampling"
    )
    if gif_flag:
        parser.add_argument(
            "--generate-gif",
            action="store_true",
            help="Generate a gif of one test episode"
        )
    select_group = parser.add_mutually_exclusive_group()
    select_group.add_argument("--random", action="store_true", help="Gif a uniformly random episode (default)")
    select_group.add_argument("--max", action="store_true", help="Gif the episode with max discounted return")
    select_group.add_argument("--min", action="store_true", help="Gif the episode with min discounted return")
    return parser


def setup(args):
    """Builds the env and network exactly as train_drone_policy.py does and loads the params."""
    drone_env = DroneEnv(
        n_agents=1,
        x_low=args.x_low,
        x_high=args.x_high,
        y_low=args.y_low,
        y_high=args.y_high,
        z_low=args.z_low,
        z_high=args.z_high,
        max_speed=args.max_speed,
        dt=args.dt,
        use_displacement_action=args.use_displacement_action,
        max_steps_in_episode=args.max_steps_in_episode,
    )

    if args.sampler in ["R", "Reach"]:
        sampler = ReachSampler(
            max_size=args.max_size,
            n_tokens=drone_env.n_tokens,
            p=None,
        )
        sampler_str = f"Reach_{args.max_size}_{drone_env.n_tokens}"
    elif args.sampler in ["ReachAvoid", "RA"]:
        sampler = ReachAvoidSampler(
            max_size=args.max_size,
            n_tokens=drone_env.n_tokens,
            p=None,
        )
        sampler_str = f"ReachAvoid_{args.max_size}_{drone_env.n_tokens}"
    elif args.sampler in ["ReachAvoidDerived", "RAD"]:
        sampler = RADSampler(
            max_size=args.max_size,
            n_tokens=drone_env.n_tokens,
            p=None,
        )
        sampler_str = f"RAD_{args.max_size}_{drone_env.n_tokens}"
    else:
        raise ValueError(f"Unknown sampler type: {args.sampler}")

    env = DFAWrapper(
        env=drone_env,
        gamma=None,
        sampler=sampler,
        binary_reward=args.binary_reward,
    )

    if args.no_rad:
        rad_str = "no_rad"
        encoder = EncoderModule(
            max_size=env.sampler.max_size
        )
    else:
        rad_str = "rad"
        # Must match train_drone_policy.py: the frozen encoder isn't stored in the checkpoint.
        encoder = Encoder(
            max_size=env.sampler.max_size,
            n_tokens=drone_env.n_tokens,
            seed=args.seed,
            binary_reward=args.binary_reward
        )

    reward_str = "binary" if args.binary_reward else "shaped"
    action_mode_str = "disp" if args.use_displacement_action else "vel"
    run_tag = (
        f"seed_{args.seed}_{sampler_str}_{rad_str}_{reward_str}"
        f"_x{args.x_low}_{args.x_high}_y{args.y_low}_{args.y_high}_z{args.z_low}_{args.z_high}"
        f"_speed{args.max_speed}_dt{args.dt}_{action_mode_str}_steps{args.max_steps_in_episode}"
    )
    if args.safe:
        run_tag += f"_safe_d{args.delta}_lr{args.lambda_lr}_w{int(args.lambda_warmup)}_l{args.lambda_init}"

    expected_name = f"policy_params_drone_{run_tag}.msgpack"
    if os.path.basename(args.model_path) != expected_name:
        print(f"WARNING: model file name does not match the given args (expected {expected_name})")

    network = ActorCritic(
        action_dim=env.action_space(env.agents[0]).shape[0],
        encoder=encoder,
        n_agents=env.num_agents,
        max_action=drone_env.max_action,
        deterministic=args.deterministic,
        safe=args.safe,
    )

    key = jax.random.PRNGKey(args.seed + 100)
    key, subkey = jax.random.split(key)
    init_x = env.observation_space(env.agents[0]).sample(subkey)
    key, subkey = jax.random.split(key)
    params = network.init(subkey, init_x)

    with open(args.model_path, "rb") as f:
        params = serialization.from_bytes(params, f.read())

    return drone_env, env, network, params, run_tag, key


def run_episodes(env, network, params, keys, deterministic, max_steps, batch_size, dfa=None):
    """Rolls out one episode per key (vmapped in chunks of batch_size).

    If `dfa` (a dfax.DFAx) is given, every episode is conditioned on it instead of a
    DFA drawn from env's sampler. Returns (successes, fails, ep_lens, ep_rewards,
    ep_disc_returns, trajs, init_dfas), with trajs the stacked DroneEnvStates
    (initial state included) of shape (n, max_steps + 1, ...).
    """

    def run_episode(params, key):
        key, subkey = jax.random.split(key)
        obs, state = env.reset(subkey)
        if dfa is not None:
            state = state.replace(
                dfas={agent: dfa for agent in env.agents},
                init_dfas={agent: dfa for agent in env.agents},
            )
            obs = env.get_obs(state=state)

        carry = {
            "key": key,
            "obs": obs,
            "state": state,
            "done": jnp.array(False),
            "success": jnp.array(False),
            "fail": jnp.array(False),
            "ep_len": jnp.array(0),
            "ep_reward": jnp.array(0.0),
            "ep_disc_return": jnp.array(0.0),
        }

        def step_fn(carry, _):
            key, act_key, step_key = jax.random.split(carry["key"], 3)
            out, _ = network.apply(params, batchify(carry["obs"], env.agents))
            actions = out if deterministic else out.sample(seed=act_key)
            actions = {agent: actions[i] for i, agent in enumerate(env.agents)}
            # step_env rather than step: step auto-resets on done, which would clobber the trace.
            obs, state, rewards, dones, _ = env.step_env(step_key, carry["state"], actions)

            done = dones["__all__"]
            reward = jnp.mean(jnp.array([rewards[a] for a in env.agents]))
            # Binary reward pays 0 (not -1) on rejection, so detect failure from the DFA
            # itself: it minimized to a rejecting sink (non-binary reward == -1).
            rejected = jnp.any(jnp.array([
                (state.dfas[a].n_states <= 1) & (state.dfas[a].reward(binary=False) < 0)
                for a in env.agents
            ]))
            new = {
                "key": key,
                "obs": obs,
                "state": state,
                "done": done,
                "success": done & (reward > 0),
                "fail": done & rejected,
                "ep_len": carry["ep_len"] + 1,
                "ep_reward": carry["ep_reward"] + reward,
                "ep_disc_return": carry["ep_disc_return"] + reward * GAMMA ** carry["ep_len"],
            }
            # Freeze the carry once the episode has ended.
            carry = jax.tree_util.tree_map(lambda old, nw: jnp.where(carry["done"], old, nw), carry, new)
            return carry, carry["state"].env_state

        final_carry, traj = jax.lax.scan(step_fn, carry, None, length=max_steps)
        traj = jax.tree_util.tree_map(
            lambda x0, xs: jnp.concatenate([x0[None], xs], axis=0), state.env_state, traj
        )
        return (
            final_carry["success"],
            final_carry["fail"],
            final_carry["ep_len"],
            final_carry["ep_reward"],
            final_carry["ep_disc_return"],
            traj,
            state.init_dfas[env.agents[0]],
        )

    run_batch = jax.jit(jax.vmap(run_episode, (None, 0)))

    n = keys.shape[0]
    results = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        results.append(run_batch(params, keys[start:end]))
    return jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *results)


def print_stats(results):
    successes, fails, ep_lens, ep_rewards, ep_disc_returns, _, _ = results
    timeouts = ~(successes | fails)
    print(f"Test completed for {successes.shape[0]} episodes.")
    print(f"Success rate: {jnp.mean(successes):.2f} +/- {jnp.std(successes):.2f}")
    print(f"Fail rate: {jnp.mean(fails):.2f} +/- {jnp.std(fails):.2f}")
    print(f"Timeout rate: {jnp.mean(timeouts):.2f} +/- {jnp.std(timeouts):.2f}")
    print(f"Average episode length: {jnp.mean(ep_lens):.2f} +/- {jnp.std(ep_lens):.2f}")
    print(f"Average episode reward: {jnp.mean(ep_rewards):.2f} +/- {jnp.std(ep_rewards):.2f}")
    print(f"Average episode discounted return: {jnp.mean(ep_disc_returns):.2f} +/- {jnp.std(ep_disc_returns):.2f}")


def save_gif(args, drone_env, results, key, gif_name):
    """Animates the episode picked by --random/--max/--min (default --random) to
    {save_dir}/gifs/{gif_name}_{select}[_det].gif."""
    successes, _, ep_lens, _, ep_disc_returns, trajs, init_dfas = results
    n = ep_lens.shape[0]

    if args.max:
        select_str = "max"
        idx = int(jnp.argmax(ep_disc_returns))
    elif args.min:
        select_str = "min"
        idx = int(jnp.argmin(ep_disc_returns))
    else:
        select_str = "random"
        idx = int(jax.random.randint(key, (), 0, n))

    ep_len = int(ep_lens[idx])
    trace = [
        jax.tree_util.tree_map(lambda x: x[idx, t], trajs)
        for t in range(ep_len + 1)
    ]
    dfa = jax.tree_util.tree_map(lambda x: x[idx], init_dfas)

    det_str = "_det" if args.deterministic else ""
    gif_dir = os.path.join(args.save_dir, "gifs")
    os.makedirs(gif_dir, exist_ok=True)
    gif_path = os.path.join(gif_dir, f"{gif_name}_{select_str}{det_str}.gif")

    animate_drone_trace(drone_env, trace, save_path=gif_path, dfa=dfa)
    print(
        f"Saved gif of episode {idx} ({select_str}; disc return {float(ep_disc_returns[idx]):.3f}, "
        f"success {bool(successes[idx])}, length {ep_len}) to {gif_path}"
    )


if __name__ == "__main__":

    parser = make_parser("Test DFA-conditioned DroneEnv policy")
    args = parser.parse_args()

    if (args.random or args.max or args.min) and not args.generate_gif:
        parser.error("--random/--max/--min can only be given together with --generate-gif")

    batch_size = args.n if args.batch_size == -1 else args.batch_size

    drone_env, env, network, params, run_tag, key = setup(args)

    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, args.n)
    results = run_episodes(
        env, network, params, keys, args.deterministic, args.max_steps_in_episode, batch_size
    )
    print_stats(results)

    if args.generate_gif:
        key, subkey = jax.random.split(key)
        save_gif(args, drone_env, results, subkey, f"drone_{run_tag}_n{args.n}")
