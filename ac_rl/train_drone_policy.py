import os
import jax
import wandb
import distrax
import argparse
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
from ppo import make_train
from wrappers import LogWrapper
from dfax import batch2graph
import flax.serialization as serialization
from flax.traverse_util import flatten_dict
from rad_embeddings import Encoder, EncoderModule
from dfa_gym import DroneEnv, DFAWrapper
from flax.linen.initializers import constant, orthogonal
from dfax.samplers import ReachSampler, ReachAvoidSampler, RADSampler


class ActorCritic(nn.Module):
    action_dim: int
    encoder: Encoder
    n_agents: int
    max_action: float
    deterministic: bool = False

    @nn.compact
    def __call__(self, batch):

        obs_batch = batch["obs"]
        if obs_batch.ndim == 1: # (position + velocity,)
            obs_batch = obs_batch[None, ...] # -> (1, position + velocity)
        elif obs_batch.ndim != 2:
            raise ValueError(f"Expected (obs_dim,) or (B, obs_dim), got {obs_batch.shape} for obs")

        # DroneEnv's observation is already a small dense (position, velocity)
        # vector, not an image -- an MLP replaces train.py's CNN stem. tanh
        # (rather than relu) and 64-unit hidden layers follow the standard
        # MuJoCo-style continuous-control PPO recipe (PureJaxRL/CleanRL/SB3).
        obs_feat = nn.Sequential([
            nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)),
            nn.tanh,
            nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)),
            nn.tanh,
        ])(obs_batch)

        dfa_batch = batch["dfa"]
        dfa_graph = batch2graph(dfa_batch)
        dfa_feat = self.encoder(dfa_graph)

        feat = jnp.concatenate([obs_feat, dfa_feat], axis=-1)

        value = nn.Sequential([
            nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)),
            nn.tanh,
            nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)),
            nn.tanh,
            nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))
        ])(feat)

        actor_mean = nn.Sequential([
            nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)),
            nn.tanh,
            nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)),
            nn.tanh,
            nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))
        ])(feat)
        # Bound the mean to the env's actual action range instead of leaving it
        # unconstrained: DroneEnv silently clips whatever action it's given, so
        # an unbounded mean that drifts outside [-max_action, max_action] just
        # wastes samples on actions that get clipped before reaching the dynamics.
        actor_mean = self.max_action * nn.tanh(actor_mean)

        # State-independent log-std (standard continuous-control PPO practice),
        # initialized relative to max_action so initial exploration noise is on
        # the same scale as the action range, rather than a fixed unit-Gaussian
        # default that could be wildly too large or small depending on the env's
        # geofence/max-speed settings.
        actor_logstd = self.param(
            "actor_logstd",
            lambda key, shape: jnp.full(shape, np.log(0.5 * self.max_action), dtype=jnp.float32),
            (self.action_dim,)
        )
        actor_std = jnp.exp(actor_logstd)

        if self.deterministic:
            return actor_mean, jnp.squeeze(value, axis=-1)
        else:
            pi = distrax.MultivariateNormalDiag(actor_mean, actor_std)
            return pi, jnp.squeeze(value, axis=-1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train DFA-conditioned DroneEnv policy")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for PRNGKey (default: 42)"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="storage",
        help="Directory for saving the trained encoder (default: storage)"
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
        "--wandb",
        action="store_true",
        help="Log to wandb"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print logs"
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Log to a csv file in the given save directory"
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
    args = parser.parse_args()

    config = {
        "LR": 3e-4,
        "NUM_ENVS": 16,
        "NUM_STEPS": 512,
        "TOTAL_TIMESTEPS": 1e7,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 16,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        # Continuous-control PPO convention (e.g. CleanRL's ppo_continuous_action.py):
        # a Gaussian's differential entropy grows simply by inflating variance,
        # which fights the bounded/clipped action space here -- drop the entropy
        # bonus rather than let it push the policy toward a saturated std.
        "ENT_COEF": 0.0,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ANNEAL_LR": False,
    }

    config["DEBUG"] = args.debug
    config["WANDB"] = args.wandb

    if config["WANDB"]:
        wandb.init(
            entity="beyazit-y-berkeley-eecs",
            project="ac-rl-drone-policy",
            config=config
        )

    key = jax.random.PRNGKey(args.seed)

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
    env = LogWrapper(env=env, config=config)

    if args.no_rad:
        rad_str = "no_rad"
        encoder = EncoderModule(
            max_size=env.sampler.max_size
        )
    else:
        rad_str = "rad"
        encoder = Encoder(
            max_size=env.sampler.max_size,
            n_tokens=drone_env.n_tokens,
            seed=args.seed
        )

    config["LOG"] = f"{args.save_dir}/log_drone_seed_{args.seed}_{sampler_str}_{rad_str}.csv" if args.log else None

    network = ActorCritic(
        action_dim=env.action_space(env.agents[0]).shape[0],
        encoder=encoder,
        n_agents=env.num_agents,
        max_action=drone_env.max_action,
    )

    for i in config:
        print(f"{i:15}: {config[i]}")

    if config["DEBUG"]:
        key, subkey = jax.random.split(key)
        init_x = env.observation_space(env.agents[0]).sample(subkey)
        key, subkey = jax.random.split(key)
        params = network.init(subkey, init_x)
        flat = flatten_dict(params, sep="/")
        total = 0
        for k, v in flat.items():
            count = v.size
            total += count
            print(f"{k:60} {v.shape} {v.dtype} ({count:,} params)")
        print(f"\nTotal parameters: {total:,}")

    train_jit = jax.jit(make_train(config, env, network))
    out = train_jit(key)

    os.makedirs(args.save_dir, exist_ok=True)

    trained_params = out["runner_state"][0].params
    with open(f"{args.save_dir}/policy_params_drone_seed_{args.seed}_{sampler_str}_{rad_str}.msgpack", "wb") as f:
        f.write(serialization.to_bytes(trained_params))

    if config["WANDB"]:
        wandb.finish()
