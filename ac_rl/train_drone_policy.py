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
from rad_embeddings.encoder import parse_p, canonical_sampler
from dfa_gym import DroneEnv, DFAWrapper
from flax.linen.initializers import constant, orthogonal
from naming import make_sampler, drone_run_name, checkpoint_path, log_path


class ActorCritic(nn.Module):
    action_dim: int
    encoder: Encoder
    n_agents: int
    max_action: float
    deterministic: bool = False
    safe: bool = False

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

        if self.safe:
            # Cost critic for PPO-Lagrangian, stacked as value[..., 1]. Built after
            # the actor so existing Dense_* param names (and checkpoints) are unchanged.
            cost_value = nn.Sequential([
                nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)),
                nn.tanh,
                nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)),
                nn.tanh,
                nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))
            ])(feat)
            value = jnp.concatenate([value, cost_value], axis=-1)
        else:
            value = jnp.squeeze(value, axis=-1)

        if self.deterministic:
            return actor_mean, value
        else:
            pi = distrax.MultivariateNormalDiag(actor_mean, actor_std)
            return pi, value


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
        help="Directory for the trained policy's checkpoint, with --log CSVs in {save-dir}/logs (default: storage)"
    )
    parser.add_argument(
        "--sampler",
        type=canonical_sampler,
        default="RAD",
        help="DFA sampler type: Reach (R), ReachAvoid (RA), ReachAvoidDerived (RAD), Mixture (MIX) (default: RAD)"
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=5,
        help="Number of DFA states (default: 5)"
    )
    parser.add_argument(
        "--p",
        type=parse_p,
        default=None,
        help="Sampler's DFA-size distribution: sizes n are drawn with weight p**n, or uniformly if None; "
             "also selects the pretrained RAD encoder (default: None)"
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
        help="Log to a csv file in {save-dir}/logs"
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
    parser.add_argument(
        "--safe",
        action="store_true",
        help="PPO-Lagrangian: maximize P(accept) subject to P(reject) <= --delta"
    )
    parser.add_argument("--delta", type=float, default=0.05, help="Allowed P(reject) for --safe (default: 0.05)")
    parser.add_argument("--lambda-lr", type=float, default=0.05, help="Lagrange multiplier step size per lambda update for --safe (default: 0.05)")
    parser.add_argument("--lambda-warmup", type=float, default=0, help="Env steps to hold lambda at --lambda-init for --safe (default: 0)")
    parser.add_argument("--lambda-init", type=float, default=0, help="Initial lambda for --safe; fixed if --lambda-lr 0 (default: 0)")
    parser.add_argument(
        "--lambda-every",
        type=int,
        default=1,
        help="Hold lambda for this many PPO updates, then update it from the P(reject) of the hold's "
             "second half, measured on the policy after it has responded to lambda (default: 1)"
    )
    parser.add_argument(
        "--lambda-clip",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=None,
        help="For --safe, keep lambda in [LOW, HIGH] (default: [0, inf))"
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
        "--use-velocity-filter",
        action="store_true",
        help="Velocity commands go through the deployed drone's velocity filter (DroneEnv defaults for its parameters)"
    )
    parser.add_argument(
        "--max-steps-in-episode",
        type=int,
        default=500,
        help="Episode horizon (default: 500)"
    )
    parser.add_argument("--gamma", type=float, default=0.999, help="PPO discount factor, for rewards and --safe costs (default: 0.999)")
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="Potential-based reward shaping with potential beta**(distance to acceptance), discounted with --gamma; "
             "shapes the reward only, not --safe's cost (default: no shaping)"
    )
    args = parser.parse_args()

    if args.safe and args.binary_reward:
        parser.error("--safe needs the +1/-1 DFA reward to tell rejections from timeouts; drop --binary-reward")
    if args.beta is not None and not 0 < args.beta < 1:
        parser.error("--beta needs 0 < BETA < 1")
    if args.lambda_clip:
        lam_low, lam_high = args.lambda_clip
        if not 0 <= lam_low <= lam_high:
            parser.error("--lambda-clip needs 0 <= LOW <= HIGH")
        # lambda sits at --lambda-init during warm-up (and always with --lambda-lr 0), unclipped.
        if not lam_low <= args.lambda_init <= lam_high:
            parser.error(f"--lambda-init {args.lambda_init} is outside --lambda-clip [{lam_low}, {lam_high}]")

    config = {
        "LR": 3e-4,
        "NUM_ENVS": 16,
        "NUM_STEPS": 512,
        "TOTAL_TIMESTEPS": 1e7,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 16,
        "GAMMA": args.gamma,
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
    if args.safe:
        config.update(SAFE=True, DELTA=args.delta, LAMBDA_LR=args.lambda_lr, LAMBDA_WARMUP=args.lambda_warmup,
                      LAMBDA_INIT=args.lambda_init, LAMBDA_EVERY=args.lambda_every)
        if args.lambda_clip:
            config["LAMBDA_CLIP"] = tuple(args.lambda_clip)

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
        use_velocity_filter=args.use_velocity_filter,
        max_steps_in_episode=args.max_steps_in_episode,
    )

    sampler = make_sampler(args.sampler, args.max_size, drone_env.n_tokens, args.p)

    env = DFAWrapper(
        env=drone_env,
        gamma=None if args.beta is None else args.gamma,
        beta=args.beta,
        sampler=sampler,
        binary_reward=args.binary_reward,
    )
    env = LogWrapper(env=env, config=config)

    if args.no_rad:
        encoder = EncoderModule(
            max_size=env.sampler.max_size
        )
    else:
        encoder = Encoder(
            max_size=env.sampler.max_size,
            n_tokens=drone_env.n_tokens,
            seed=args.seed,
            binary_reward=args.binary_reward,
            sampler=args.sampler,
            p=args.p,
        )

    # run_tag encodes every setting that changes training, so distinct runs never share
    # files; an identical rerun is refused instead of appending to its CSV / overwriting its checkpoint.
    run_tag = drone_run_name(dict(vars(args), n_tokens=drone_env.n_tokens))
    ckpt_path = checkpoint_path(args.save_dir, run_tag)
    config["LOG"] = log_path(args.save_dir, run_tag) if args.log else None
    if os.path.exists(ckpt_path):
        parser.error(f"already trained: {ckpt_path} (delete it or use another --save-dir)")
    if config["LOG"] and os.path.exists(config["LOG"]):
        # The checkpoint is only written once training finishes, so a log without one is from an interrupted run.
        parser.error(f"{config['LOG']} exists without a checkpoint, left over from an interrupted run "
                     "(delete it or use another --save-dir)")
    os.makedirs(f"{args.save_dir}/logs", exist_ok=True)

    if config["WANDB"]:
        wandb.init(
            entity="beyazit-y-berkeley-eecs",
            project="ac-rl-drone-policy",
            name=run_tag,
            config=config
        )

    network = ActorCritic(
        action_dim=env.action_space(env.agents[0]).shape[0],
        encoder=encoder,
        n_agents=env.num_agents,
        max_action=drone_env.max_action,
        safe=args.safe,
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

    # train_jit returns before training finishes (JAX dispatches asynchronously); serializing waits for it. Only then
    # create the checkpoint, so an interrupted run never leaves an empty one that counts as "already trained".
    ckpt_bytes = serialization.to_bytes(out["runner_state"][0].params)
    with open(ckpt_path, "wb") as f:
        f.write(ckpt_bytes)

    if config["WANDB"]:
        wandb.finish()
