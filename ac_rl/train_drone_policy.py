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
from rad_embeddings.encoder import parse_p
from dfa_gym import DroneEnv, DFAWrapper
from flax.linen.initializers import constant, orthogonal
from dfax.samplers import ReachSampler, ReachAvoidSampler, RADSampler


# Bounds and initial value of the policy's std, before tanh squashing (see SquashedGaussian): action
# noise is about std * max_action mid-range and shrinks toward the action bounds. The floor sits below
# the 0.08-0.15 that successful drone policies converged to, so it never keeps them from getting precise.
STD_MIN, STD_MAX, STD_INIT = 0.05, 1.0, 0.5
# How far inside +-1 log_prob keeps tanh values, since atanh(+-1) is infinite.
SQUASH_EPS = 1e-6


class SquashedGaussian:
    """Diagonal Gaussian over u, acting with max_action * tanh(u): actions always lie inside the bounds,
    so the env never clips them and no two samples collapse onto the same clipped action.

    Implements the parts of a distrax distribution that ppo.py uses. log_prob scores an action by the
    Gaussian density of u = atanh(action / max_action) without tanh's log-Jacobian: that term depends
    only on the action, so it cancels in PPO's probability ratio, the only place log_prob is used.
    entropy is the Gaussian's before squashing (the squashed one has no closed form); with ENT_COEF 0
    it only appears in the logs.
    """

    def __init__(self, loc, scale, max_action):
        self.base = distrax.MultivariateNormalDiag(loc, scale)
        self.max_action = max_action

    def sample(self, seed):
        return self.max_action * jnp.tanh(self.base.sample(seed=seed))

    def log_prob(self, action):
        # float32 tanh rounds far-out samples to exactly +-1; clamping gives them a finite u, the same
        # one for the old and new policy, so PPO's ratio stays exact.
        u = jnp.arctanh(jnp.clip(action / self.max_action, -1 + SQUASH_EPS, 1 - SQUASH_EPS))
        return self.base.log_prob(u)

    def entropy(self):
        return self.base.entropy()


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

        actor_feat = nn.Sequential([
            nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)),
            nn.tanh,
            nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)),
            nn.tanh,
        ])(feat)
        # Mean of the Gaussian before squashing: unbounded, since SquashedGaussian's tanh keeps the actions
        # themselves in [-max_action, max_action], and a tanh here would saturate it at full speed.
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_feat)

        # State-dependent std from the same features as the mean, bounded in log space: sigmoid(z) slides
        # log(std) from log(STD_MIN) to log(STD_MAX), so std = STD_MIN^(1-s) * STD_MAX^s
        # and each unit of z scales it by a roughly constant factor across a 20x range.
        # The bias makes it start at STD_INIT in every state, since the small kernel starts z near the bias.
        # Built after the mean's output layer so Dense_0..Dense_7 keep their names (export_onnx.py reads them).
        log_min, log_max = np.log(STD_MIN), np.log(STD_MAX)
        s_init = (np.log(STD_INIT) - log_min) / (log_max - log_min)
        z = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(np.log(s_init / (1 - s_init)))
        )(actor_feat)
        actor_std = jnp.exp(log_min + (log_max - log_min) * nn.sigmoid(z))

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
            # The squashed mean, the same max_action * tanh(Dense_7) that export_onnx.py builds.
            return self.max_action * jnp.tanh(actor_mean), value
        else:
            pi = SquashedGaussian(actor_mean, actor_std, self.max_action)
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
    parser.add_argument("--gamma", type=float, default=0.99, help="PPO discount factor, for rewards and --safe costs (default: 0.99)")
    args = parser.parse_args()

    if args.safe and args.binary_reward:
        parser.error("--safe needs the +1/-1 DFA reward to tell rejections from timeouts; drop --binary-reward")

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
        # No entropy bonus: ActorCritic's std has a floor (STD_MIN), so exploration can't collapse, and a
        # Gaussian's entropy grows simply by inflating variance, so a bonus would only push the std
        # toward its ceiling (STD_MAX). The logged entropy is the pre-squash Gaussian's (SquashedGaussian).
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
            p=args.p,
        )
        sampler_str = f"Reach_{args.max_size}_{drone_env.n_tokens}"
    elif args.sampler in ["ReachAvoid", "RA"]:
        sampler = ReachAvoidSampler(
            max_size=args.max_size,
            n_tokens=drone_env.n_tokens,
            p=args.p,
        )
        sampler_str = f"ReachAvoid_{args.max_size}_{drone_env.n_tokens}"
    elif args.sampler in ["ReachAvoidDerived", "RAD"]:
        sampler = RADSampler(
            max_size=args.max_size,
            n_tokens=drone_env.n_tokens,
            p=args.p,
        )
        sampler_str = f"RAD_{args.max_size}_{drone_env.n_tokens}"
    else:
        raise ValueError(f"Unknown sampler type: {args.sampler}")
    sampler_str += f"_p{args.p}" if args.p is not None else ""  # p = None keeps earlier names

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
            seed=args.seed,
            binary_reward=args.binary_reward,
            sampler=args.sampler,
            p=args.p,
        )

    reward_str = "binary" if args.binary_reward else "shaped"
    action_mode_str = "disp" if args.use_displacement_action else "vel"
    run_tag = (
        f"seed_{args.seed}_{sampler_str}_{rad_str}_{reward_str}"
        f"_x{args.x_low}_{args.x_high}_y{args.y_low}_{args.y_high}_z{args.z_low}_{args.z_high}"
        f"_speed{args.max_speed}_dt{args.dt}_{action_mode_str}_steps{args.max_steps_in_episode}"
    )
    # State-dependent std and tanh-squashed actions: keeps these checkpoints apart from the earlier
    # learned-log-std ones, a different policy that would otherwise share names (and count as "already trained").
    run_tag += "_sdstd"
    run_tag += f"_g{args.gamma}" if args.gamma != 0.99 else ""  # gamma = 0.99 keeps earlier names
    if args.safe:
        run_tag += f"_safe_d{args.delta}_lr{args.lambda_lr}_w{int(args.lambda_warmup)}_l{args.lambda_init}"
        run_tag += f"_k{args.lambda_every}" if args.lambda_every > 1 else ""  # k = 1 keeps earlier names

    # run_tag encodes every setting that changes training, so distinct runs never share
    # files; an identical rerun is refused instead of appending to its CSV / overwriting its checkpoint.
    ckpt_path = f"{args.save_dir}/policy_params_drone_{run_tag}.msgpack"
    config["LOG"] = f"{args.save_dir}/log_drone_{run_tag}.csv" if args.log else None
    if os.path.exists(ckpt_path):
        parser.error(f"already trained: {ckpt_path} (delete it or use another --save-dir)")
    if config["LOG"] and os.path.exists(config["LOG"]):
        # The checkpoint is only written once training finishes, so a log without one is from an interrupted run.
        parser.error(f"{config['LOG']} exists without a checkpoint, left over from an interrupted run "
                     "(delete it or use another --save-dir)")
    os.makedirs(args.save_dir, exist_ok=True)

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

    trained_params = out["runner_state"][0].params
    with open(ckpt_path, "wb") as f:
        f.write(serialization.to_bytes(trained_params))

    if config["WANDB"]:
        wandb.finish()
