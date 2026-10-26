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
from dfax import list2batch, batch2graph
from dfa_gym import TokenEnv, DFAWrapper
import flax.serialization as serialization
from flax.traverse_util import flatten_dict
from rad_embeddings import Encoder, EncoderModule
from rad_embeddings.encoder import canonical_sampler
from flax.linen.initializers import constant, orthogonal
from naming import make_sampler, token_run_name, checkpoint_path, log_path

class ActorCritic(nn.Module):
    action_dim: int
    encoder: Encoder
    n_agents: int
    n_tokens: int
    deterministic: bool = False
    padding: str = "VALID"
    dynamic_alphabet: bool = False

    @nn.compact
    def __call__(self, batch):

        obs_batch = batch["obs"]
        if obs_batch.ndim == 3: # (C, H, W)
            obs_batch = obs_batch[None, ...] # -> (1, C, H, W)
        elif obs_batch.ndim != 4:
            raise ValueError(f"Expected (C, H, W) or (B, C, H, W), got {obs_batch.shape} for obs")
        obs_batch = jnp.transpose(obs_batch, (0, 2, 3, 1)) # -> (B, H, W, C)

        obs_feat = nn.Sequential([
            nn.Conv(16, (2, 2), padding=self.padding, kernel_init=orthogonal(np.sqrt(2))),
            nn.relu,
            nn.Conv(32, (2, 2), padding=self.padding, kernel_init=orthogonal(np.sqrt(2))),
            nn.relu,
            nn.Conv(64, (2, 2), padding=self.padding, kernel_init=orthogonal(np.sqrt(2))),
            nn.relu,
            lambda x: x.reshape((x.shape[0], -1)),
        ])(obs_batch)

        dfa_batch = batch["dfa"]
        dfa_graph = batch2graph(dfa_batch)
        dfa_feat = self.encoder(dfa_graph)

        if self.dynamic_alphabet:
            tkn_batch = batch["tkn"]
            if tkn_batch.ndim == 1: # (n_symbols,)
                tkn_batch = tkn_batch[None, ...] # -> (1, n_symbols,)
            elif tkn_batch.ndim != 2:
                raise ValueError(f"Expected (n_symbols,) or (B, n_symbols), got {tkn_batch.shape} for obs")

            tkn_feat = nn.Embed(self.n_tokens, 8)(tkn_batch).reshape(tkn_batch.shape[0], -1)

            tsk_feat = nn.Sequential([
                nn.Dense(1024, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)),
                nn.relu,
                nn.Dense(1024, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)),
                nn.relu,
                nn.Dense(32, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
            ])(jnp.concatenate([dfa_feat, tkn_feat], axis=-1))

            feat = jnp.concatenate([obs_feat, tsk_feat], axis=-1)
        else:
            feat = jnp.concatenate([obs_feat, dfa_feat], axis=-1)

        value = nn.Sequential([
            nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)),
            nn.relu,
            nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)),
            nn.relu,
            nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))
        ])(feat)

        logits = nn.Sequential([
            nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)),
            nn.relu,
            nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)),
            nn.relu,
            nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)),
            nn.relu,
            nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))
        ])(feat)

        if self.deterministic:
            action = jnp.argmax(logits, axis=-1)
            return action, jnp.squeeze(value, axis=-1)
        else:
            pi = distrax.Categorical(logits=logits)
            return pi, jnp.squeeze(value, axis=-1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train DFA-conditioned TokenEnv policy")
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
        help="Directory for the trained policy's checkpoint in {save-dir}/models, with --log CSVs in {save-dir}/logs (default: storage)"
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
        "--n-tokens",
        type=int,
        default=10,
        help="Number tokens (default: 10)"
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
        help="Don't use pretrained RAD embeddings"
    )
    parser.add_argument(
        "--binary-reward",
        action="store_true",
        help="Use binary reward"
    )
    parser.add_argument(
        "--dynamic-alphabet",
        action="store_true",
        help="Use dynamic alphabet"
    )
    args = parser.parse_args()

    if args.dynamic_alphabet:
        config = {
            "LR": 3e-4,
            "NUM_ENVS": 64,
            "NUM_STEPS": 1024,
            "TOTAL_TIMESTEPS": 1e7,
            "UPDATE_EPOCHS": 8,
            "NUM_MINIBATCHES": 8,
            "GAMMA": 0.99,
            "GAE_LAMBDA": 0.95,
            "CLIP_EPS": 0.2,
            "ENT_COEF": 0.01,
            "VF_COEF": 0.5,
            "MAX_GRAD_NORM": 0.5,
            "ANNEAL_LR": False,
        }
    else:
        config = {
            "LR": 3e-4,
            "NUM_ENVS": 16,
            "NUM_STEPS": 128,
            "TOTAL_TIMESTEPS": 1e6,
            "UPDATE_EPOCHS": 8,
            "NUM_MINIBATCHES": 8,
            "GAMMA": 0.99,
            "GAE_LAMBDA": 0.95,
            "CLIP_EPS": 0.2,
            "ENT_COEF": 0.01,
            "VF_COEF": 0.5,
            "MAX_GRAD_NORM": 0.5,
            "ANNEAL_LR": False,
        }


    config["DEBUG"] = args.debug
    config["WANDB"] = args.wandb

    key = jax.random.PRNGKey(args.seed)

    # DFA sizes are sampled uniformly (p = None)
    sampler = make_sampler(args.sampler, args.max_size, args.n_tokens, None)

    token_env = TokenEnv(
        n_agents=1,
        max_steps_in_episode=100,
        fixed_map_seed=args.seed
    )

    env = DFAWrapper(
        env=token_env,
        gamma=None,
        sampler=sampler,
        binary_reward=args.binary_reward,
        dynamic_alphabet=args.dynamic_alphabet
    )
    env = LogWrapper(env=env, config=config)

    if args.no_rad:
        encoder = EncoderModule(
            max_size=env.sampler.max_size
        )
    else:
        # The encoder pretrained on the same task distribution and reward, as in train_drone_policy.py
        encoder = Encoder(
            max_size=env.sampler.max_size,
            n_tokens=token_env.n_tokens,
            seed=args.seed,
            binary_reward=args.binary_reward,
            sampler=args.sampler,
            p=sampler.p,
        )

    # As in train_drone_policy.py: the run name holds every setting that changes training, and an identical rerun is
    # refused instead of appending to its CSV / overwriting its checkpoint.
    run_name = token_run_name(dict(vars(args), p=sampler.p))
    ckpt_path = checkpoint_path(args.save_dir, run_name)
    config["LOG"] = log_path(args.save_dir, run_name) if args.log else None
    if os.path.exists(ckpt_path):
        parser.error(f"already trained: {ckpt_path} (delete it or use another --save-dir)")
    if config["LOG"] and os.path.exists(config["LOG"]):
        # The checkpoint is only written once training finishes, so a log without one is from an interrupted run.
        parser.error(f"{config['LOG']} exists without a checkpoint, left over from an interrupted run "
                     "(delete it or use another --save-dir)")
    os.makedirs(f"{args.save_dir}/models", exist_ok=True)
    os.makedirs(f"{args.save_dir}/logs", exist_ok=True)

    if config["WANDB"]:
        wandb.init(
            entity="beyazit-y-berkeley-eecs",
            project="rad-rl-jax",
            name=run_name,
            config=config
        )

    network = ActorCritic(
        action_dim=env.action_space(env.agents[0]).n,
        encoder=encoder,
        n_agents=env.num_agents,
        n_tokens=env.n_tkns,
        dynamic_alphabet=args.dynamic_alphabet
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

    # As in train_drone_policy.py: serializing waits for training to finish, and only then is the checkpoint created,
    # so an interrupted run never leaves an empty one that counts as "already trained".
    ckpt_bytes = serialization.to_bytes(out["runner_state"][0].params)
    with open(ckpt_path, "wb") as f:
        f.write(ckpt_bytes)

    if config["WANDB"]:
        wandb.finish()

