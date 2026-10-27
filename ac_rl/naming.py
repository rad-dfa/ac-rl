"""Run names of the trained policies, built and parsed only here so that training, testing, plotting and exporting
always agree. A run name holds every setting that changes training, so distinct runs never share a checkpoint or log,
and the same settings always give the same name:

  drone_seed_42_RAD_5_10_p0.5_rad_shaped_x-1.0_1.0_y-1.0_1.0_z-1.0_1.0_speed1.0_dt0.1_vel_steps500_g0.999
       [_total20000000][_pbrs0.9][_safe_d0.05_lr0.05_w0_l1.0_k1_c0.0_1.0]     train_drone_policy.py
  token_seed_42_RAD_5_10_pNone_rad_shaped[_dynalpha]       train.py

That is the env, seed, sampler, max DFA size, number of tokens, p (None: uniformly sampled DFA sizes), the encoder
(pretrained RAD or trained from scratch) and the DFA reward (binary 1/0 or shaped +1/-1/0). Drone runs then add the
env and PPO settings, the total timesteps if not the default 1e7 (runs from before --total-timesteps keep their
names), PBRS's beta if the reward is shaped with PBRS, and PPO-Lagrangian's settings if --safe (no
--lambda-clip is written as its [0, inf) default). Samplers go by rad_embeddings' short names (R, RA, RAD, MIX), as
in its encoder checkpoints, whichever alias was given. Floats are written as repr(float), which round-trips exactly.

Checkpoints are {save_dir}/models/policy_params_{run}.msgpack and CSV logs {save_dir}/logs/log_{run}.csv.
"""
import os
import re
from rad_embeddings.encoder import SAMPLERS, canonical_sampler

# train_drone_policies.sh's settings shared by all of its runs (it varies seed, sampler, max_size, p, no_rad and
# use_velocity_filter), to find its runs by name, with its PBRS beta (it also runs without PBRS, beta None). Keep in
# sync with the script.
DRONE_SWEEP = dict(
    binary_reward=False, x_low=-1.0, x_high=1.0, y_low=-1.0, y_high=1.0, z_low=-1.0, z_high=1.0, max_speed=1.0,
    dt=0.1, use_displacement_action=False, max_steps_in_episode=500, gamma=0.999, total_timesteps=20_000_000, beta=0.7,
    safe=True, delta=0.05, lambda_lr=0.05, lambda_warmup=0, lambda_init=1.0, lambda_every=1, lambda_clip=(0.0, 1.0),
)

# train_drone_policy.py's --total-timesteps default, left out of run names.
DEFAULT_TOTAL_TIMESTEPS = 10_000_000


def parse_timesteps(s):
    """--total-timesteps' argparse type: a positive whole number of steps, also written like 2e7."""
    steps = float(s)
    if not (steps > 0 and steps == int(steps)):
        raise ValueError(s)
    return int(steps)


def make_sampler(name, max_size, n_tokens, p):
    return SAMPLERS[canonical_sampler(name)](max_size=max_size, n_tokens=n_tokens, p=p)


def num(x):
    return "None" if x is None else repr(float(x))


def identity(env, cfg):
    # The part of the run name before its settings, shared by both envs
    return (f"{env}_seed_{int(cfg['seed'])}_{canonical_sampler(cfg['sampler'])}_{int(cfg['max_size'])}"
            f"_{int(cfg['n_tokens'])}_p{num(cfg['p'])}_{'no_rad' if cfg['no_rad'] else 'rad'}")


def drone_run_settings(cfg):
    """The part of a drone run name after the encoder, i.e. everything but the seed, sampler, sizes, p and encoder."""
    action = "disp" if cfg["use_displacement_action"] else "vfilt" if cfg["use_velocity_filter"] else "vel"
    settings = (
        f"{'binary' if cfg['binary_reward'] else 'shaped'}"
        f"_x{num(cfg['x_low'])}_{num(cfg['x_high'])}_y{num(cfg['y_low'])}_{num(cfg['y_high'])}"
        f"_z{num(cfg['z_low'])}_{num(cfg['z_high'])}_speed{num(cfg['max_speed'])}_dt{num(cfg['dt'])}"
        f"_{action}_steps{int(cfg['max_steps_in_episode'])}_g{num(cfg['gamma'])}"
    )
    if int(cfg["total_timesteps"]) != DEFAULT_TOTAL_TIMESTEPS:
        settings += f"_total{int(cfg['total_timesteps'])}"
    if cfg["beta"] is not None:
        settings += f"_pbrs{num(cfg['beta'])}"
    if cfg["safe"]:
        low, high = cfg["lambda_clip"] or (0.0, float("inf"))
        # Timesteps are integers, so t > lambda_warmup exactly when t > int(lambda_warmup)
        settings += (f"_safe_d{num(cfg['delta'])}_lr{num(cfg['lambda_lr'])}_w{int(cfg['lambda_warmup'])}"
                     f"_l{num(cfg['lambda_init'])}_k{int(cfg['lambda_every'])}_c{num(low)}_{num(high)}")
    return settings


def drone_run_name(cfg):
    """cfg: train_drone_policy.py's arguments by their dest names, plus n_tokens."""
    return f"{identity('drone', cfg)}_{drone_run_settings(cfg)}"


def token_run_name(cfg):
    """cfg: train.py's arguments by their dest names, plus p."""
    reward = "binary" if cfg["binary_reward"] else "shaped"
    return f"{identity('token', cfg)}_{reward}" + ("_dynalpha" if cfg["dynamic_alphabet"] else "")


def checkpoint_path(save_dir, run):
    return os.path.join(save_dir, "models", f"policy_params_{run}.msgpack")


def log_path(save_dir, run):
    return os.path.join(save_dir, "logs", f"log_{run}.csv")


_NUM = r"-?(?:inf|\d+(?:\.\d+)?(?:e[-+]\d+)?)"


def _num(name):
    return rf"(?P<{name}>{_NUM})"


_DRONE_RE = re.compile(
    rf"drone_seed_(?P<seed>\d+)_(?P<sampler>{'|'.join(SAMPLERS)})_(?P<max_size>\d+)_(?P<n_tokens>\d+)"
    rf"_p(?P<p>None|{_NUM})_(?P<encoder>rad|no_rad)_(?P<reward>binary|shaped)"
    rf"_x{_num('x_low')}_{_num('x_high')}_y{_num('y_low')}_{_num('y_high')}_z{_num('z_low')}_{_num('z_high')}"
    rf"_speed{_num('max_speed')}_dt{_num('dt')}_(?P<action>vel|vfilt|disp)_steps(?P<max_steps_in_episode>\d+)"
    rf"_g{_num('gamma')}(?:_total(?P<total_timesteps>\d+))?(?:_pbrs{_num('beta')})?"
    rf"(?P<safe>_safe_d{_num('delta')}_lr{_num('lambda_lr')}_w(?P<lambda_warmup>\d+)_l{_num('lambda_init')}"
    rf"_k(?P<lambda_every>\d+)_c{_num('clip_low')}_{_num('clip_high')})?"
)


def parse_drone_run_name(run):
    """The cfg that drone_run_name turns into `run`, or None if `run` isn't a drone run name."""
    m = _DRONE_RE.fullmatch(run)
    if m is None:
        return None
    g = m.groupdict()
    floats = ["x_low", "x_high", "y_low", "y_high", "z_low", "z_high", "max_speed", "dt", "gamma"]
    cfg = {k: int(g[k]) for k in ("seed", "max_size", "n_tokens", "max_steps_in_episode")}
    cfg.update({k: float(g[k]) for k in floats})
    cfg.update(
        sampler=g["sampler"],
        p=None if g["p"] == "None" else float(g["p"]),
        no_rad=g["encoder"] == "no_rad",
        binary_reward=g["reward"] == "binary",
        use_displacement_action=g["action"] == "disp",
        use_velocity_filter=g["action"] == "vfilt",
        total_timesteps=DEFAULT_TOTAL_TIMESTEPS if g["total_timesteps"] is None else int(g["total_timesteps"]),
        beta=None if g["beta"] is None else float(g["beta"]),
        safe=g["safe"] is not None,
    )
    if cfg["safe"]:
        cfg.update({k: float(g[k]) for k in ("delta", "lambda_lr", "lambda_init")})
        cfg.update(lambda_warmup=int(g["lambda_warmup"]), lambda_every=int(g["lambda_every"]),
                   lambda_clip=(float(g["clip_low"]), float(g["clip_high"])))
    return cfg


def _parse_file_name(prefix, ext, path):
    m = re.fullmatch(rf"{prefix}_(.+){re.escape(ext)}", os.path.basename(path))
    return None if m is None else parse_drone_run_name(m[1])


def parse_checkpoint_name(path):
    """The cfg of a drone checkpoint's run, or None if its file name isn't one."""
    return _parse_file_name("policy_params", ".msgpack", path)


def parse_log_name(path):
    """The cfg of a drone CSV log's run, or None if its file name isn't one."""
    return _parse_file_name("log", ".csv", path)
