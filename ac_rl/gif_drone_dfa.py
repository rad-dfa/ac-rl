"""Generates a gif of a trained drone policy (train_drone_policy.py) solving the DFA
hardcoded below, instead of one drawn from the training sampler.

Takes the same arguments as test_drone_policy.py (minus --generate-gif, since a gif is
always generated): rolls out --n episodes on the DFA, prints their stats, and animates
the one picked by --random (default) / --max / --min discounted return.

To try another task, edit DFA_NAME / START / ACCEPTING / TRANSITIONS. DroneEnv's
tokens are 0..4 (see DroneEnv.label_regions and the gif legend for their locations).
"""
import jax
import jax.numpy as jnp
from dfax import DFAx
from test_drone_policy import make_parser, setup, run_episodes, print_stats, save_gif

# ---------------------------------------------------------------------------
# Hardcoded DFA: visit token 1, then token 3, then token 0, never touching token 4.
# States not listed in TRANSITIONS self-loop on every token.
# ---------------------------------------------------------------------------
DFA_NAME = "1_then_3_then_0_avoid_4"
START = 0
ACCEPTING = {3}
TRANSITIONS = {  # (state, token) -> next state
    (0, 1): 1,
    (0, 3): 2,
    (0, 4): 2,
    (1, 0): 3,
    (1, 3): 2,
    (1, 4): 2,
}


def build_dfa(max_size, n_tokens):
    """Builds the hardcoded DFA as a DFAx padded to max_size states (the shape the
    encoder expects), minimized like the training samplers' outputs."""
    states = {START, *ACCEPTING, *(s for s, _ in TRANSITIONS), *TRANSITIONS.values()}
    assert max(states) < max_size, f"hardcoded DFA uses {max(states) + 1} states, but --max-size is {max_size}"
    assert all(0 <= a < n_tokens for _, a in TRANSITIONS), f"tokens must be in [0, {n_tokens})"

    transitions = jnp.tile(jnp.arange(max_size, dtype=jnp.int32)[:, None], (1, n_tokens))
    for (s, a), t in TRANSITIONS.items():
        transitions = transitions.at[s, a].set(t)
    labels = jnp.zeros((max_size,), dtype=bool).at[jnp.array(list(ACCEPTING))].set(True)

    return DFAx.create(start=START, transitions=transitions, labels=labels).minimize()


if __name__ == "__main__":

    parser = make_parser("Generate a gif of a DroneEnv policy on a hardcoded DFA", n_default=10, gif_flag=False)
    args = parser.parse_args()

    batch_size = args.n if args.batch_size == -1 else args.batch_size

    drone_env, env, network, params, run_tag, key = setup(args)
    dfa = build_dfa(args.max_size, drone_env.n_tokens)

    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, args.n)
    results = run_episodes(
        env, network, params, keys, args.deterministic, args.max_steps_in_episode, batch_size, dfa=dfa
    )
    print_stats(results)

    key, subkey = jax.random.split(key)
    save_gif(args, drone_env, results, subkey, f"drone_{run_tag}_dfa_{DFA_NAME}_n{args.n}")
