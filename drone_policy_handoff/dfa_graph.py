"""Build and step the goal-DFA graph tensors these policies expect -- in plain
numpy, no JAX/dfax/dfa-gym required.

This is a faithful port of two functions from the training codebase:
  - build_graph()    mirrors dfax.DFAx.to_graph()
  - advance_state()  mirrors dfax.DFAx.advance()
  - label_from_position() mirrors dfa_gym.DroneEnv.label_f()/label_regions()
See README.md for the full picture (why these exist, tensor layout, units).
"""
import numpy as np

MAX_SIZE = 5
N_TOKENS = 5


def build_graph(transitions, labels, start, max_size=MAX_SIZE, n_tokens=N_TOKENS):
    """Turn a small DFA (transitions/labels/start) into the graph tensors a
    model expects: node_features, edge_features, current_state, n_states.

    transitions : int array (max_size, n_tokens)
        transitions[s, t] is the state reached from state s on token t.
        Unused/padding states must self-loop on every token, e.g. row s
        should be [s, s, ..., s] -- build_graph relies on this to figure out
        which states are actually part of the automaton (see is_reach below).
    labels : bool array (max_size,)
        labels[s] is True iff s is an accepting state.
    start : int
        index of the initial state.
    """
    transitions = np.asarray(transitions, dtype=np.int64)
    labels = np.asarray(labels, dtype=bool)
    assert transitions.shape == (max_size, n_tokens)
    assert labels.shape == (max_size,)

    # is_reach[s]: is s reachable from `start` by following transitions?
    # (only reachable states get real node/edge features -- unreachable
    # padding states are left all-zero, exactly like dfax.DFAx.is_reach)
    is_reach = np.zeros(max_size, dtype=bool)
    is_reach[start] = True
    frontier = [int(start)]
    while frontier:
        s = frontier.pop()
        for t in range(n_tokens):
            nxt = int(transitions[s, t])
            if not is_reach[nxt]:
                is_reach[nxt] = True
                frontier.append(nxt)
    n_states = int(is_reach.sum())

    srcs, tgts = np.meshgrid(np.arange(max_size), np.arange(max_size), indexing="ij")
    srcs = srcs.flatten()
    tgts = tgts.flatten()

    state_idx = np.arange(max_size)
    is_init = is_reach & (state_idx == start)
    is_accept = is_reach & labels
    # a reachable, non-accepting state that self-loops on every token is a
    # permanent-fail sink
    is_reject = is_reach & np.all((transitions == state_idx[:, None]) & (~labels[:, None]), axis=1)
    is_non_terminal = is_reach & ~is_accept & ~is_reject

    node_features = np.stack([is_init, is_accept, is_reject, is_non_terminal], axis=1).astype(np.float32)

    edge_features_raw = (
        is_reach[:, None, None] & (transitions[:, None, :] == state_idx[None, :, None])
    ).astype(np.float32).reshape(-1, n_tokens)

    mask = np.any(edge_features_raw != 0, axis=-1)[:, None]
    edge_features = np.concatenate(
        [node_features[srcs] * mask, edge_features_raw, node_features[tgts] * mask], axis=-1
    ).astype(np.float32)

    return {
        "node_features": node_features,
        "edge_features": edge_features,
        "current_state": np.array([start], dtype=np.int64),
        "n_states": np.full(max_size, n_states, dtype=np.int64),
    }


def reach_avoid_chain(sequence, hazards=(), max_size=MAX_SIZE, n_tokens=N_TOKENS):
    """Convenience builder for the task family these policies were trained on
    (dfax.samplers.ReachAvoidSampler): visit `sequence` (a list of tokens) in
    order; from any not-yet-finished state, seeing a token in `hazards` jumps
    to a permanent fail state; every other token is ignored (stutter).

    Returns (transitions, labels, start) -- feed these into build_graph().
    `sequence` and `hazards` should be disjoint at each step, and
    2 + len(sequence) must be <= max_size (one state per sequence step, plus
    a success sink and a fail sink).
    """
    n = len(sequence) + 2
    assert n <= max_size, f"sequence of length {len(sequence)} needs {n} states > max_size={max_size}"
    success, fail = n - 2, n - 1

    transitions = np.tile(np.arange(max_size).reshape(-1, 1), (1, n_tokens))  # default: self-loop everywhere
    labels = np.zeros(max_size, dtype=bool)
    labels[success] = True
    transitions[success, :] = success
    transitions[fail, :] = fail

    for i, tok in enumerate(sequence):
        transitions[i, tok] = i + 1  # last step lands on `success`
        for h in hazards:
            transitions[i, h] = fail

    return transitions, labels, 0


def advance_state(edge_features, current_state, token, max_size=MAX_SIZE, n_tokens=N_TOKENS):
    """Step current_state given an observed token (0..n_tokens-1), or a
    negative token for "no label observed right now" (stutter). Mirrors
    dfax.DFAx.advance() exactly, reading the transition directly out of the
    edge_features tensor you're already feeding the model -- no separate
    transition table needed at runtime.
    """
    if token is None or token < 0 or token >= n_tokens:
        return int(current_state)
    base = int(current_state) * max_size
    token_col = 4 + token  # edge_features layout: [src_feat(4), token one-hot(n_tokens), tgt_feat(4)]
    for tgt in range(max_size):
        if edge_features[base + tgt, token_col] > 0.5:
            return tgt
    return int(current_state)  # no matching transition -- shouldn't happen for a well-formed DFA


def is_terminal(node_features, current_state):
    """(is_accept, is_reject) for the current state -- True once the goal is
    permanently satisfied or permanently failed; stop commanding the policy
    and land/hover once either is True."""
    is_accept = bool(node_features[int(current_state), 1] > 0.5)
    is_reject = bool(node_features[int(current_state), 2] > 0.5)
    return is_accept, is_reject


# ---------------------------------------------------------------------------
# Token labeling: mirrors dfa_gym.DroneEnv.label_regions()/label_f() exactly,
# for the geofence these policies were trained with (see README.md). If you
# change the geofence bounds, pass matching bounds here too.
# ---------------------------------------------------------------------------
def label_regions(x_low=-1.0, x_high=1.0, y_low=-1.0, y_high=1.0, z_low=-1.0, z_high=1.0):
    x_mid = 0.5 * (x_low + x_high)
    y_mid = 0.5 * (y_low + y_high)
    r = 0.1 * min(x_high - x_low, y_high - y_low)
    z_mid = 0.5 * (z_low + z_high)
    z_lo, z_hi = z_mid - 0.75, z_mid + 0.25

    x_lo_in, x_hi_in = x_low + r, x_high - r
    y_lo_in, y_hi_in = y_low + r, y_high - r
    corners = [(x_lo_in, y_lo_in), (x_lo_in, y_hi_in), (x_hi_in, y_lo_in), (x_hi_in, y_hi_in)]
    edge_mids = [(x_mid, y_lo_in), (x_mid, y_hi_in), (x_lo_in, y_mid), (x_hi_in, y_mid)]
    vert_edges = [
        (x_low, x_low + 2 * r, y_low + 2 * r, y_mid - r),
        (x_low, x_low + 2 * r, y_mid + r, y_high - 2 * r),
        (x_high - 2 * r, x_high, y_low + 2 * r, y_mid - r),
        (x_high - 2 * r, x_high, y_mid + r, y_high - 2 * r),
    ]
    horiz_edges = [
        (x_low + 2 * r, x_mid - r, y_low, y_low + 2 * r),
        (x_mid + r, x_high - 2 * r, y_low, y_low + 2 * r),
        (x_low + 2 * r, x_mid - r, y_high - 2 * r, y_high),
        (x_mid + r, x_high - 2 * r, y_high - 2 * r, y_high),
    ]

    regions = [(0, "circle", (x_mid, y_mid, r * 2, z_hi, z_hi + 0.25))]
    regions += [(1, "circle", (cx, cy, r, z_lo, z_hi)) for cx, cy in corners]
    regions += [(2, "circle", (mx, my, r, z_lo, z_hi)) for mx, my in edge_mids]
    regions += [(3, "rect", b + (z_lo, z_hi)) for b in vert_edges]
    regions += [(4, "rect", b + (z_lo, z_hi)) for b in horiz_edges]
    return regions


def label_from_position(pos, x_low=-1.0, x_high=1.0, y_low=-1.0, y_high=1.0, z_low=-1.0, z_high=1.0):
    """Map a (x, y, z) position (in the same normalized frame as `obs`) to a
    token 0..4, or -1 if the drone isn't in any labeled region. Regions can
    overlap -- an earlier entry in label_regions() wins."""
    x, y, z = pos
    label = -1
    for token, kind, params in reversed(label_regions(x_low, x_high, y_low, y_high, z_low, z_high)):
        if kind == "circle":
            cx, cy, r, z_lo, z_hi = params
            hit = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2 and z_lo <= z <= z_hi
        else:
            xl, xh, yl, yh, z_lo, z_hi = params
            hit = xl <= x <= xh and yl <= y <= yh and z_lo <= z <= z_hi
        if hit:
            label = token
    return label
