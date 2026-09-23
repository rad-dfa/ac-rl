"""Convert trained drone-policy ActorCritic checkpoints (train_drone_policy.py)
to standalone ONNX models, without needing TensorFlow/jax2tf.

Why hand-built instead of jax2tf/jax2onnx: on Intel macOS, TensorFlow>=2.17 (the
first version whose ml-dtypes pin is compatible with this project's jax>=0.4.38)
publishes no macosx_x86_64 wheel, and jax2onnx needs jaxlib>=0.6.2, which has also
dropped Intel-mac wheels. So the ONNX graph is built directly with onnx.helper,
mirroring train_drone_policy.py's ActorCritic.__call__ (with deterministic=True,
so it returns (action_mean, value) instead of a distrax distribution) and
rad_embeddings.encoder.{EncoderModule,GATv2Conv} node-for-node -- including the
jraph segment_max/segment_softmax/segment_sum ops, reimplemented via a plain
Reshape+ReduceMax/ReduceSum (edge_index's meshgrid structure makes every
source node's edges a contiguous block, so grouped reduction is just a
reshape -- see GraphBuilder.segment_reduce). Every exported checkpoint is
verified against a real jax network.apply(..., deterministic=True) call on
random inputs before being accepted (see verify_onnx()).

ONNX model I/O (matches a single, unbatched DroneEnv/DFAWrapper step; see
CLAUDE.md's "Task setup" section for what these mean):
  inputs:
    obs            float32[6]     -- drone (position(3), velocity(3))
    node_features  float32[5,4]   -- DFAx.to_graph() node features, max_size=5
    edge_features  float32[25,13] -- DFAx.to_graph() edge features, n_tokens+8=13
    current_state  int64[1]       -- index of the DFA's current state
    n_states       int64[5]       -- true state count, broadcast across all 5 slots
  outputs:
    action_mean    float32[1,3]   -- deterministic policy output (already
                                      tanh-squashed and scaled by max_action)
    value          float32[1]     -- critic value estimate

`edge_index` is NOT an input: DFAx.to_graph() always builds it as the full
max_size x max_size meshgrid, independent of the sampled DFA, so it's baked in
as a constant for this max_size=5 export.
"""
import os
import re
import numpy as np
import jax
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper
from flax.traverse_util import flatten_dict
import flax.serialization as serialization

from dfa_gym import DroneEnv, DFAWrapper
from dfax.samplers import ReachAvoidSampler
from rad_embeddings import Encoder, EncoderModule
from wrappers import LogWrapper
from train_drone_policy import ActorCritic

OPSET = 18
MAX_SIZE = 5
N_HEADS = 4
ENCODER_DIM = 32
HIDDEN_DIM = ENCODER_DIM * 2  # 64, GATv2Conv out_dim
N_EDGES = MAX_SIZE * MAX_SIZE  # 25


# --------------------------------------------------------------------------
# Minimal ONNX graph builder
# --------------------------------------------------------------------------
class GraphBuilder:
    def __init__(self):
        self.nodes = []
        self.initializers = []
        self._counter = 0

    def _name(self, prefix):
        self._counter += 1
        return f"{prefix}_{self._counter}"

    def const(self, arr, name=None):
        arr = np.asarray(arr)
        name = name or self._name("const")
        self.initializers.append(numpy_helper.from_array(arr, name=name))
        return name

    def op(self, op_type, inputs, n_outputs=1, **attrs):
        outputs = [self._name(op_type.lower()) for _ in range(n_outputs)]
        self.nodes.append(helper.make_node(op_type, inputs, outputs, **attrs))
        return outputs[0] if n_outputs == 1 else outputs

    # -- convenience wrappers --
    def matmul(self, a, b):
        return self.op("MatMul", [a, b])

    def add(self, a, b):
        return self.op("Add", [a, b])

    def sub(self, a, b):
        return self.op("Sub", [a, b])

    def mul(self, a, b):
        return self.op("Mul", [a, b])

    def div(self, a, b):
        return self.op("Div", [a, b])

    def tanh(self, x):
        return self.op("Tanh", [x])

    def exp(self, x):
        return self.op("Exp", [x])

    def leaky_relu(self, x, alpha):
        return self.op("LeakyRelu", [x], alpha=float(alpha))

    def concat(self, xs, axis):
        return self.op("Concat", xs, axis=axis)

    def reshape(self, x, shape):
        shape_name = self.const(np.array(shape, dtype=np.int64))
        return self.op("Reshape", [x, shape_name])

    def gather(self, x, indices, axis):
        return self.op("Gather", [x, indices], axis=axis)

    def unsqueeze(self, x, axes):
        axes_name = self.const(np.array(axes, dtype=np.int64))
        return self.op("Unsqueeze", [x, axes_name])

    def squeeze(self, x, axes):
        axes_name = self.const(np.array(axes, dtype=np.int64))
        return self.op("Squeeze", [x, axes_name])

    def where(self, cond, x, y):
        return self.op("Where", [cond, x, y])

    def less(self, a, b):
        return self.op("Less", [a, b])

    def equal(self, a, b):
        return self.op("Equal", [a, b])

    def reduce_op(self, op_type, x, axes, keepdims=0):
        return self.op(op_type, [x, self.const(np.array(axes, dtype=np.int64))], keepdims=keepdims)

    def reduce_sum(self, x, axes, keepdims=0):
        return self.reduce_op("ReduceSum", x, axes, keepdims)

    def segment_reduce(self, updates, updates_shape, reduction):
        """Group `updates` (shape (MAX_SIZE*MAX_SIZE, ...)) by source node and
        reduce ('max' or 'add') within each group, giving shape (MAX_SIZE, ...).

        DFAx.to_graph()'s edge_index is np.meshgrid(arange(MAX_SIZE),
        arange(MAX_SIZE), indexing="ij") flattened, so the MAX_SIZE edges
        belonging to source node `n` are exactly the contiguous block
        updates[n*MAX_SIZE:(n+1)*MAX_SIZE] -- a plain reshape (no gather/
        scatter needed) turns "reduce per source node" into "reduce over
        axis 1", which every ONNX runtime supports (unlike ScatterElements'
        newer duplicate-index reduction modes).
        """
        reshaped = self.reshape(updates, [MAX_SIZE, MAX_SIZE] + list(updates_shape[1:]))
        op_type = {"max": "ReduceMax", "add": "ReduceSum"}[reduction]
        return self.reduce_op(op_type, reshaped, axes=[1], keepdims=0)

    def dense(self, x, kernel, bias=None):
        y = self.matmul(x, kernel)
        if bias is not None:
            y = self.add(y, bias)
        return y


# --------------------------------------------------------------------------
# Config parsing / env+network reconstruction (mirrors train_drone_policy.py)
# --------------------------------------------------------------------------
CKPT_RE = re.compile(
    r"policy_params_drone_seed_(?P<seed>\d+)_ReachAvoid_(?P<max_size>\d+)_(?P<n_tokens>\d+)_"
    r"(?P<rad>rad|no_rad)_(?P<reward>binary|shaped)_"
    r"x(?P<x_low>-?[\d.]+)_(?P<x_high>-?[\d.]+)_"
    r"y(?P<y_low>-?[\d.]+)_(?P<y_high>-?[\d.]+)_"
    r"z(?P<z_low>-?[\d.]+)_(?P<z_high>-?[\d.]+)_"
    r"speed(?P<speed>[\d.]+)_dt(?P<dt>[\d.]+)_"
    r"(?P<action_mode>vel|disp)_steps(?P<steps>\d+)\.msgpack$"
)


def parse_checkpoint_name(fname):
    m = CKPT_RE.match(fname)
    if not m:
        raise ValueError(f"Unrecognized checkpoint filename: {fname}")
    g = m.groupdict()
    return dict(
        seed=int(g["seed"]),
        max_size=int(g["max_size"]),
        n_tokens=int(g["n_tokens"]),
        rad=(g["rad"] == "rad"),
        binary_reward=(g["reward"] == "binary"),
        x_low=float(g["x_low"]), x_high=float(g["x_high"]),
        y_low=float(g["y_low"]), y_high=float(g["y_high"]),
        z_low=float(g["z_low"]), z_high=float(g["z_high"]),
        max_speed=float(g["speed"]), dt=float(g["dt"]),
        use_displacement_action=(g["action_mode"] == "disp"),
        max_steps_in_episode=int(g["steps"]),
    )


def build_network_and_params(ckpt_path):
    cfg = parse_checkpoint_name(os.path.basename(ckpt_path))
    assert cfg["max_size"] == MAX_SIZE, "this exporter hardcodes MAX_SIZE=5"

    drone_env = DroneEnv(
        n_agents=1,
        x_low=cfg["x_low"], x_high=cfg["x_high"],
        y_low=cfg["y_low"], y_high=cfg["y_high"],
        z_low=cfg["z_low"], z_high=cfg["z_high"],
        max_speed=cfg["max_speed"], dt=cfg["dt"],
        use_displacement_action=cfg["use_displacement_action"],
        max_steps_in_episode=cfg["max_steps_in_episode"],
    )
    sampler = ReachAvoidSampler(max_size=cfg["max_size"], n_tokens=drone_env.n_tokens, p=None)
    env = DFAWrapper(env=drone_env, gamma=None, sampler=sampler, binary_reward=cfg["binary_reward"])
    env = LogWrapper(env=env, config={"LOG": None})

    if cfg["rad"]:
        encoder = Encoder(max_size=sampler.max_size, n_tokens=drone_env.n_tokens, seed=cfg["seed"])
    else:
        encoder = EncoderModule(max_size=sampler.max_size)

    network = ActorCritic(
        action_dim=env.action_space(env.agents[0]).shape[0],
        encoder=encoder,
        n_agents=env.num_agents,
        max_action=drone_env.max_action,
        deterministic=True,
    )

    key = jax.random.PRNGKey(0)
    init_x = env.observation_space(env.agents[0]).sample(key)
    params = network.init(key, init_x)
    with open(ckpt_path, "rb") as f:
        params = serialization.from_bytes(params, f.read())

    return cfg, env, network, params, init_x


def get_weights(cfg, network, params):
    flat = flatten_dict(params, sep="/")
    w = {k: np.asarray(v) for k, v in flat.items()}

    head = {
        "Dense_0": (w["params/Dense_0/kernel"], w["params/Dense_0/bias"]),
        "Dense_1": (w["params/Dense_1/kernel"], w["params/Dense_1/bias"]),
        "Dense_2": (w["params/Dense_2/kernel"], w["params/Dense_2/bias"]),
        "Dense_3": (w["params/Dense_3/kernel"], w["params/Dense_3/bias"]),
        "Dense_4": (w["params/Dense_4/kernel"], w["params/Dense_4/bias"]),
        "Dense_5": (w["params/Dense_5/kernel"], w["params/Dense_5/bias"]),
        "Dense_6": (w["params/Dense_6/kernel"], w["params/Dense_6/bias"]),
        "Dense_7": (w["params/Dense_7/kernel"], w["params/Dense_7/bias"]),
    }

    if cfg["rad"]:
        enc_flat = flatten_dict(network.encoder.encoder_params, sep="/")
        enc_w = {k: np.asarray(v) for k, v in enc_flat.items()}
        prefix = "params"
    else:
        enc_w = w
        prefix = "params/encoder"

    encoder = {
        "linear_h": enc_w[f"{prefix}/linear_h/kernel"],
        "linear_e": enc_w[f"{prefix}/linear_e/kernel"],
        "W_a": enc_w[f"{prefix}/gatv2/W_a/kernel"],
        "W_m": enc_w[f"{prefix}/gatv2/W_m/kernel"],
        "a": enc_w[f"{prefix}/gatv2/a/kernel"],
        "g_embed": enc_w[f"{prefix}/g_embed/kernel"],
    }

    return head, encoder


# --------------------------------------------------------------------------
# ONNX graph construction
# --------------------------------------------------------------------------
def build_onnx_model(cfg, head_w, enc_w, max_action, action_dim):
    b = GraphBuilder()

    # graph inputs
    obs_in = "obs"
    node_features_in = "node_features"
    edge_features_in = "edge_features"
    current_state_in = "current_state"
    n_states_in = "n_states"

    inputs = [
        helper.make_tensor_value_info(obs_in, TensorProto.FLOAT, [6]),
        helper.make_tensor_value_info(node_features_in, TensorProto.FLOAT, [MAX_SIZE, 4]),
        helper.make_tensor_value_info(edge_features_in, TensorProto.FLOAT, [N_EDGES, cfg["n_tokens"] + 8]),
        helper.make_tensor_value_info(current_state_in, TensorProto.INT64, [1]),
        helper.make_tensor_value_info(n_states_in, TensorProto.INT64, [MAX_SIZE]),
    ]

    def w(arr, name):
        return b.const(arr.astype(np.float32), name=name)

    # ---- obs MLP: Dense_0 -> tanh -> Dense_1 -> tanh ----
    obs_b = b.unsqueeze(obs_in, axes=[0])  # (1, 6)
    k0, bs0 = head_w["Dense_0"]
    k1, bs1 = head_w["Dense_1"]
    h = b.tanh(b.dense(obs_b, w(k0, "d0_k"), w(bs0, "d0_b")))
    obs_feat = b.tanh(b.dense(h, w(k1, "d1_k"), w(bs1, "d1_b")))  # (1, 64)

    # ---- DFA encoder: EncoderModule / GATv2Conv, node-for-node ----
    srcs, tgts = np.meshgrid(np.arange(MAX_SIZE), np.arange(MAX_SIZE), indexing="ij")
    src_idx = b.const(srcs.flatten().astype(np.int64), name="src_idx")
    tgt_idx = b.const(tgts.flatten().astype(np.int64), name="tgt_idx")

    h0 = b.matmul(node_features_in, w(enc_w["linear_h"], "lin_h_k"))  # (5, 64)
    e = b.matmul(edge_features_in, w(enc_w["linear_e"], "lin_e_k"))  # (25, 64)

    abs_sum = b.reduce_sum(b.op("Abs", [edge_features_in]), axes=[-1], keepdims=0)  # (25,)
    attn_mask = b.op("Greater", [abs_sum, b.const(np.float32(0.0), name="zero_f")])  # (25,) bool
    attn_mask_col = b.unsqueeze(attn_mask, axes=[-1])  # (25, 1) bool, broadcasts vs (25, H)

    neg_inf = b.const(np.float32(-np.inf), name="neg_inf")
    zeros_e_h = b.const(np.zeros((N_EDGES, N_HEADS), dtype=np.float32), name="zeros_e_h")

    W_a = w(enc_w["W_a"], "W_a")
    W_m = w(enc_w["W_m"], "W_m")
    a_kernel = w(enc_w["a"], "a_k")

    h_cur = h0
    for i in range(MAX_SIZE):
        node_cat = b.concat([h_cur, h0], axis=-1)  # (5, 128)
        src_feat = b.gather(node_cat, src_idx, axis=0)  # (25, 128)
        tgt_feat = b.gather(node_cat, tgt_idx, axis=0)  # (25, 128)

        ha_in = b.concat([src_feat, e, tgt_feat], axis=-1)  # (25, 320)
        ha = b.reshape(b.matmul(ha_in, W_a), [N_EDGES, N_HEADS, HIDDEN_DIM])  # (25,4,64)

        hm_in = b.concat([e, tgt_feat], axis=-1)  # (25, 192)
        hm = b.reshape(b.matmul(hm_in, W_m), [N_EDGES, N_HEADS, HIDDEN_DIM])  # (25,4,64)

        ha_act = b.leaky_relu(ha, alpha=0.2)
        logits3d = b.matmul(ha_act, a_kernel)  # (25, 4, 1)
        logits = b.squeeze(logits3d, axes=[-1])  # (25, 4)
        logits = b.where(attn_mask_col, logits, neg_inf)

        max_per_node = b.segment_reduce(logits, (N_EDGES, N_HEADS), "max")  # (5,4)
        col0 = b.gather(max_per_node, b.const(np.array(0, dtype=np.int64), name=f"c0_{i}"), axis=1)  # (5,)
        dead_nodes = b.equal(col0, neg_inf)  # (5,) bool
        dead_mask = b.gather(dead_nodes, src_idx, axis=0)  # (25,) bool
        dead_mask_col = b.unsqueeze(dead_mask, axes=[-1])
        safe_logits = b.where(dead_mask_col, zeros_e_h, logits)  # (25, 4)

        smax = b.segment_reduce(safe_logits, (N_EDGES, N_HEADS), "max")
        smax_g = b.gather(smax, src_idx, axis=0)
        shifted = b.sub(safe_logits, smax_g)
        expv = b.exp(shifted)
        ssum = b.segment_reduce(expv, (N_EDGES, N_HEADS), "add")
        ssum_g = b.gather(ssum, src_idx, axis=0)
        attn = b.div(expv, ssum_g)  # (25, 4)

        msgs = b.mul(b.unsqueeze(attn, axes=[-1]), hm)  # (25, 4, 64)
        h_scatter = b.segment_reduce(msgs, (N_EDGES, N_HEADS, HIDDEN_DIM), "add")  # (5,4,64)
        h_new = b.tanh(b.reduce_sum(h_scatter, axes=[1], keepdims=0))  # (5, 64)

        mask_i = b.less(b.const(np.array(i, dtype=np.int64), name=f"iter_{i}"), n_states_in)  # (5,) bool
        mask_i_col = b.unsqueeze(mask_i, axes=[-1])
        h_cur = b.where(mask_i_col, h_new, h_cur)

    dfa_node = b.gather(h_cur, current_state_in, axis=0)  # (1, 64)
    dfa_feat = b.matmul(dfa_node, w(enc_w["g_embed"], "g_embed_k"))  # (1, 32)

    # ---- heads ----
    feat = b.concat([obs_feat, dfa_feat], axis=-1)  # (1, 96)

    k2, bs2 = head_w["Dense_2"]
    k3, bs3 = head_w["Dense_3"]
    k4, bs4 = head_w["Dense_4"]
    v = b.tanh(b.dense(feat, w(k2, "d2_k"), w(bs2, "d2_b")))
    v = b.tanh(b.dense(v, w(k3, "d3_k"), w(bs3, "d3_b")))
    v = b.dense(v, w(k4, "d4_k"), w(bs4, "d4_b"))  # (1, 1)
    value_out = b.squeeze(v, axes=[-1])  # (1,)

    k5, bs5 = head_w["Dense_5"]
    k6, bs6 = head_w["Dense_6"]
    k7, bs7 = head_w["Dense_7"]
    m = b.tanh(b.dense(feat, w(k5, "d5_k"), w(bs5, "d5_b")))
    m = b.tanh(b.dense(m, w(k6, "d6_k"), w(bs6, "d6_b")))
    m = b.dense(m, w(k7, "d7_k"), w(bs7, "d7_b"))  # (1, action_dim)
    action_mean_out = b.mul(b.tanh(m), b.const(np.float32(max_action), name="max_action"))

    outputs = [
        helper.make_tensor_value_info(action_mean_out, TensorProto.FLOAT, [1, action_dim]),
        helper.make_tensor_value_info(value_out, TensorProto.FLOAT, [1]),
    ]

    graph = helper.make_graph(b.nodes, "drone_actor_critic", inputs, outputs, initializer=b.initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    # onnx 1.23's make_model() defaults ir_version to the installed onnx package's
    # own (very new) IR_VERSION, not to what opset 18 actually needs -- pin it to
    # IR 9 (the version opset 18 was introduced with) so onnxruntime 1.20 can load it.
    model.ir_version = 9
    onnx.checker.check_model(model)
    return model


# --------------------------------------------------------------------------
# Verification against the real JAX forward pass
# --------------------------------------------------------------------------
def verify_onnx(onnx_path, network, params, env, n_trials=8, seed=0):
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    key = jax.random.PRNGKey(seed)
    max_diff_action = 0.0
    max_diff_value = 0.0
    for t in range(n_trials):
        key, k_obs, k_dfa = jax.random.split(key, 3)
        obs_sample = env.observation_space(env.agents[0]).sample(k_obs)
        # keep obs continuous but use a fresh random DFA graph each trial by resampling
        dfa = env.sampler.sample(k_dfa)
        graph = dfa.to_graph()
        obs_sample = dict(obs_sample)
        obs_sample["dfa"] = graph

        jax_out = network.apply(params, obs_sample)
        action_mean_jax, value_jax = jax.tree_util.tree_map(np.asarray, jax_out)

        onnx_inputs = {
            "obs": np.asarray(obs_sample["obs"], dtype=np.float32),
            "node_features": np.asarray(graph["node_features"], dtype=np.float32),
            "edge_features": np.asarray(graph["edge_features"], dtype=np.float32),
            "current_state": np.asarray(graph["current_state"], dtype=np.int64),
            "n_states": np.asarray(graph["n_states"], dtype=np.int64),
        }
        action_mean_onnx, value_onnx = sess.run(None, onnx_inputs)

        max_diff_action = max(max_diff_action, float(np.max(np.abs(action_mean_jax - action_mean_onnx))))
        max_diff_value = max(max_diff_value, float(np.max(np.abs(value_jax - value_onnx))))

    return max_diff_action, max_diff_value


def main():
    storage_dir = os.path.join(os.path.dirname(__file__), "storage")
    out_dir = os.path.join(storage_dir, "onnx")
    os.makedirs(out_dir, exist_ok=True)

    ckpts = sorted(
        f for f in os.listdir(storage_dir)
        if f.startswith("policy_params_drone_") and f.endswith(".msgpack")
    )

    results = []
    for fname in ckpts:
        ckpt_path = os.path.join(storage_dir, fname)
        print(f"\n=== {fname} ===")
        cfg, env, network, params, init_x = build_network_and_params(ckpt_path)
        head_w, enc_w = get_weights(cfg, network, params)

        model = build_onnx_model(cfg, head_w, enc_w, network.max_action, network.action_dim)
        out_path = os.path.join(out_dir, fname.replace(".msgpack", ".onnx"))
        onnx.save(model, out_path)

        max_diff_action, max_diff_value = verify_onnx(out_path, network, params, env)
        print(f"  saved: {out_path}")
        print(f"  max |Δ action_mean| = {max_diff_action:.3e}, max |Δ value| = {max_diff_value:.3e}")
        results.append((fname, max_diff_action, max_diff_value))

    print("\n=== summary ===")
    worst = 0.0
    for fname, da, dv in results:
        worst = max(worst, da, dv)
        print(f"{fname:95} action_diff={da:.3e} value_diff={dv:.3e}")
    print(f"\nworst max-abs-diff across all checkpoints: {worst:.3e}")


if __name__ == "__main__":
    main()
