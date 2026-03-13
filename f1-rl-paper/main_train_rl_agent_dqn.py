"""
TRAINING SCRIPT (DQN)
- Joint training across 2019 races.
- Always pads observations to target_dim=40 to avoid spec mismatch.
"""

import os
import csv
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

import machine_learning_rl_training
import racesim

from tf_agents.environments import tf_py_environment
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory, time_step as ts, policy_step as ps
from tf_agents.utils import common

from machine_learning_rl_training.src.rsrl_reward_wrapper import RSRLRewardWrapper
from machine_learning_rl_training.reward.step_reward import RewardConfig
from machine_learning_rl_training.estimators.ep_value import EPFromStateMean
from machine_learning_rl_training.estimators.baseline import BaselineOneLap
from machine_learning_rl_training.logging.ep_logger import EPLogger
from machine_learning_rl_training.src.pad_obs_wrapper import PadObservationWrapper
from machine_learning_rl_training.ep_next_predictor import EpNextPredictor
from machine_learning_rl_training.ep_next_dataset import EpNextDatasetWriter
from pathlib import Path
# 맨 위 (import tensorflow 이전!)
import sys
from tqdm import trange

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


PARAM_DIR = ROOT / "racesim" / "input" / "parameters"

def discover_races(season: int):
    files = sorted(PARAM_DIR.glob(f"pars_*_{season}.ini"))
    races = []
    for f in files:
        name = f.stem  # pars_Austin_2019
        track = name[len("pars_") : -(len(f"_{season}"))]
        races.append(f"{track}_{season}")
    return races

# Seasons can be configured via env; default to 2017/2018/2019 for backward-compat
TRAIN_SEASON = int(os.environ.get("TRAIN_SEASON", "2017"))
VAL_SEASON   = int(os.environ.get("VAL_SEASON", "2018"))
TEST_SEASON  = int(os.environ.get("TEST_SEASON", "2019"))

RACES_TRAIN = discover_races(TRAIN_SEASON)
RACES_VAL   = discover_races(VAL_SEASON)
RACES_TEST  = discover_races(TEST_SEASON)

DEBUG_ONLY = os.environ.get("DEBUG_ONLY", "0") == "1"
DEBUG_RACES = [r.strip() for r in os.environ.get("DEBUG_RACES", "Baku_2019,Spa_2019").split(",") if r.strip()]

if DEBUG_ONLY:
    # 디버그면 train/val/test 다 같은 디버그 레이스로 맞춰도 되고,
    # 아니면 train만 디버그로 줄여도 됨. 여기선 단순하게 다 동일로 맞춤.
    RACES_TRAIN = DEBUG_RACES
    RACES_VAL   = DEBUG_RACES
    RACES_TEST  = DEBUG_RACES

print("[TRAIN]", len(RACES_TRAIN), RACES_TRAIN[:5])
print("[VAL  ]", len(RACES_VAL),   RACES_VAL[:5])
print("[TEST ]", len(RACES_TEST),  RACES_TEST[:5])



os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

# ---------- paths ----------
repo_path_ = os.path.dirname(os.path.abspath(__file__))
output_path_ = os.path.join(repo_path_, "machine_learning_rl_training", "output")
os.makedirs(output_path_, exist_ok=True)

# ---------- config ----------
race = os.environ.get("RACE", "Shanghai_2019")
vse_others = "supervised"
mcs_pars_file = "pars_mcs.ini"

# Allow total training steps to be configured via env (default 250k).
try:
    num_iterations = int(os.environ.get("NUM_ITERATIONS", "250000"))
except ValueError:
    num_iterations = 250_000
replay_buffer_max_length = 200_000
initial_collect_steps = 200
collect_steps_per_iteration = 1

fc_layer_params = (64, 64)
batch_size = 64
learning_rate = 1e-3
gamma = 1.0
n_step_update = 1
target_update_period = 1

# logging / eval / checkpoint intervals
_default_log_interval = max(num_iterations // 10, 1)  # 10% progress
log_interval = int(os.environ.get("LOG_INTERVAL", str(_default_log_interval)))
eval_interval = int(os.environ.get("EVAL_INTERVAL", "50000"))
checkpoint_interval = int(os.environ.get("CHECKPOINT_INTERVAL", "10000"))
num_eval_episodes = 100

# ---------- SMOKE ----------
SMOKE = os.environ.get("SMOKE", "0") == "1"
if SMOKE:
    num_iterations = 500
    initial_collect_steps = 20
    num_eval_episodes = 3
    log_interval = 100
    eval_interval = 200
    checkpoint_interval = 100
    replay_buffer_max_length = 10_000
    print("[SMOKE] enabled")

# ---------- SEED ----------
def set_seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism(True)
    except Exception:
        pass

SEED = int(os.environ.get("SEED", "20250601"))
set_seed_all(SEED)
print(f"[SEED] {SEED}")

# ---------- reward config ----------
def _f_env(key, default):
    v = os.environ.get(key)
    return type(default)(v) if v is not None else default

rsrl_cfg = RewardConfig(
    l_lap=_f_env("L_LAP", 0.2),
    l_pos=_f_env("L_POS", 1.0),
    l_pit=_f_env("L_PIT", 0.2),
    l_risk=_f_env("L_RISK", 0.0),
    l_stint=_f_env("L_STINT", 0.0),
    k_stint=_f_env("K_STINT", 1.0),
    min_stint_laps=2,
    k_t=_f_env("K_T", 1 / 7.0),
    k_pos=_f_env("K_POS", 1.0),
    k_pit=_f_env("K_PIT", 1 / 7.0),
    use_deltaEP_next=os.environ.get("USE_DELTA_EP", "0") == "1",
    use_potential=os.environ.get("USE_POTENTIAL", "0") == "1",
    ep_next_clip=_f_env("EP_NEXT_CLIP", 5.0),
    ep_next_warmup_steps=_f_env("EP_NEXT_WARMUP", 5000),
)
print("[REWARD]", rsrl_cfg)

def _append_result_csv(csv_path: str, row: dict):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            w.writeheader()
        w.writerow(row)

def _load_ep_logs(track, season):
    p = Path(f"datasets/logs/{track}_{season}")
    files = list(p.glob("*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else pd.DataFrame()

def build_wrapped_env_for_race(race_name: str, *, log_ep: bool, create_rand_events: bool):
    race_pars_file = f"pars_{race_name}.ini"

    # wet race check (SKIP instead of crash)
    pars_in = racesim.src.import_pars.import_pars(
        use_print=False,
        use_vse=False,
        race_pars_file=race_pars_file,
        mcs_pars_file=mcs_pars_file,
    )[0]

    is_wet = False
    for driver in pars_in["driver_pars"]:
        if any(strat[1] in ["I", "W"] for strat in pars_in["driver_pars"][driver]["strategy_info"]):
            is_wet = True
            break

    if is_wet:
        print(f"[SKIP] {race_name}: wet race (I/W in strategy) -> skip this race")
        return None  # ✅ 핵심: 에러 대신 None 반환
    # base env
    base_env = machine_learning_rl_training.src.rl_environment_single_agent.RaceSimulation(
        race_pars_file=race_pars_file,
        mcs_pars_file=mcs_pars_file,
        vse_type=vse_others,
        use_prob_infl=True,
        create_rand_events=create_rand_events,
    )

        
    # track/season
    try:
        track, season = race_name.split("_")[0], int(race_name.split("_")[1])
    except Exception:
        track, season = race_name, 0

    # EP estimators
    df_logs = _load_ep_logs(track, season)
    ep_from_state = EPFromStateMean(df_logs=df_logs, default_pts=6.0)

    def _env_factory():
        e = machine_learning_rl_training.src.rl_environment_single_agent.RaceSimulation(
            race_pars_file=race_pars_file,
            mcs_pars_file=mcs_pars_file,
            vse_type=vse_others,
            use_prob_infl=True,
            create_rand_events=create_rand_events,
        )
        try:
            e.seed(SEED)
        except Exception:
            pass
        return e

    baseline_next = BaselineOneLap(env_factory=_env_factory, ep_from_state=ep_from_state)

    ep_logger = EPLogger(outdir=f"datasets/logs/{track}_{season}") if log_ep else None

    epnext_logger = None
    if os.environ.get("EP_NEXT_LOG", "0") == "1":
        epnext_logger = EpNextDatasetWriter(outdir=Path(f"datasets/epnext_logs/{track}_{season}"))

    epnext_predictor = None
    model_path = os.environ.get("EP_NEXT_MODEL_PATH", "").strip()
    if model_path:
        try:
            epnext_predictor = EpNextPredictor(Path(model_path))
            print(f"[EP_NEXT] loaded model from {model_path}")
        except Exception as e:
            print("[WARN] failed to load ep_next predictor:", e)
            epnext_predictor = None

    wrapped = RSRLRewardWrapper(
        base_env=base_env,
        cfg=rsrl_cfg,
        ep_from_state=ep_from_state,
        baseline_nextlap=baseline_next,
        ep_logger=ep_logger,
        meta={"track": track, "season": season, "round": 1},
        ep_next_predictor=epnext_predictor,
        ep_next_logger=epnext_logger,
    )


    # ALWAYS pad in final pipeline (so diagnosis matches training)
    padded = PadObservationWrapper(wrapped, target_dim=40)
    
    return base_env, wrapped, padded

# ====== eval config ======
joint_tag = os.environ.get(
    "RUN_TAG",
    "train2017_val2018_test2019"
)

VAL_EPISODES  = int(os.environ.get("VAL_EPISODES", "20"))
TEST_EPISODES = int(os.environ.get("TEST_EPISODES", "30"))

metrics_csv = os.path.join(output_path_, "metrics", f"{joint_tag}_episodes.csv")


# ---------- batching helpers ----------
def _to_batched_ts(t: ts.TimeStep) -> ts.TimeStep:
    st = t.step_type
    rw = t.reward
    dc = t.discount
    ob = t.observation
    if tf.rank(st) == 0: st = tf.expand_dims(st, 0)
    if tf.rank(rw) == 0: rw = tf.expand_dims(rw, 0)
    if tf.rank(dc) == 0: dc = tf.expand_dims(dc, 0)
    if tf.rank(ob) == 1: ob = tf.expand_dims(ob, 0)
    return ts.TimeStep(step_type=st, reward=rw, discount=dc, observation=ob)

def _to_batched_action(a: ps.PolicyStep):
    act = a.action
    if tf.rank(act) == 0:
        act = tf.expand_dims(act, 0)
    return ps.PolicyStep(action=act, state=a.state, info=a.info)

def collect_step(env: tf_py_environment.TFPyEnvironment, policy, buffer):
    time_step_ = _to_batched_ts(env.current_time_step())
    action_step = _to_batched_action(policy.action(time_step_))
    next_time_step = _to_batched_ts(env.step(action_step.action))
    traj = trajectory.from_transition(time_step_, action_step, next_time_step)
    buffer.add_batch(traj)

def collect_data(env, policy, buffer, steps: int):
    for _ in range(steps):
        collect_step(env, policy, buffer)

# ---------- build envs ----------
# ---------- build envs ----------
def build_env_list(race_list, *, log_ep: bool, create_rand_events: bool):
    py_envs = []
    for rn in race_list:
        out = build_wrapped_env_for_race(rn, log_ep=log_ep, create_rand_events=create_rand_events)
        if out is None:
            continue
        _, _, padded = out
        py_envs.append(padded)
    return py_envs

train_py_envs = build_env_list(RACES_TRAIN, log_ep=True,  create_rand_events=True)
val_py_envs   = build_env_list(RACES_VAL,   log_ep=False, create_rand_events=False)
test_py_envs  = build_env_list(RACES_TEST,  log_ep=False, create_rand_events=False)

if len(train_py_envs) == 0:
    raise RuntimeError("No train envs built (all wet?)")
if len(val_py_envs) == 0:
    raise RuntimeError("No val envs built (all wet?)")
if len(test_py_envs) == 0:
    raise RuntimeError("No test envs built (all wet?)")

train_tf_envs = [tf_py_environment.TFPyEnvironment(e) for e in train_py_envs]
val_tf_envs   = [tf_py_environment.TFPyEnvironment(e) for e in val_py_envs]
test_tf_envs  = [tf_py_environment.TFPyEnvironment(e) for e in test_py_envs]

ref_tf_env = train_tf_envs[0]
print("INFO: built train/val/test =", len(train_tf_envs), len(val_tf_envs), len(test_tf_envs))
print("INFO: obs spec:", ref_tf_env.observation_spec())

spec = ref_tf_env.action_spec()
print("ACTION SPEC:", spec)

# ✅ 학습 중 eval은 val로 고정
eval_tf_envs = val_tf_envs


# ---------- agent ----------
q_net = q_network.QNetwork(
    input_tensor_spec=ref_tf_env.observation_spec(),
    action_spec=ref_tf_env.action_spec(),
    fc_layer_params=fc_layer_params,
)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
train_step_counter = tf.Variable(0)

boltzmann_fn = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=1.0,
    decay_steps=num_iterations,
    end_learning_rate=0.01,
)

agent = dqn_agent.DqnAgent(
    time_step_spec=ref_tf_env.time_step_spec(),
    action_spec=ref_tf_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    n_step_update=n_step_update,
    target_update_period=target_update_period,
    td_errors_loss_fn=common.element_wise_squared_loss,
    gamma=gamma,
    train_step_counter=train_step_counter,
    epsilon_greedy=None,
    boltzmann_temperature=lambda: boltzmann_fn(train_step_counter),
)
agent.initialize()

eval_policy = agent.policy
collect_policy = agent.collect_policy
random_policy = random_tf_policy.RandomTFPolicy(
    time_step_spec=ref_tf_env.time_step_spec(),
    action_spec=ref_tf_env.action_spec(),
)

rb_batch_size = ref_tf_env.batch_size if ref_tf_env.batch_size is not None else 1
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=rb_batch_size,
    max_length=replay_buffer_max_length,
)

# initial collect (balanced)
steps_per_env = max(initial_collect_steps // len(train_tf_envs), 1)
for e in train_tf_envs:
    collect_data(e, random_policy, replay_buffer, steps_per_env)

dataset = replay_buffer.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=batch_size,
    num_steps=n_step_update + 1,
    single_deterministic_pass=False,
).prefetch(3)
dataset_iterator = iter(dataset)


def _unwrap_pyenv(tf_env):
    e = tf_env.pyenv
    if hasattr(e, "envs"):
        e = e.envs[0]

    # ✅ PadObservationWrapper 같은 바깥 래퍼만 벗기고,
    # ✅ RSRLRewardWrapper에서 멈춘다 (meta가 여기 있음)
    while hasattr(e, "_env") and not isinstance(e, RSRLRewardWrapper):
        e = e._env
    return e

def run_episodes_and_dump(tf_env, policy, num_episodes: int, *, split: str, csv_path: str) -> float:
    """각 episode 끝날 때 summary+return을 CSV로 저장하고, 평균 return 반환"""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = not os.path.exists(csv_path)

    pyenv = _unwrap_pyenv(tf_env)

    total_return = 0.0
    with open(csv_path, "a", newline="") as f:
        fieldnames = [
            "ts","split","track","season","round","seed",
            "episode_idx",
            "return",
            "episode_length","pit_count","avg_lap_time","final_position",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()

        for ep_i in range(num_episodes):
            t = _to_batched_ts(tf_env.reset())
            ep_ret = 0.0

            while not bool(tf.reduce_any(t.is_last()).numpy()):
                a = policy.action(t)
                t = _to_batched_ts(tf_env.step(a.action))
                ep_ret += float(tf.reduce_mean(t.reward).numpy())

            total_return += ep_ret

            # ✅ wrapper가 저장해둔 episode summary 꺼내기
            summ = {}
            if hasattr(pyenv, "get_episode_summary"):
                try:
                    summ = pyenv.get_episode_summary()
                except Exception:
                    summ = {}

            row = {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "split": split,
                "track": summ.get("track"),
                "season": summ.get("season"),
                "round": summ.get("round"),
                "seed": summ.get("seed"),
                "episode_idx": ep_i,
                "return": ep_ret,
                "episode_length": summ.get("episode_length"),
                "pit_count": summ.get("pit_count"),
                "avg_lap_time": summ.get("avg_lap_time"),
                "final_position": summ.get("final_position"),
            }
            w.writerow(row)

    return total_return / float(num_episodes)


print("INFO: Starting training...")
agent.train = common.function(agent.train)

# checkpoint manager (create early for resume)
ckpt_dir = os.path.join(output_path_, "checkpoints", joint_tag)
Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
ckpt = tf.train.Checkpoint(step=train_step_counter, q_net=agent._q_network, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=10)
print("INFO: checkpoint_interval =", checkpoint_interval)

resume = os.environ.get("RESUME", "0") == "1"
if resume and ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    step_after_restore = int(agent.train_step_counter.numpy())
    print("INFO: Resumed from checkpoint, step =", step_after_restore)
    # Re-fill replay buffer (lost on restart)
    steps_per_env = max(initial_collect_steps // len(train_tf_envs), 1)
    for e in train_tf_envs:
        collect_data(e, collect_policy, replay_buffer, steps_per_env)
    print("INFO: Replay buffer refilled with", initial_collect_steps, "steps")
    dataset_iterator = iter(dataset)
    avg_return_pre = None  # skip val_pre when resuming
else:
    agent.train_step_counter.assign(0)
    # ---- pre eval (val_pre) ----
    pre_returns = [
        run_episodes_and_dump(e, eval_policy, VAL_EPISODES, split="val_pre", csv_path=metrics_csv)
        for e in eval_tf_envs
    ]
    avg_return_pre = float(np.mean(pre_returns))
    print(f"INFO: Pre avg_return(val) across {len(eval_tf_envs)} races = {avg_return_pre:.3f}")
    print("INFO: episode metrics saved to:", metrics_csv)

start_step = int(agent.train_step_counter.numpy())
num_steps_to_run = max(0, num_iterations - start_step)
print("INFO: Training from step %d to %d (%d steps)" % (start_step, num_iterations, num_steps_to_run))

for i in trange(num_steps_to_run):
    env = random.choice(train_tf_envs)
    for j in range(collect_steps_per_iteration):
        collect_step(env, collect_policy, replay_buffer)

    experience = next(dataset_iterator)[0]
    loss = agent.train(experience).loss
    step = int(agent.train_step_counter.numpy())  # ✅ 여기서 step 생김

    if step % log_interval == 0:
        print("INFO: Step %d, loss %.3f" % (step, loss))

    if step % eval_interval == 0:
        rs = [
            run_episodes_and_dump(
                e, eval_policy, VAL_EPISODES,
                split=f"val_step{step}", csv_path=metrics_csv
            )
            for e in eval_tf_envs
        ]
        print("INFO: Step %d, avg_return(val) %.3f" % (step, float(np.mean(rs))))

    if step % checkpoint_interval == 0:
        ckpt_manager.save(checkpoint_number=step)
        print("INFO: checkpoint saved at step %d" % step)
# ---- post eval ----
post_returns = [
    run_episodes_and_dump(e, eval_policy, VAL_EPISODES, split="val_post", csv_path=metrics_csv)
    for e in eval_tf_envs
]
avg_return_post = float(np.mean(post_returns))
print(f"INFO: Post avg_return(val) across {len(eval_tf_envs)} races = {avg_return_post:.3f}")

# ---- final test ----
test_returns = [
    run_episodes_and_dump(e, eval_policy, TEST_EPISODES, split="test_final", csv_path=metrics_csv)
    for e in test_tf_envs
]
print(f"RESULT: Test avg_return across {len(test_tf_envs)} races = {float(np.mean(test_returns)):.3f}")
print("RESULT: episode metrics saved to:", metrics_csv)

# save final checkpoint/policy
ckpt_manager.save(checkpoint_number=int(agent.train_step_counter.numpy()))
print("RESULT: checkpoint saved to:", ckpt_dir)

policy_dir = os.path.join(output_path_, "policy_savedmodel", joint_tag)
os.makedirs(policy_dir, exist_ok=True)
from tf_agents.policies import policy_saver
policy_saver.PolicySaver(agent.policy).save(policy_dir)
print("RESULT: policy saved to:", policy_dir)

# write CSV
results_dir = os.path.join(output_path_, "rsrl_sweeps")
results_csv = os.path.join(results_dir, "multi_2019_results.csv")
row = {
    "ts": datetime.now().isoformat(timespec="seconds"),
    "race": joint_tag,
    "seed": SEED,
    "SMOKE": int(SMOKE),
    "num_iter": num_iterations,
    "batch_size": batch_size,
    "gamma": gamma,
    "lr": learning_rate,
    "fc_layers": "-".join(map(str, fc_layer_params)),
    "avg_return_pre": avg_return_pre,
    "avg_return_post": avg_return_post,
    # reward cfg dump
    "l_lap": rsrl_cfg.l_lap,
    "l_pos": rsrl_cfg.l_pos,
    "l_pit": rsrl_cfg.l_pit,
    "l_stint": rsrl_cfg.l_stint,
    "k_stint": rsrl_cfg.k_stint,
    "k_t": rsrl_cfg.k_t,
    "k_pos": rsrl_cfg.k_pos,
    "k_pit": rsrl_cfg.k_pit,
    "use_deltaEP_next": int(bool(rsrl_cfg.use_deltaEP_next)),
    "use_potential": int(bool(rsrl_cfg.use_potential)),
}
_append_result_csv(results_csv, row)
print("RESULT: sweep row appended to", results_csv)

