# PySaacSim — Remote Train-on-Photo Implementation Plan

**Goal.** A user takes a phone photo of a wooden-block track, uploads it from an iOS app to a FastAPI server running on the user's desktop (3060 Ti, Ryzen 7 5800X), confirms the reconstructed track, and ≤10 minutes later receives optimized policy weights ready to flash to the real car.

**Non-goals.** Multi-user SaaS. Cloud GPU. Real-time policy inference from the phone. Online learning on the car.

---

## 0. System overview

```
iPhone app  ──HTTPS──▶  Cloudflare Tunnel  ──▶  FastAPI (desktop)
  │ (ARKit rectifies                              │
  │  floor plane before upload)                   │
  │ 1) auth (Sign in with Apple)                  ├─ CV pipeline (SAM3 + scale-check)
  │ 2) POST /tracks (photo)                       ├─ Track store
  │ 3) confirm or edit track                      ├─ Job queue (1 GPU worker)
  │ 4) POST /jobs/train                           ├─ Training: SubprocVecEnv × 8
  │ 5) WS /jobs/{id}/events (progress)            └─ Policy export (.npz + .h)
  │ 6) GET /jobs/{id}/artifact (weights)
  └─ 7) APNs push when done
```

**End-to-end budget.** CV < 15 s. Fine-tune ≤ 8 min. Eval < 45 s. Export < 5 s. Total ≤ 10 min.

---

## 1. Prerequisites (already in repo)

- `sim/` — physics, geometry, sensors, calibration, default oval world. **Keep as-is.**
- `control/` — `AbstractController` + C-bridge. Used by hardware, not by RL training.
- `gui/` — PyQt desktop. **Keep as-is.** The new server runs alongside it.
- `examples/train_ppo.py` — existing PPO script. Will be refactored, not deleted.
- `requirements.txt` — will be extended, not replaced.

Missing (must be built):

- `env/robot_env.py` — referenced by `examples/train_ppo.py` and `gui/training/worker.py` but **does not exist**. This is the first thing to build.
- `env/multi_robot_env.py` — also referenced, same situation. Out of scope here; ignore for now.

---

## 2. Repository layout after this work

```
PySaacSim/
├── env/                        NEW
│   ├── __init__.py
│   ├── robot_env.py            Gymnasium env (was missing)
│   ├── track_gen.py            Procedural track generator for base-policy pre-training
│   └── reward.py               Reward shaping helpers
├── sim/
│   └── geometry_np.py          NEW  Vectorized ray-casting (NumPy)
├── training/                   NEW
│   ├── __init__.py
│   ├── base_policy.py          Offline base-policy trainer (one overnight run)
│   ├── finetune.py             Per-track fine-tuner (called from server)
│   ├── eval_gate.py            Post-train eval
│   └── export.py               npz + C-header export
├── cv/                         NEW
│   ├── __init__.py
│   ├── validate.py             Scale sanity-check via plank length (no ArUco)
│   ├── segment.py              SAM3 mask generation + filtering
│   └── build_track.py          Masks → Segment list + spawn pose
├── server/                     NEW
│   ├── __init__.py
│   ├── app.py                  FastAPI entrypoint
│   ├── auth.py                 Sign in with Apple → JWT
│   ├── jobs.py                 Job queue + state
│   ├── ws.py                   WebSocket progress hub
│   ├── apns.py                 APNs push client
│   ├── schemas.py              Pydantic models
│   └── storage.py              Filesystem layout under $PYSAAC_DATA
├── models/                     NEW  (gitignored)
│   ├── base_policy.zip         Offline-trained SB3 checkpoint
│   └── sam3.pt                 SAM3 weights (downloaded on first run)
├── docs/
│   └── implementation_plan.md  This file
├── ios/                        NEW  (Xcode project — not Python)
│   └── PysaacRC/               SwiftUI app
├── scripts/
│   ├── train_base_policy.sh
│   └── run_server.sh
├── tests/                      NEW
│   ├── test_robot_env.py
│   ├── test_geometry_np.py
│   ├── test_cv.py
│   ├── test_server_auth.py
│   ├── test_server_jobs.py
│   └── test_server_tracks.py
└── ...existing...
```

---

## 3. Phase 1 — `env/robot_env.py`

This is the missing piece blocking everything else. Build this first.

**Library.** `gymnasium >= 0.29`, already in requirements.

**Observation space.** `Box(low=0, high=1, shape=(obs_dim,), dtype=float32)` where `obs_dim = 5 sensors × frame_stack + 2 last_action`. Default `frame_stack=4` → `obs_dim = 22`. All values normalized:

| Index | Meaning                                        | Normalization                          |
|------:|------------------------------------------------|----------------------------------------|
| 0..3  | Center lidar distance (last 4 frames)          | `cm / lidar.max_cm` clipped to [0, 1]  |
| 4..7  | Left lidar distance                            | same                                   |
| 8..11 | Right lidar distance                           | same                                   |
| 12..15| Left IR distance                               | `cm / ir.max_cm` clipped to [0, 1]     |
| 16..19| Right IR distance                              | same                                   |
| 20    | Last throttle                                  | already in [−1, 1] → shift to [0, 1]   |
| 21    | Last steer                                     | same                                   |

Keep frame-stacking inside the env (deque of 4). Do **not** use SB3's `VecFrameStack` — we want per-env statefulness.

**Action space.** `Box(low=-1, high=1, shape=(2,), dtype=float32)`:
- `action[0]` throttle ∈ [−1, 1] → maps to `duty = int(action[0] * MOTOR_PWM_MAX_COUNT)`, sign sets `dir_l` / `dir_r`.
- `action[1]` steer ∈ [−1, 1] → maps to servo count: `SERVO_CENTER_COUNT + action[1] * (SERVO_MAX_COUNT - SERVO_CENTER_COUNT)`.

No slew limiting inside the env — let the physics sim handle it (it already models servo slew at `SERVO_RAD_PER_SEC`).

**Step loop.**

1. Convert action → `MotorCommand`.
2. Integrate physics for `control_period_s = 0.02` (50 Hz control), calling `sim.physics.step()` 20× at `PHYSICS_DT_S = 0.001`.
3. Sample sensors once via `sim.sensors.sample_sensors(walls, pose, cal)`.
4. Check collision (see below). If collided → terminate with penalty.
5. Compute reward (see below).
6. Push obs into frame-stack, increment step count, return.

**Collision check.** Use `sim.geometry.chassis_segments(pose, CHASSIS_LENGTH_CM, CHASSIS_WIDTH_CM)` and test each chassis segment against each wall with `seg_intersect`. Any hit = collision. Keep cheap: early-exit on first hit.

**Reward.** Implemented in `env/reward.py`:

```
r = w_progress * delta_progress_cm
  + w_speed    * max(0, v_forward_cms)
  - w_action   * (|throttle - last_throttle| + |steer - last_steer|)
  - w_offtrack * dist_from_centerline_cm_when_>_threshold
  - w_reverse  * max(0, -v_forward_cms)
collision: terminate, r += -50
timeout (max_episode_steps): truncate, no extra reward
```

Defaults: `w_progress=0.1`, `w_speed=0.005`, `w_action=0.02`, `w_offtrack=0.001`, `w_reverse=0.05`.

`delta_progress_cm` requires a **centerline**. Store one as part of the track (see §5 below). Progress = arclength projection of the robot's (x, y) onto the centerline polyline, minus the previous step's projection. Handle the loop closure so lap crossings don't produce a big negative jump.

**Termination vs truncation.** Collision → `terminated=True`. Step count ≥ `max_episode_steps=1500` → `truncated=True`. Required by Gymnasium API.

**Reset.**
- `options["walls"]`, `options["centerline"]`, `options["spawn"]` may be passed to override per-episode. Used by base-policy training for track randomization.
- If not provided, use `sim.world.build_default_world()` + `DEFAULT_SPAWN`.

**Pose / velocity.** Read `sim/physics.py` first — it defines a `Pose` (x, y, theta, v_body_x, v_body_y, omega) and a `step(pose, command, walls, dt) -> Pose` (verify the exact signature in the file; adapt if different). The env holds the current `Pose` between steps and derives `v_forward_cms = Pose.v_body_x * 100` for the reward. The 50 Hz control rate (`control_period_s = 0.02`) must match the firmware control loop; **verify against `hal/` before training**, otherwise the policy won't transfer.

**Centerline progress (pseudocode).** Maintain an integer `_arc_idx` of the last-projected centerline vertex; search only within `_arc_idx ± 10` for the next nearest segment, project, advance. This avoids `O(K)` nearest-neighbor per step and the loop-closure wrap issue (`delta` is always small and positive). Pre-compute cumulative arclength at env construction.

**Track JSON.** `sim.geometry.Segment` is a NamedTuple of Vec2 NamedTuples; it does not JSON-serialize natively. Add helpers `track_to_json(track) -> dict` / `track_from_json(d) -> dict` in `env/__init__.py` that convert Segments to `[[ax, ay], [bx, by]]` lists and back.

**Deliverable.** `env/robot_env.py` with a `RobotEnv(gym.Env)` class. Add a smoke test `tests/test_robot_env.py` that runs 100 random steps and asserts obs shape / dtype / action handling.

---

## 4. Phase 2 — Vectorized ray casting

`sim/geometry.py:cast_ray` loops over walls in Python. With 24 walls × 5 sensors × 50 Hz × 8 envs = 48k ray-segment tests per wall-clock second, which is the **dominant cost**. Vectorize it once.

**New file `sim/geometry_np.py`:**

```python
def cast_rays_batch(
    wall_a: np.ndarray,   # (W, 2) float32
    wall_b: np.ndarray,   # (W, 2) float32
    origins: np.ndarray,  # (N, 2) float32  — N = num rays in this batch
    dirs:    np.ndarray,  # (N, 2) float32 unit
    max_dist: np.ndarray  # (N,)   float32
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (distances (N,), hit_mask (N,)). distance = max_dist if no hit."""
```

Implementation: broadcast over (N, W) with the same cross-product math as `ray_hits_segment`; take `min` along wall axis. ~20 lines of NumPy. This replaces the per-ray Python loop and gives roughly 5–10× speedup per env.

Wire this into `sim/sensors.py` by replacing the per-sensor `cast_ray` calls with one batched call per `sample_sensors()` invocation (stack 5 rays per step). Keep the scalar `cast_ray` around for the GUI visualization path that wants hit points one at a time.

**Sanity test.** `tests/test_geometry_np.py` — fuzz 1000 random (origin, dir, wall) configs and assert the batched function matches the scalar one to `1e-5`.

---

## 5. Phase 3 — Procedural tracks + offline base policy

### 5.1 Track representation

A **track** is a dict:

```python
{
    "walls": list[Segment],          # all wall line segments in cm
    "centerline": np.ndarray,        # (K, 2), ordered, closed loop
    "spawn": {"x": float, "y": float, "theta": float},
    "lane_width_cm": float,
    "bounds": {"min_x", "min_y", "max_x", "max_y"},
}
```

The default oval in `sim/world.py` already produces almost this — add a centerline sampler (concentric midline between inner and outer rounded rects, 200 points).

### 5.2 Procedural generator (`env/track_gen.py`)

Used for pre-training. Generate a wide distribution of closed-loop tracks:

1. Sample a random smooth closed curve: N=6..10 control points on a circle of random radius (80..250 cm), perturbed radially by ±30%, then Catmull-Rom interpolated to 200 points. Reject self-intersecting curves.
2. Offset inward and outward by `lane_width / 2` (80 cm default ± 20%) to get inner/outer boundaries.
3. Discretize each boundary into straight segments every 10–20 cm → `Segment` list.
4. Centerline = the original curve.
5. Spawn = first centerline point, heading = tangent direction.

**Deliverable.** `generate_track(rng) -> dict` + a `render_track(track, path)` helper that dumps a PNG so you can eyeball the distribution.

### 5.3 Base policy training (`training/base_policy.py`)

Overnight run. Produces `models/base_policy.zip`.

- SB3 `PPO`, `MlpPolicy`, `net_arch=[128, 128]`, `activation_fn=nn.Tanh`.
- `SubprocVecEnv` with `n_envs=8`.
- Custom `VecEnv` subclass or a `make_env` closure that calls `env.reset(options={"walls": ..., "centerline": ...})` with a fresh procedural track **every episode** (domain randomization).
- Also randomize on reset: `calibration.lidar.noise_std_cm` ∈ [0.2, 1.5], `calibration.ir.voltage_noise_std` ∈ [0.02, 0.08], `MOTOR_LAG_TAU_S` via a wrapper ∈ [0.05, 0.2], wall thickness ∈ [6, 12] cm.
- `total_timesteps = 50_000_000`. Expect 6–10 hours on a 5800X once ray casting is vectorized.
- Hyperparams: `n_steps=2048`, `batch_size=512`, `learning_rate=3e-4`, `gamma=0.995`, `gae_lambda=0.95`, `clip_range=0.2`, `ent_coef=0.005`.
- TensorBoard to `./tb_logs/base/`.

Thread settings, applied in each subprocess' env constructor:

```python
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import torch
torch.set_num_threads(1)
```

**Checkpoint cadence.** Save every 1M steps to `models/base_policy_ckpt_{steps}.zip` so a crash mid-run isn't fatal.

**Eval gate at end of pre-training.** Run 50 fresh procedural tracks, require ≥ 70% completion rate and median lap time < 30 s. If not met, extend training.

### 5.4 Per-track fine-tuner (`training/finetune.py`)

Called by the server. Signature:

```python
def finetune(
    track: dict,
    base_ckpt_path: str,
    out_dir: str,
    total_timesteps: int = 2_000_000,
    n_envs: int = 8,
    progress_cb: Callable[[dict], None] | None = None,
    stop_event: threading.Event | None = None,
) -> dict:  # returns {"policy_path": ..., "eval": {...}}
```

- Loads `PPO.load(base_ckpt_path)`.
- Builds `SubprocVecEnv` with all 8 envs fixed to the given track. Reset only re-spawns at track start, does **not** regenerate geometry.
- Keeps domain randomization **on sensor calibration** but **off track geometry** (small DR to stay robust to CV error).
- Calls `model.set_env(vec)` then `model.learn(total_timesteps, callback=ProgressCb)`.
- `ProgressCb._on_rollout_end` calls `progress_cb({"step": t, "mean_reward": r, "fps": ...})`.
- Polls `stop_event` inside `_on_step` to support cancel.
- Target wall-clock: 2M steps @ ~15k sps = ~2.5 min on the dev box. Leave headroom up to 8 min.

### 5.5 Eval gate (`training/eval_gate.py`)

Given a trained policy and a track, run 20 episodes, collect:

- `completion_rate` — fraction that reached 1 full lap without collision.
- `mean_lap_time_s` — over completed laps.
- `collision_rate`.
- `mean_reward`.

Pass criteria: `completion_rate ≥ 0.8 AND collision_rate ≤ 0.1`. The server includes this dict in the job artifact and does **not** block delivery on failure — it surfaces the numbers to the app and lets the user choose.

### 5.6 Export (`training/export.py`)

Two outputs:

- `policy.npz` — keys: `obs_mean`, `obs_std` (from SB3's `VecNormalize` if used, else zeros/ones), `W0`, `b0`, `W1`, `b1`, `Wout`, `bout`, plus metadata string `meta_json`. Used by the Python eval path.
- `policy.h` — pure-C header with weights as `static const float` arrays + an inline `policy_forward(const float *obs, float *action)` function. For hardware deployment, flashable alongside the firmware. Generate with a simple Jinja-less f-string template; no extra dependency.

**Important deploy details.**

- SB3's continuous-control MLP outputs a Gaussian `(mean, log_std)`. For deployment we take **the mean only**, pass through `tanh` to match training-time squashing, then clip to `[-1, 1]`. Extract `Wout`/`bout` from the policy's **action net mean head** only; ignore the log-std parameter.
- Both `policy.npz` and `policy.h` must apply the **same obs normalization** used at training time (subtract `obs_mean`, divide by `obs_std`). Bake these into `policy.h` as `static const` arrays.
- Firmware integration: add `control/policy_controller.py` (sim-side, Python) and a C `policy_controller.c` that implement `AbstractController` / the firmware's equivalent, consuming sensor readings **normalized identically** to `env/robot_env.py` §3 observation table. Drift in obs formatting between train and deploy is the most common cause of silent failure.

---

## 6. Phase 4 — CV pipeline

### 6.1 Client-side rectification via ARKit (no fiducials)

**Decision.** Rectification happens **on the iPhone using ARKit**, not on the server. The server receives an already-top-down, metric-scaled image. ArUco markers are not used. See [§8.2](#82-screens) for the iOS side.

**What the server receives** from the `/tracks` multipart upload:

- `photo`: the rectified top-down JPEG (already warped by the iOS client to the floor plane).
- `px_per_cm`: float, scale factor used for the rectification (e.g. `10.0`).
- `arkit_confidence`: float in [0, 1], ARKit's plane-tracking confidence at capture time.
- `camera_height_m`: float, ARKit-measured distance from camera to floor plane at capture (used for sanity check).
- `image_bounds_cm`: `[w_cm, h_cm]` of the rectified image in world units.

Store these in `tracks/{track_id}/meta.json` alongside the photo.

### 6.1a Scale sanity check (`cv/validate.py`)

ARKit is usually accurate to ~2%, but drifts on glass floors, low-texture scenes, and fast phone motion. Validate against the known 80 cm plank length:

1. Run the §6.2 segmentation first (we need the detected block rectangles).
2. For each detected block, take the long axis length in cm (using `px_per_cm`).
3. Compute `median_long_cm`.
4. Assert `0.72 <= median_long_cm / 80 <= 1.10` (−10% / +10% band, slightly asymmetric because foreshortening biases low).
5. If outside the band, return error code `SCALE_MISMATCH` with both numbers, prompting the app to request a re-capture. Do not proceed to training.
6. If `< 4` blocks detected, skip the check (report a warning, not an error — the user may have built a minimal course).

The plank length `80.0` cm and the tolerance band live in `config/server.yaml` under `cv.plank_length_cm` and `cv.plank_tolerance`.

**Error cases to surface to the app:**

- `ARKIT_LOW_CONFIDENCE` — `arkit_confidence < 0.5` at capture. (Emitted by the server when it reads the meta.)
- `SCALE_MISMATCH` — plank length check failed; include `expected_cm`, `observed_cm`.
- `IMAGE_TOO_SMALL` — rectified image < 1 MP (warn but allow).

### 6.2 Segmentation (`cv/segment.py`)

**Model.** SAM3 (Meta's third-generation Segment Anything). Pick the smallest variant that holds mask quality on wooden planks — start with the "small"/"base" tier and only escalate if recall < 90% on the test photos. Download on first server boot if missing; pre-cache at server install so the first real request doesn't pay download latency.

**Strategy.** SAM3 supports **text-promptable** segmentation in addition to automatic mask generation. Prefer the text-prompted path for this task — it's much more selective:

```python
# Pseudocode — swap in the real SAM3 API from your access docs.
masks = sam3.segment(
    image=rectified_rgb,
    text_prompt="rectangular wooden plank on the floor",
    # or multiple prompts: ["wooden plank", "wood block"]
    score_threshold=0.5,
)
```

If the text-prompt path is unavailable or underperforms, fall back to SAM3's automatic mask generator with the same tuning philosophy as before (dense grid of point prompts, high stability threshold, reject small regions). Record which path was used in `tracks/{id}/meta.json` so regressions are traceable.

Run on the rectified top-down image (§6.1). Yields ~5–50 candidate masks (text prompt) or ~20–200 (automatic).

**Filtering.**

1. Drop masks whose area in cm² is outside `[50, 2000]` (eliminates single grains and the whole floor).
2. Drop masks whose bounding-box aspect ratio is outside `[1:1, 1:6]` (blocks are roughly rectangular; very thin masks are shadows).
3. Drop masks whose fit residual to a minimum-area rotated rectangle > 15% of the rectangle area (non-rectangular → not a block).
4. Drop masks that touch the image border (likely the arena edge or a hand).
5. Drop masks whose bounding box is within 20 px of the rectified image edge (the user's bounding-box crop in §8.2b should have already excluded off-track area, but double-check).

After filtering, each surviving mask → a rotated rectangle `(cx_cm, cy_cm, w_cm, h_cm, theta_rad)` via `cv2.minAreaRect`.

Output: `list[BlockRect]`.

### 6.3 Blocks → walls (`cv/build_track.py`)

A single block contributes four wall segments (its four edges) as `sim.geometry.Segment`. For race tracks built from rows of adjacent blocks, this produces many collinear short segments; keep them — the ray caster cost scales with segment count but the vectorized version handles hundreds easily, and merging is a source of bugs.

**Centerline extraction.** The interior of the track (the drivable area) is the complement of the blocks within the rectified bounds. Compute:

1. Build a binary mask: everything inside the rectified image (i.e. the full `image_bounds_cm` region the user traced in §8.2b), minus all block masks dilated by `chassis_width / 2` (safety margin).
2. Skeletonize the drivable mask with `skimage.morphology.skeletonize` → a 1-px wide curve.
3. Convert the skeleton to an ordered polyline by walking the largest connected component. Smooth with a Savitzky–Golay filter (window=21). Resample to 200 equispaced points.
4. If the skeleton isn't closed (non-loop course, e.g. time trial), allow both open and closed centerlines. `env/robot_env.py` must handle both (for open courses, `progress` saturates at the end and truncation reward is based on fraction completed).

**Spawn.** First centerline point, heading = tangent direction.

**Output.** A `track` dict identical in shape to §5.1, plus the rectified preview PNG stored on disk.

**Failure modes to surface:**

- `< 3 blocks detected` → `TRACK_TOO_SPARSE`.
- Skeleton has more than one large component → `TRACK_DISCONNECTED`.
- Centerline self-intersects → `TRACK_SELF_INTERSECTING`.

### 6.4 Unit tests

`tests/test_cv.py` with 5 synthetic top-down renders (different track shapes) generated from hand-authored `Segment` lists, rasterized with PIL, then fed through §6.2 and §6.3. Assert block count matches within ±10% and centerline Hausdorff distance < 5 cm.

---

## 7. Phase 5 — FastAPI server

### 7.1 Dependencies (add to `requirements.txt`)

```
fastapi>=0.110
uvicorn[standard]>=0.27
python-jose[cryptography]>=3.3   # JWT
httpx>=0.27                      # APNs
opencv-python>=4.9
scikit-image>=0.22
sam3                             # Meta SAM3 (use access-controlled wheel or local install)
torch>=2.2                       # CUDA build
numba>=0.59                      # optional, for ray casting
aiofiles>=23
pyyaml>=6
```

### 7.2 Filesystem layout (`$PYSAAC_DATA`, default `~/.pysaac/`)

```
$PYSAAC_DATA/
├── tracks/
│   └── {track_id}/
│       ├── photo.jpg
│       ├── rectified.png
│       ├── blocks.json
│       ├── track.json          # the §5.1 dict
│       └── preview.png
├── jobs/
│   └── {job_id}/
│       ├── state.json          # queued|running|done|failed|cancelled
│       ├── progress.jsonl      # one row per rollout
│       ├── stdout.log
│       ├── policy.npz
│       ├── policy.h
│       └── eval.json
└── users/
    └── {apple_sub}.json        # user record
```

IDs are ULIDs. One user = one Apple `sub` claim.

### 7.3 Endpoints

All responses are JSON; all auth-required endpoints require `Authorization: Bearer <JWT>`.

| Method | Path                         | Purpose                                                       |
|--------|------------------------------|---------------------------------------------------------------|
| POST   | `/auth/apple`                | Body: `{identityToken}`. Returns app JWT + expiry.             |
| POST   | `/devices`                   | Body: `{apns_token}`. Registers device for push.               |
| POST   | `/tracks`                    | Multipart upload of photo. Returns `{track_id, status}`. Triggers §6 pipeline async. |
| GET    | `/tracks/{id}`               | Returns CV result: blocks, centerline, preview URL, error codes. |
| PATCH  | `/tracks/{id}`               | Body: edited blocks (user corrections from app). Recomputes centerline. |
| POST   | `/tracks/{id}/confirm`       | Marks track as user-confirmed, required before training.       |
| POST   | `/jobs/train`                | Body: `{track_id, minutes?, n_envs?}`. Returns `{job_id}`. Enqueues fine-tune. |
| GET    | `/jobs/{id}`                 | Returns `state.json` + last progress row.                      |
| WS     | `/jobs/{id}/events`          | Streams progress rows + state transitions. Sends `{"kind":"done","artifact_url":...}` on success. |
| GET    | `/jobs/{id}/artifact`        | Returns `policy.npz` (Content-Type: application/octet-stream). |
| GET    | `/jobs/{id}/artifact.h`      | Returns the C header.                                          |
| DELETE | `/jobs/{id}`                 | Cancels a running job.                                         |

### 7.4 Job queue (`server/jobs.py`)

- **In-process**, single worker — the GPU serializes work anyway.
- `asyncio.Queue[JobID]` for pending jobs.
- One long-lived `asyncio.Task` pops jobs and runs them.
- Training is **blocking CPU+GPU work**. Run it in a dedicated `multiprocessing.Process` (not a thread) so the event loop stays responsive and we can `terminate()` on cancel.
- Parent ↔ child communication via `multiprocessing.Queue` for progress rows, and a `multiprocessing.Event` for stop.
- The async task drains the progress queue into `progress.jsonl` and fans it out to WebSocket subscribers via `server/ws.py`.

Job states, persisted to `state.json` on every transition:

```
queued → running → done
                 → failed   (error field set)
                 → cancelled
```

Queue policy: at most **1 running job per user**; further submissions rejected with 409. Global concurrent jobs = 1 (one GPU).

### 7.5 WebSocket hub (`server/ws.py`)

- `Dict[job_id, Set[WebSocket]]`.
- On connect, replay all rows from `progress.jsonl` so late subscribers catch up.
- Then stream live rows from the progress queue.
- Heartbeat ping every 20 s (Cloudflare idle timeout guard).
- On `done`/`failed`/`cancelled`, send final event and close.

Message schema:

```json
{"kind":"progress","step":123456,"mean_reward":42.1,"fps":12800,"ts":"2026-..."}
{"kind":"state","state":"running"}
{"kind":"done","artifact_url":"/jobs/.../artifact","eval":{...}}
{"kind":"error","code":"OOM","message":"..."}
```

### 7.6 Auth (`server/auth.py`)

**Sign in with Apple, server-side verification.**

1. iOS app performs Sign in with Apple → receives `identityToken` (a JWT signed by Apple).
2. App POSTs token to `/auth/apple`.
3. Server fetches Apple's JWKs from `https://appleid.apple.com/auth/keys` (cache 24 h), verifies signature, verifies `aud == <BUNDLE_ID>`, `iss == https://appleid.apple.com`, and `exp`.
4. Extracts `sub` claim → user ID. Creates `users/{sub}.json` if not present.
5. Issues an **app JWT** (HS256, secret from `$PYSAAC_JWT_SECRET`, 7-day expiry) containing `{sub, exp}`. Returns to app.
6. `Depends(current_user)` dependency validates app JWT on every protected endpoint.

Secrets stored in `.env` (gitignored): `PYSAAC_JWT_SECRET`, `APPLE_BUNDLE_ID`, `APPLE_TEAM_ID`, `APNS_KEY_ID`, `APNS_AUTH_KEY_PATH`.

### 7.7 APNs (`server/apns.py`)

- APNs HTTP/2 endpoint: `https://api.push.apple.com/3/device/{apns_token}`.
- Auth: JWT (ES256) signed with the `.p8` key, `kid=$APNS_KEY_ID`, `iss=$APPLE_TEAM_ID`, 55-min expiry, cached.
- Payload: `{"aps":{"alert":{"title":"Training done","body":"Lap time: 18.3s"},"sound":"default"},"job_id":"..."}`.
- Send on `done`, `failed`, `cancelled`. Best-effort — log failures, do not block the job.
- Use `httpx.AsyncClient(http2=True)`.

### 7.8 Rate limiting

- Use `slowapi` (Flask-Limiter fork for FastAPI).
- `/auth/apple`: 10/min per IP.
- `/tracks`: 20/hour per user.
- `/jobs/train`: 6/hour per user (each job ≈ 10 min, more would starve the GPU).
- WebSocket connection: 10/min per user.

### 7.9 Entrypoint (`server/app.py`)

Standard FastAPI wiring:

```python
app = FastAPI(title="PysaacRC")
app.include_router(auth_router)
app.include_router(tracks_router)
app.include_router(jobs_router)
app.include_router(ws_router)
app.state.job_queue = JobQueue()
@app.on_event("startup") ...  # launch job worker task, warmup CV models
@app.on_event("shutdown") ... # cancel in-flight jobs, flush progress
```

Run with `uvicorn server.app:app --host 127.0.0.1 --port 8787 --proxy-headers`.

### 7.9a Multiprocessing start method

Because training uses `SubprocVecEnv` nested inside a `multiprocessing.Process` spawned by the server, and PyTorch + CUDA is fork-hostile, set the start method **at the very top of `server/app.py` and `training/finetune.py`** before any other imports:

```python
import multiprocessing as mp
mp.set_start_method("spawn", force=True)
```

Also set `SB3`'s `SubprocVecEnv(start_method="spawn")`. Test on your actual desktop OS early — "works on my Mac" does not imply "works on the desktop Linux/Windows box" here.

### 7.9b GPU / upload guards

- At server startup, assert `torch.cuda.is_available()` and log `torch.cuda.get_device_name(0)`; refuse to start otherwise (CPU fine-tune won't hit the budget).
- FastAPI: set the request size limit for `/tracks` to 25 MB (`Starlette`'s `MultiPartParser` accepts `max_file_size`). Phone photos are ~5–10 MB; 25 MB is plenty.
- Explicitly `torch.cuda.empty_cache()` after SAM3 runs, before the training subprocess is spawned, to avoid fragmentation.

### 7.10 Cloudflare Tunnel

`scripts/run_server.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
export PYSAAC_DATA="${PYSAAC_DATA:-$HOME/.pysaac}"
mkdir -p "$PYSAAC_DATA"
cloudflared tunnel run pysaac &
uvicorn server.app:app --host 127.0.0.1 --port 8787
```

`~/.cloudflared/config.yml`:

```yaml
tunnel: <TUNNEL_ID>
credentials-file: /home/<user>/.cloudflared/<TUNNEL_ID>.json
ingress:
  - hostname: api.<your-domain>
    service: http://localhost:8787
    originRequest:
      noTLSVerify: true
      disableChunkedEncoding: false
  - service: http_status:404
```

Cloudflare's WebSocket support is on by default; no extra config needed. Idle timeout is 100 s for the free tier — the 20 s heartbeat in §7.5 handles this.

**One-time tunnel setup** (not already done):

```bash
cloudflared tunnel login              # browser auth, picks the Cloudflare account
cloudflared tunnel create pysaac      # writes the TUNNEL_ID credential JSON
cloudflared tunnel route dns pysaac api.<your-domain>   # DNS CNAME
```

Then populate `config.yml` with the printed `TUNNEL_ID`. `cloudflared tunnel run pysaac` reads `~/.cloudflared/config.yml`.

### 7.11 Tests

- `tests/test_server_auth.py` — mock Apple JWK, verify JWT round-trip.
- `tests/test_server_jobs.py` — submit a fake job (monkeypatch the trainer to sleep), assert state transitions and WS broadcasts.
- `tests/test_server_tracks.py` — upload a synthetic photo, expect a non-error track response.

---

## 8. Phase 6 — iOS app

### 8.1 Project setup

- Xcode project at `ios/PysaacRC/`.
- SwiftUI + async/await URLSession.
- iOS 17 minimum (for Sign in with Apple polish, PhotosPicker, RealityKit 4).
- Frameworks: `ARKit`, `RealityKit`, `AVFoundation`, `AuthenticationServices`, `UserNotifications`.
- Capabilities: Sign in with Apple, Push Notifications, Background Modes (remote-notification), Camera, ARKit (requires A12 Bionic or later — iPhone XS+).
- Dependencies (SwiftPM): **none** — stdlib URLSession does HTTP + WebSocket + multipart. Keep it dependency-free.

### 8.2 Screens

1. **Auth** — "Sign in with Apple" button. On success, POST `/auth/apple`, store JWT in Keychain. Registers for remote notifications; on token receipt, POST `/devices`.
2. **Scan** — ARKit scene (`ARView` / `ARSession` with `ARWorldTrackingConfiguration`, `planeDetection: .horizontal`). User sweeps the phone over the track; the app shows detected plane extents as a translucent overlay. When the plane covers the track for ≥ 2 seconds and tracking state is `.normal`, enable the "Capture" button. Display a live **camera height (m)** readout from the plane anchor so the user can verify scale.
3. **Capture** — on shutter:
   1. Grab the current `ARFrame` (`session.currentFrame`).
   2. Read `frame.camera.intrinsics` (3×3), `frame.camera.transform`, and the selected plane's `transform`.
   3. Compute the camera → floor-plane homography in Swift (see §8.2a below). Target a canvas of `px_per_cm = 10`.
   4. Warp `frame.capturedImage` (CVPixelBuffer → CIImage → CGImage) with the homography. This is the rectified top-down photo.
   5. Crop rectified image to the user's traced bounding box (§8.2b).
   6. Upload via multipart POST to `/tracks` with fields: `photo` (JPEG), `px_per_cm`, `arkit_confidence`, `camera_height_m`, `image_bounds_cm`.
4. **Track confirm** — shows the rectified photo overlaid with detected block rectangles and the proposed centerline, pulled from `GET /tracks/{id}`. User can:
   - Tap a block to delete.
   - Drag to add a block (rotated rectangle gesture).
   - Tap "Regenerate centerline" (PATCH `/tracks/{id}` with edited block list).
   - Tap "Confirm" (POST `/tracks/{id}/confirm`).
   - If the server returned `SCALE_MISMATCH`, show "Re-scan" instead of "Confirm" and bounce back to Scan.
5. **Training** — shows live mean-reward curve from the WebSocket and a log tail. Buttons: "Cancel". On `done`, shows the eval dict.
6. **Artifacts** — downloads `policy.npz` and `policy.h` to the Files app.

### 8.2a Homography from ARKit (Swift)

The transforms ARKit hands you are in a right-handed Y-up world. To rectify to a top-down XZ view of the floor plane:

```swift
// Inputs:
//   K: simd_float3x3  — camera intrinsics (frame.camera.intrinsics)
//   camT: simd_float4x4 — camera pose (frame.camera.transform)
//   planeT: simd_float4x4 — plane anchor transform (floor plane)
//   pxPerMeter: Float — e.g. 1000.0 for 10 px/cm
//
// Output: 3×3 homography H that maps image pixels (u, v, 1) to (x_world_cm, z_world_cm, 1)
//
// Method:
//   1. Build the plane's basis in world: e_x = planeT.columns.0.xyz, e_z = planeT.columns.2.xyz, o = planeT.columns.3.xyz.
//   2. For 4 corners of a rectangle on the plane centered at o (meters), project to pixel coords:
//      P_cam = inverse(camT) * [p_world, 1]
//      p_pix = K * (P_cam.xyz / P_cam.z)
//   3. Solve the 4-point homography: cv2.getPerspectiveTransform(pixCorners, planeCornersPx) — do this in Accelerate/simd or call into OpenCV-iOS (carry a thin wrapper). Using a pure-Swift 4-point solver (~40 lines) avoids the OpenCV-iOS dependency.
//   4. Warp with vImage or Metal. For MVP, use CIFilter.perspectiveCorrection with the 4 corner points (no custom matrix math needed — Core Image accepts the 4 pixel corners directly and does the warp).
```

Simplest MVP path: skip the explicit homography matrix. Use **`CIFilter.perspectiveCorrection`** and feed it the four pixel coordinates corresponding to a known-size rectangle on the floor plane (say 2 m × 1.5 m centered on the plane origin). Core Image does the warp. Scale = output size / 2 m.

### 8.2b Bounding box picker

After Scan, before Capture, the user drags a rectangle on the AR preview to indicate the track's extent on the floor plane. This rectangle (in plane coordinates) becomes the `image_bounds_cm`. Without this, the rectified image is arbitrary-sized and may include irrelevant floor area that degrades SAM3.

### 8.2c ARKit availability gate

At app launch, check `ARWorldTrackingConfiguration.isSupported`. On failure (iPhone older than XS, unlikely), show a graceful "This device is not supported" screen and exit. Do not implement a non-AR fallback in v1.

### 8.3 Networking

- `APIClient` actor with methods 1:1 to the server endpoints.
- WebSocket via `URLSessionWebSocketTask`. Reconnect with exponential backoff on transient disconnects; since the server replays from `progress.jsonl` on connect, reconnects are safe.
- All requests attach `Authorization: Bearer <JWT>`; on 401, try `/auth/apple` refresh once, then bounce to login screen.

### 8.4 Push notifications

- Register with `UIApplication.shared.registerForRemoteNotifications()`.
- In `application(_:didRegisterForRemoteNotificationsWithDeviceToken:)`, POST the token to `/devices`.
- Handle `didReceiveRemoteNotification` with `job_id` → deep-link to the job screen.

### 8.5 Apple Developer setup (one-time)

This has to happen **before** writing server auth or the iOS app. All three pieces use the same Bundle ID.

1. In developer.apple.com → Certificates, Identifiers & Profiles → Identifiers → `+`, App ID, Bundle ID `com.<you>.pysaacrc`. Enable capabilities: **Sign In with Apple**, **Push Notifications**.
2. Keys → `+` → **Apple Push Notifications service (APNs)** → download the `.p8`. Copy the Key ID. Copy your Team ID (top-right of the page). Store the `.p8` at `$APNS_AUTH_KEY_PATH` on the desktop.
3. For Sign in with Apple: no extra key needed for server-side identity-token verification — Apple's public JWKs are fetched from `https://appleid.apple.com/auth/keys`. Just make sure the `aud` in the token matches the Bundle ID.
4. Xcode project: under Signing & Capabilities, add Sign in with Apple and Push Notifications. Set the Bundle ID to the one registered in step 1.

Values required in `.env`: `APPLE_BUNDLE_ID`, `APPLE_TEAM_ID`, `APNS_KEY_ID`, `APNS_AUTH_KEY_PATH`.

### 8.6 App Store Connect / TestFlight

Out of scope for an internal tool. Distribute to self via Xcode "Run on device" or to one beta tester via TestFlight internal testing. No review needed for internal testing.

---

## 9. Phase 7 — End-to-end timing budget

| Step                                  | Target | Notes                                                          |
|---------------------------------------|--------|----------------------------------------------------------------|
| Photo upload (4 MB over LTE)          |  3 s   |                                                                |
| ARKit capture + rectify (client-side) |  1 s   | Happens on the phone, not the server                           |
| Scale sanity check                    |  0.1 s |                                                                |
| SAM3 segmentation (tiny, 3060 Ti)     |  4 s   | GPU warmed up at server boot                                   |
| Filter + fit rectangles + centerline  |  1 s   |                                                                |
| User confirm UI (interactive)         |  0 s   | Does not count toward the 10 min budget                        |
| Fine-tune 2 M steps (8 envs)          |  3 min | @ ~11 k sps after vectorization                                |
| Eval 20 episodes                      | 30 s   |                                                                |
| Export + notify                       |  5 s   |                                                                |
| **Total (hands-off portion)**         | ~4 min | Leaves 6 min headroom                                          |

If this slips, the first knob is `total_timesteps` (drop 2M → 1M); the second is `n_envs` (raise to 12 and retest SMT).

---

## 10. Phase 8 — Testing and rollout

Build in this order. Each step must be green before starting the next.

1. `env/robot_env.py` + smoke test. Validate reward curve trends up with a 10k-step PPO toy run on the default oval.
2. `sim/geometry_np.py` + fuzz test + wiring into `sample_sensors`. Profile env step before/after with `cProfile` — expect ≥ 3× speedup on the oval.
3. `env/track_gen.py` + 20 rendered PNGs for visual sanity.
4. `training/base_policy.py` — launch overnight, check TensorBoard at 1M and 10M.
5. `training/finetune.py` — with a mocked base ckpt, verify 2M steps in < 4 min.
6. `cv/` — end-to-end on a real phone photo of a hand-built track (do this offline before wiring to the server).
7. `server/` — local-only first (`--host 127.0.0.1`). Use `curl` + `websocat` to exercise every endpoint.
8. Cloudflare Tunnel exposure + reach server from phone on LTE (not WiFi — different path). Verify WebSocket survives 8 min idle-ish with heartbeats.
9. iOS app, screen by screen. Each screen gated by the server endpoint it calls working.
10. First full end-to-end run with a real track. Measure all steps in §9 and update the budget table.

---

## 11. Risks and mitigations

| Risk                                                           | Mitigation                                                        |
|----------------------------------------------------------------|-------------------------------------------------------------------|
| CV fails on an unseen track (novel lighting, glare)            | Mandatory user confirmation screen; let user delete/add blocks.    |
| 10-min budget blown                                            | Fine-tune from base policy; fallback to shorter `total_timesteps`. |
| Base policy doesn't transfer to a weird new track shape        | Wider pre-training distribution; user can request `minutes=10`.   |
| Public endpoint abused                                         | Sign in with Apple + rate limits + per-user job cap.               |
| Cloudflare idle-timeout kills long jobs                        | WebSocket heartbeat every 20 s; also jobs run async of any request.|
| GPU OOM during SAM3 + training overlap                         | Serialize: CV step releases GPU before training starts (explicit `torch.cuda.empty_cache()`). |
| Sim-to-real gap from CV-reconstructed geometry ≠ real geometry | 10 % Gaussian jitter on block center/size during fine-tune DR.     |
| Training crash loses progress                                  | Checkpoint fine-tune every 200k steps; on resume, reload latest.   |
| iOS app can't keep WebSocket open backgrounded                 | APNs push is the source of truth for "done"; WS is UX-only.        |

---

## 12. Configuration reference

`config/server.yaml` (new):

```yaml
data_dir: ~/.pysaac
host: 127.0.0.1
port: 8787
jwt_secret_env: PYSAAC_JWT_SECRET
jwt_ttl_days: 7
apple_bundle_id: com.example.pysaacrc
apns:
  key_id_env: APNS_KEY_ID
  team_id_env: APPLE_TEAM_ID
  auth_key_path_env: APNS_AUTH_KEY_PATH
  bundle_id: com.example.pysaacrc
  use_production: false
rate_limits:
  auth: "10/minute"
  tracks: "20/hour"
  jobs:   "6/hour"
training:
  base_policy_path: models/base_policy.zip
  default_timesteps: 2_000_000
  n_envs: 8
  max_minutes: 10
cv:
  sam3_weights: models/sam3.pt
  sam3_mode: text_prompt            # text_prompt | automatic
  sam3_text_prompts:
    - "rectangular wooden plank on the floor"
    - "wood block"
  sam3_score_threshold: 0.5
  plank_length_cm: 80.0
  plank_tolerance:
    lower: 0.72    # observed / expected — asymmetric because foreshortening biases low
    upper: 1.10
  min_arkit_confidence: 0.5
  target_px_per_cm: 10.0
```

Every piece of code reads from this file; no magic numbers scattered in source.

---

## 13. Definition of done

- [ ] `env/robot_env.py` passes smoke test; a random 10k-step PPO run shows non-trivial reward.
- [ ] Vectorized ray casting yields ≥ 3× env throughput.
- [ ] `models/base_policy.zip` meets the §5.3 eval gate.
- [ ] `cv/` produces correct track on ≥ 5 real phone photos across 2 lighting conditions.
- [ ] `server/` passes all tests; all endpoints documented and reachable over Cloudflare Tunnel.
- [ ] iOS app performs the full flow end-to-end from cold launch in ≤ 10 min wall-clock (excluding user confirm time).
- [ ] Exported `policy.npz` runs in the existing Python eval path; exported `policy.h` compiles alongside `hal/` with no warnings.
- [ ] One successful run with the weights flashed to the real car on a real track.

---

## 14. Gotchas a builder will actually hit

Collected here so nobody has to rediscover them.

1. **Obs-format drift between train and deploy silently wrecks transfer.** The firmware's sensor readings must be normalized identically to the §3 observation table — same units, same clipping, same frame-stack order (oldest→newest), same `last_action` in [0, 1] shifted form. Write the normalization once, in one place, and import it from both the env and the C header generator.
2. **Action mapping signs.** `SERVO_CENTER_COUNT + action[1] * (SERVO_MAX_COUNT - SERVO_CENTER_COUNT)`: `action[1] = +1` gives `SERVO_MAX_COUNT`, which (per `servo_count_to_steer_angle`) means **negative** steer angle, which is a right turn. Be deliberate about this convention when wiring to the real car; mirror it in the C header if needed.
3. **`SubprocVecEnv` + CUDA on macOS defaults to `fork` and will hang.** Always pass `start_method="spawn"`.
4. **SB3 env-per-subprocess `pickle` gotcha.** The `make_env` closure you pass to `SubprocVecEnv` must be picklable under `spawn`. Use a top-level factory function, not a lambda capturing a track dict. Pass the track via env-var or file path and re-load inside the subprocess.
5. **Monitor wrapper.** Without `stable_baselines3.common.monitor.Monitor` around each env (or `VecMonitor` around the vec), `ep_info_buffer` stays empty and mean reward reads 0. The existing GUI worker already gets this right ([gui/training/worker.py](../gui/training/worker.py)) — copy that pattern.
6. **ARKit plane tracking drifts on textureless floors and glass.** The app should gate the shutter on `arkit_confidence ≥ 0.5` and `tracking state == .normal`, and enforce a minimum scan sweep duration (2 s) before enabling capture. The server-side scale sanity check in §6.1a is the backstop when the client gates fail.
7. **SAM3 weights may be access-gated.** Confirm the download path before deploying; pre-cache on the desktop so the first user request doesn't block on auth or network. Record the exact model version in `tracks/{id}/meta.json` for reproducibility.
8. **FastAPI + `multiprocessing.Process` + PyTorch:** the child process must re-import torch after `spawn`. Do any `torch.set_num_threads(1)` / env var setup at the top of the child entrypoint, not in the parent.
9. **Cloudflare Tunnel does not rewrite the `Host` header for WebSocket upgrade** unless `originRequest.disableChunkedEncoding: false` is set (it is, in the config above). If WS silently 404s, that's the first place to look.
10. **"Minutes" from the iOS app → `total_timesteps` on the server.** Map `minutes=3` to ~1.5M steps, `minutes=8` to ~4M, using a lookup table rather than a linear formula — throughput varies with track complexity.
11. **Eval gate uses the same sim as training.** Passing the gate ≠ working on the real car. Treat eval numbers as a smoke test, not a guarantee.
12. **Progress reward wrap-around.** On the first step after the robot crosses the start/finish line, the arclength projection jumps by `-total_length`. Use the arc-pointer pseudocode in §3 to prevent this (search within a sliding window around the last known idx).
13. **Do not commit `.env`, `models/`, `$PYSAAC_DATA`, the `.p8` key, or the Cloudflare credentials JSON.** Add to `.gitignore` before the first commit.
14. **Time estimates in §9 are for the steady state.** First request after server boot pays SAM3 lazy-init (~8 s) and CUDA kernel cache (~3 s). Warm them both at startup so the user never sees cold latency.
15. **Progress WS reconnection race.** A client that reconnects between the final `progress` row and the `done` event could miss `done`. Always persist the terminal state to `state.json` before closing WS, and have the client GET `/jobs/{id}` on reconnect to catch up.

---

## 15. Required reads before touching a section

| Section you're implementing | Read first                                                                 |
|-----------------------------|----------------------------------------------------------------------------|
| §3 RobotEnv                 | `sim/physics.py`, `sim/sensors.py`, `sim/calibration.py`, `sim/constants.py` |
| §4 Vectorized rays          | `sim/geometry.py`                                                          |
| §5 Base policy              | `examples/train_ppo.py`, `gui/training/worker.py` (for SB3 patterns)       |
| §5.6 Export                 | `gui/training/linear_policy.py` (weight extraction reference)              |
| §6 CV                       | `sim/world.py` (to understand the Segment format you're producing)         |
| §7 Server                   | — (greenfield)                                                              |
| §8 iOS                      | Apple: "Sign in with Apple REST API" docs; "Sending Push Notifications Using On-Demand Reference Docs" |
