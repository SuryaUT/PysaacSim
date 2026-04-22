"""End-to-end sim↔real calibration pipeline.

Reads:
    * one `IR_Calib.xlsx` (IR analog front-end constants per side)
    * one motors-on stationary CSV (for IMU bias + noise std)
    * N driving CSVs (at least one; ideally 4) for dynamics fitting

Writes (to `--out-dir`, defaults to PySaacSim/artifacts/calib_<timestamp>):
    * `summary.txt`     — human-readable diff + fit stats
    * `overlays/*.png`  — per-log sim-vs-real overlay plots (if matplotlib)
    * `proposed.yaml`   — the proposed calibration (copy of current + fits)
    * `diff.txt`        — yaml diff current vs proposed
    * `dynamics.json`   — optimizer transcript (best x, loss, per-log loss)

Nothing is applied to PySaacSim/config/calibration.yaml unless `--apply` is
passed, in which case the script writes to that path after prompting.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


# Ensure the repo's top-level package imports work when this script is run
# as `python scripts/calibrate_from_log.py ...` from anywhere.
sys.path.insert(0, str(_repo_root().parent))


from PySaacSim.sim.calibration import (  # noqa: E402
    SensorCalibration, IRCalibration, IRSideCalibration,
)
from PySaacSim.calib.log_io import load_log, load_logs  # noqa: E402
from PySaacSim.calib.ir_xlsx import fit_xlsx as fit_ir_xlsx  # noqa: E402
from PySaacSim.calib.imu_bias import estimate_bias  # noqa: E402
from PySaacSim.calib.noise_fit import fit_noise  # noqa: E402
from PySaacSim.calib.latency import estimate as estimate_latency  # noqa: E402
from PySaacSim.calib.dynamics_fit import fit_dynamics, DEFAULT_X0, PARAM_NAMES  # noqa: E402
from PySaacSim.calib import report as report_mod  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ir-xlsx", type=Path, required=True)
    ap.add_argument("--steady-csv", type=Path, required=True,
                    help="motors-on stationary log (for IMU bias + noise)")
    ap.add_argument("--drive-csvs", type=Path, nargs="+", required=True,
                    help="driving logs (>=1; hold one out for validation)")
    ap.add_argument("--holdout-csv", type=Path, default=None,
                    help="optional validation log (default: last of --drive-csvs)")
    ap.add_argument("--current-yaml", type=Path,
                    default=_repo_root() / "config" / "calibration.yaml")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--skip-dynamics", action="store_true",
                    help="skip the CMA-ES fit (useful for sensor-only passes)")
    ap.add_argument("--max-evals", type=int, default=150)
    ap.add_argument("--apply", action="store_true",
                    help="overwrite --current-yaml with the proposed values"
                         " (prompts for confirmation)")
    args = ap.parse_args(argv)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (_repo_root() / "artifacts" / f"calib_{stamp}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    current = SensorCalibration.from_yaml(args.current_yaml)

    # --- Phase 2a: IR xlsx per-side fit -------------------------------------
    print("[1/5] Fitting IR formula per side from xlsx...")
    ir_fit = fit_ir_xlsx(args.ir_xlsx)

    # --- Phase 2b: IMU bias from steady capture -----------------------------
    print("[2/5] IMU bias + noise from steady capture...")
    steady = load_log(args.steady_csv)
    bias = estimate_bias(steady)
    noise = fit_noise(steady, current.ir)

    # --- Phase 3: latency cross-correlation across drive logs ---------------
    print("[3/5] Latency identification...")
    drive_paths = list(args.drive_csvs)
    drive_logs = load_logs(drive_paths)
    latency = estimate_latency(drive_logs)

    # --- Phase 4: CMA-ES dynamics fit (optional) ----------------------------
    dyn_result = None
    if not args.skip_dynamics:
        print(f"[4/5] CMA-ES dynamics fit (max_evals={args.max_evals})...")
        if args.holdout_csv is not None:
            holdout = load_log(args.holdout_csv)
            train = drive_logs
        elif len(drive_logs) > 1:
            holdout = drive_logs[-1]
            train = drive_logs[:-1]
        else:
            holdout = None
            train = drive_logs
        # Apply the new IR and IMU bias to an intermediate cal so replay sees
        # the right IMU LPF/avg knobs from the start.
        interim = report_mod.apply_to_calibration(
            current, ir_fit=ir_fit, noise=noise, imu_bias=bias,
        )
        # Use the IMU noise std from the steady capture to whiten the loss.
        sigma = (
            max(bias.gyro_std_lsb, 1.0),
            max(bias.accel_x_std_lsb, 1.0),
            max(bias.accel_y_std_lsb, 1.0),
        )
        bias_tuple = (
            bias.gyro_bias[2],           # gyro_z bias only (log has no x/y gyro)
            bias.accel_bias[0],          # accel_x bias
            bias.accel_bias[1],          # accel_y bias
        )
        dyn_result = fit_dynamics(
            train_logs=train,
            imu_cal=interim.imu,
            sigma=sigma,
            bias=bias_tuple,
            max_evals=args.max_evals,
            holdout_log=holdout,
            verbose=True,
        )
    else:
        print("[4/5] --skip-dynamics: skipping CMA-ES fit.")

    # --- Phase 5: artifacts + proposed yaml ---------------------------------
    print("[5/5] Writing artifacts...")
    proposed = report_mod.apply_to_calibration(
        current, ir_fit=ir_fit, noise=noise, imu_bias=bias,
    )
    proposed.to_yaml(out_dir / "proposed.yaml")

    diff_text = report_mod.yaml_diff(current, proposed)
    (out_dir / "diff.txt").write_text(diff_text + "\n")

    lines = report_mod.summary_lines(ir_fit, noise, bias, latency, dyn_result)
    summary_path = out_dir / "summary.txt"
    summary_path.write_text("\n".join(lines) + "\n\n--- yaml diff ---\n" + diff_text + "\n")

    if dyn_result is not None:
        dyn_json = {
            "param_names": list(PARAM_NAMES),
            "x": dyn_result.x.tolist(),
            "loss_train": dyn_result.loss,
            "per_log_loss": dyn_result.per_log_loss,
            "n_evals": dyn_result.n_evals,
        }
        if dyn_result.holdout_loss is not None:
            dyn_json["loss_holdout"] = dyn_result.holdout_loss
        if dyn_result.cov is not None:
            dyn_json["cov_diag"] = np.diag(dyn_result.cov).tolist()
        (out_dir / "dynamics.json").write_text(json.dumps(dyn_json, indent=2))

        overlays_dir = out_dir / "overlays"
        written = report_mod.plot_overlays(
            drive_logs, dyn_result, proposed.imu,
            bias=(bias.gyro_bias[2], bias.accel_bias[0], bias.accel_bias[1]),
            out_dir=overlays_dir,
        )
        print(f"  wrote {len(written)} overlay(s) to {overlays_dir}")

    print("\n=== summary ===")
    for l in lines:
        print(l)
    print("\n=== yaml diff ===")
    print(diff_text)
    print(f"\nArtifacts in: {out_dir}")

    if args.apply:
        resp = input(
            f"\n--apply given: overwrite {args.current_yaml} with the proposed"
            f" values above? [y/N] "
        ).strip().lower()
        if resp == "y":
            proposed.to_yaml(args.current_yaml)
            print(f"wrote {args.current_yaml}")
        else:
            print("aborted; nothing written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
