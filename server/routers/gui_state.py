from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, Depends, Request, Response

from ..auth import User, current_user
from ..schemas import (
    ControllerCompileBody, ControllerCompileResponse,
    RobotSpawnBody, RobotPatchBody, PlaybackBody, LiveTrainingStartBody
)
from ...sim.geometry import Segment, Vec2
from ...sim.calibration import SensorCalibration
from ...sim.state import RobotDims
from ...gui.pages.controller_editor import _STARTER, _QUICK_PASTE_PROMPT

router = APIRouter(prefix="/gui", tags=["gui"])

@router.get("/track")
async def get_track(request: Request, user: User = Depends(current_user)) -> dict:
    runner = request.app.state.sim_runner
    return runner.snapshot()

@router.post("/track")
async def post_track(body: dict, request: Request, user: User = Depends(current_user)) -> dict:
    runner = request.app.state.sim_runner
    await runner.load_track(body)
    return {"status": "ok"}

@router.get("/dims")
async def get_dims(request: Request, user: User = Depends(current_user)) -> dict:
    runner = request.app.state.sim_runner
    return asdict(runner._dims)

@router.post("/dims")
async def post_dims(body: dict, request: Request, user: User = Depends(current_user)) -> dict:
    runner = request.app.state.sim_runner
    runner._dims = RobotDims(**body)
    return {"status": "ok"}

@router.get("/calibration")
async def get_calibration(request: Request, user: User = Depends(current_user)) -> dict:
    runner = request.app.state.sim_runner
    return runner._cal.to_dict() if hasattr(runner._cal, "to_dict") else asdict(runner._cal)

@router.post("/calibration")
async def post_calibration(body: dict, request: Request, user: User = Depends(current_user)) -> dict:
    runner = request.app.state.sim_runner
    if hasattr(SensorCalibration, "from_dict"):
        runner._cal = SensorCalibration.from_dict(body)
    else:
        import tempfile
        import yaml
        from pathlib import Path
        tmp = Path(tempfile.mkstemp(suffix=".yaml")[1])
        tmp.write_text(yaml.safe_dump(body))
        runner._cal = SensorCalibration.from_yaml(tmp)
        tmp.unlink()
    return {"status": "ok"}

@router.post("/track/reset")
async def reset_track(request: Request, user: User = Depends(current_user)) -> dict:
    runner = request.app.state.sim_runner
    from ...sim.world import build_default_world
    runner._walls = build_default_world()["walls"]
    return {"status": "ok"}

@router.post("/dims/reset")
async def reset_dims(request: Request, user: User = Depends(current_user)) -> dict:
    runner = request.app.state.sim_runner
    runner._dims = RobotDims.default()
    return {"status": "ok"}

@router.post("/calibration/reset")
async def reset_calibration(request: Request, user: User = Depends(current_user)) -> dict:
    runner = request.app.state.sim_runner
    runner._cal = SensorCalibration.default()
    return {"status": "ok"}

# --- Playback ---

@router.post("/playback")
async def post_playback(body: PlaybackBody, request: Request, user: User = Depends(current_user)) -> dict:
    runner = request.app.state.sim_runner
    async with runner._lock:
        if body.time_scale is not None:
            runner.engine.time_scale = body.time_scale
        if body.cars_interact is not None:
            runner.engine.cars_interact = body.cars_interact
        if body.auto_respawn is not None:
            runner.engine.auto_respawn = body.auto_respawn
        if body.playing is not None:
            runner.playing = body.playing
    return {"status": "ok"}

@router.post("/playback/reset")
async def post_playback_reset(request: Request, user: User = Depends(current_user)) -> dict:
    runner = request.app.state.sim_runner
    await runner.reset()
    return {"status": "ok"}

# --- Live Training ---

@router.post("/training/start")
async def start_training(body: LiveTrainingStartBody, request: Request, user: User = Depends(current_user)) -> dict:
    runner = request.app.state.sim_runner
    runner.start_live_training(
        total_timesteps=body.total_timesteps,
        lr=body.learning_rate,
        device=body.device,
        n_envs=body.n_envs,
        same_scene=body.same_scene,
        save_path=body.save_path,
        resume=body.resume,
        save_every=body.save_every
    )
    return {"status": "ok"}

@router.post("/training/stop")
async def stop_training(request: Request, user: User = Depends(current_user)) -> dict:
    runner = request.app.state.sim_runner
    runner.stop_live_training()
    return {"status": "ok"}

# --- Robots CRUD ---

@router.get("/robots")
async def get_robots(request: Request, user: User = Depends(current_user)) -> list[dict]:
    runner = request.app.state.sim_runner
    return [asdict(r) for r in runner.robots]

@router.post("/robots")
async def spawn_robot(body: RobotSpawnBody, request: Request, user: User = Depends(current_user)) -> dict:
    runner = request.app.state.sim_runner
    async with runner._lock:
        spec = runner.add_robot(body.x, body.y, body.theta, body.controller_id)
        # Immediately snap its runtime pose
        runner.engine.force_robot_pose(spec.id, spec.x, spec.y, spec.theta)
    return {"status": "ok", "robot": asdict(spec)}

@router.patch("/robots/{robot_id}")
async def patch_robot(robot_id: int, body: RobotPatchBody, request: Request, user: User = Depends(current_user)) -> dict:
    runner = request.app.state.sim_runner
    async with runner._lock:
        update_dict = {k: v for k, v in body.model_dump().items() if v is not None}
        runner.update_robot(robot_id, **update_dict)
        # If pose changed, snap the physics engine
        if "x" in update_dict or "y" in update_dict or "theta" in update_dict:
            spec = next((r for r in runner.robots if r.id == robot_id), None)
            if spec:
                runner.engine.force_robot_pose(spec.id, spec.x, spec.y, spec.theta)
    return {"status": "ok"}

@router.delete("/robots/{robot_id}")
async def delete_robot(robot_id: int, request: Request, user: User = Depends(current_user)) -> dict:
    runner = request.app.state.sim_runner
    async with runner._lock:
        runner.remove_robot(robot_id)
    return {"status": "ok"}

@router.post("/robots/clear")
async def clear_robots(request: Request, user: User = Depends(current_user)) -> dict:
    runner = request.app.state.sim_runner
    async with runner._lock:
        runner.clear_robots()
    return {"status": "ok"}

# --- Controllers ---

@router.get("/controller_templates")
async def get_controller_templates(user: User = Depends(current_user)) -> dict:
    from pathlib import Path
    example_path = Path(__file__).resolve().parents[3] / "examples" / "controller_template.c"
    example_code = example_path.read_text() if example_path.exists() else ""
    return {
        "starter": _STARTER,
        "quick_paste": _QUICK_PASTE_PROMPT,
        "example": example_code
    }

@router.post("/compile_controller", response_model=ControllerCompileResponse)
async def compile_controller(
    body: ControllerCompileBody,
    request: Request,
    user: User = Depends(current_user),
) -> ControllerCompileResponse:
    sim_runner = request.app.state.sim_runner
    result = await sim_runner.load_c_controller(body.name, body.code)
    if result == "ok":
        return ControllerCompileResponse(status="ok")
    else:
        return ControllerCompileResponse(status="error", error=result)

@router.delete("/controllers/{name}")
async def remove_controller(name: str, request: Request, user: User = Depends(current_user)) -> dict:
    sim_runner = request.app.state.sim_runner
    await sim_runner.unload_controller(name)
    return {"status": "ok"}

# --- YAML File Export ---

@router.get("/export/track.yaml")
async def export_track(request: Request, user: User = Depends(current_user)):
    runner = request.app.state.sim_runner
    data = {"walls": [{"ax": w.a.x, "ay": w.a.y, "bx": w.b.x, "by": w.b.y} for w in runner._walls]}
    import yaml
    yaml_str = yaml.safe_dump(data, sort_keys=False)
    return Response(content=yaml_str, media_type="text/yaml", headers={
        "Content-Disposition": 'attachment; filename="track.yaml"'
    })

@router.get("/export/dims.yaml")
async def export_dims(request: Request, user: User = Depends(current_user)):
    runner = request.app.state.sim_runner
    import yaml
    yaml_str = yaml.safe_dump(asdict(runner._dims), sort_keys=False)
    return Response(content=yaml_str, media_type="text/yaml", headers={
        "Content-Disposition": 'attachment; filename="dims.yaml"'
    })

@router.get("/export/calibration.yaml")
async def export_calibration(request: Request, user: User = Depends(current_user)):
    runner = request.app.state.sim_runner
    import yaml
    cal_dict = runner._cal.to_dict() if hasattr(runner._cal, "to_dict") else asdict(runner._cal)
    yaml_str = yaml.safe_dump(cal_dict, sort_keys=False)
    return Response(content=yaml_str, media_type="text/yaml", headers={
        "Content-Disposition": 'attachment; filename="calibration.yaml"'
    })

from fastapi import UploadFile, File
import tempfile
from pathlib import Path

@router.post("/import/track")
async def import_track(request: Request, file: UploadFile = File(...), user: User = Depends(current_user)):
    runner = request.app.state.sim_runner
    tmp = Path(tempfile.mkstemp(suffix=".yaml")[1])
    tmp.write_bytes(await file.read())
    from ...gui.persistence import load_track
    try:
        walls = load_track(tmp)
        runner._walls = walls
    finally:
        tmp.unlink()
    return {"status": "ok"}

@router.post("/import/dims")
async def import_dims(request: Request, file: UploadFile = File(...), user: User = Depends(current_user)):
    runner = request.app.state.sim_runner
    tmp = Path(tempfile.mkstemp(suffix=".yaml")[1])
    tmp.write_bytes(await file.read())
    from ...gui.persistence import load_dims
    try:
        runner._dims = load_dims(tmp)
    finally:
        tmp.unlink()
    return {"status": "ok"}

@router.post("/import/calibration")
async def import_calibration(request: Request, file: UploadFile = File(...), user: User = Depends(current_user)):
    runner = request.app.state.sim_runner
    tmp = Path(tempfile.mkstemp(suffix=".yaml")[1])
    tmp.write_bytes(await file.read())
    from ...gui.persistence import load_calibration
    try:
        runner._cal = load_calibration(tmp)
    finally:
        tmp.unlink()
    return {"status": "ok"}

