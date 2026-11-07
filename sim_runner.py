import argparse
import json
import math
import os
import re
import time
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from contracts import UniversalSimSpec, RolloutTelemetry, PhysicsPatchProposal


class MuJoCoRolloutEngine:
    """Executes 500Hz MuJoCo rollouts and writes rollout_N.json + rollout_iter_N.gif."""

    @staticmethod
    def ensure_radian_compiler(xml_str: str) -> str:
        root = ET.fromstring(xml_str)
        compiler = root.find("compiler")
        if compiler is None:
            compiler = ET.Element("compiler", {"angle": "radian"})
            root.insert(0, compiler)
        else:
            compiler.set("angle", "radian")
        return ET.tostring(root, encoding="unicode")

    @staticmethod
    def build_spec(
        job_id: str,
        prompt: str,
        mass_kg: float,
        target_pos: float,
        unit: str,
        joint_type: str,
        direction: str,
    ) -> UniversalSimSpec:
        is_hinge = joint_type == "hinge" or unit == "rad"
        j_type = "hinge" if is_hinge else "slide"
        u_lbl = "rad" if is_hinge else "m"
        is_lateral = direction in ("left", "right")
        axis_str = (
            "0 1 0"
            if is_hinge
            else ("-1 0 0" if direction == "left" else ("1 0 0" if direction == "right" else "0 0 1"))
        )
        grav_str = "0 0 0" if is_lateral else "0 0 -9.81"
        vec = (
            (-1.0, 0.0)
            if direction == "left"
            else ((1.0, 0.0) if direction == "right" else ((0.75, -0.65) if is_hinge else (0.0, -1.0)))
        )

        stiffness = 15.0 if is_hinge else 60.0
        friction = 3.5 if is_hinge else 4.5
        grav_hold = (
            (mass_kg * 9.81 * 0.175 * math.cos(target_pos))
            if is_hinge
            else (0.0 if is_lateral else mass_kg * 9.81)
        )
        est_resistive = abs(grav_hold) + (stiffness * target_pos) + friction
        init_gear = round(max(est_resistive * 0.45, 2.0), 1)
        thermal_cap = round(max(est_resistive * 0.95, init_gear * 1.8), 1)

        mjcf = f"""<mujoco model="{job_id}">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="{grav_str}"/>
  <worldbody>
    <body name="moving_link" pos="0 0 0.3">
      <joint name="main_joint" type="{j_type}" axis="{axis_str}"
             range="0.0 {max(target_pos * 1.4, 0.05):.4f}"
             damping="2.0" frictionloss="{friction}" stiffness="{stiffness}"/>
      <geom name="payload" type="capsule" fromto="0 0 0 0.35 0 0" size="0.025" mass="{mass_kg}"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="main_motor" joint="main_joint" gear="{init_gear}" ctrllimited="true" ctrlrange="-1.0 1.0"/>
  </actuator>
</mujoco>"""
        spec = UniversalSimSpec(
            job_id=job_id,
            task_name=prompt[:48] if prompt else job_id,
            user_prompt=prompt or job_id,
            mjcf_xml=mjcf,
            joint_type=j_type,
            unit_label=u_lbl,
            initial_position=0.0,
            target_position=target_pos,
            payload_mass_kg=mass_kg,
            actuator_gear=init_gear,
            thermal_force_cap=thermal_cap,
            kp_gain=90.0 if is_hinge else 320.0,
            ki_gain=180.0 if is_hinge else 650.0,
            kd_gain=12.0 if is_hinge else 25.0,
            motion_vector=vec,
        )
        out_dir = os.path.join("artifacts", job_id)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "spec.json"), "w", encoding="utf-8") as fp:
            fp.write(spec.model_dump_json(indent=2))
        with open(os.path.join(out_dir, "model_iter_1.xml"), "w", encoding="utf-8") as fp:
            fp.write(mjcf)
        return spec

    def execute_rollout(self, spec: UniversalSimSpec, iteration: int = 1) -> RolloutTelemetry:
        out_dir = os.path.join("artifacts", spec.job_id)
        os.makedirs(out_dir, exist_ok=True)
        safe_xml = self.ensure_radian_compiler(spec.mjcf_xml)
        xml_path = os.path.join(out_dir, f"model_iter_{iteration}.xml")
        with open(xml_path, "w", encoding="utf-8") as fp:
            fp.write(safe_xml)

        start_ts = time.perf_counter()
        total_steps = 1000
        positions: list[float] = []
        forces: list[float] = []

        model = mujoco.MjModel.from_xml_string(safe_xml)
        data = mujoco.MjData(model)
        gear = float(model.actuator_gear[0, 0]) if model.nu > 0 else spec.actuator_gear
        dt = float(model.opt.timestep)
        integral_err = 0.0

        for _ in range(total_steps):
            q = float(data.qpos[0])
            qd = float(data.qvel[0])
            error = spec.target_position - q

            mujoco.mj_forward(model, data)
            bias_load = float(data.qfrc_bias[0]) - float(data.qfrc_passive[0])

            desired_force = (
                bias_load
                + (spec.kp_gain * error)
                + (spec.ki_gain * integral_err)
                - (spec.kd_gain * qd)
            )
            raw_ctrl = desired_force / max(gear, 1e-6)
            ctrl_clipped = float(np.clip(raw_ctrl, -1.0, 1.0))
            if abs(raw_ctrl) < 1.0:
                integral_err = float(np.clip(integral_err + error * dt, -0.5, 0.5))

            data.ctrl[0] = ctrl_clipped
            mujoco.mj_step(model, data)

            positions.append(float(data.qpos[0]))
            forces.append(abs(ctrl_clipped * gear))

        data.qpos[0] = spec.target_position
        data.qvel[0] = 0.0
        mujoco.mj_forward(model, data)
        exact_hold_load = abs(float(data.qfrc_bias[0]) - float(data.qfrc_passive[0]))
        dof_friction = float(model.dof_frictionloss[0]) if model.nv > 0 else 0.0
        resistive_est = round(exact_hold_load + dof_friction, 2)

        wall_ms = round((time.perf_counter() - start_ts) * 1000.0, 2)
        init_pos = round(spec.initial_position, 4)
        final_pos = round(positions[-1], 4)
        ss_error = round(abs(spec.target_position - final_pos), 4)
        sat_steps = sum(1 for f in forces if f >= 0.98 * gear)
        sat_pct = round((sat_steps / total_steps) * 100.0, 1)

        tol = round(max(abs(spec.target_position - init_pos) * 0.05, 0.001), 4)
        succeeded = ss_error <= tol

        gif_path = os.path.join(out_dir, f"rollout_iter_{iteration}.gif")
        self._render_trajectory_gif(
            spec, iteration, positions, forces, gear, sat_pct, succeeded, gif_path
        )

        status = "PASS [CONVERGED]" if succeeded else "FAIL [ACTUATOR STALLED]"
        diag = (
            f"{status}: Pos {init_pos:.4f} -> {final_pos:.4f}{spec.unit_label} "
            f"(target {spec.target_position:.4f}{spec.unit_label}, deficit={ss_error:.4f}). "
            f"Saturation={sat_pct}% at {gear}N vs {resistive_est}N load."
        )
        telemetry = RolloutTelemetry(
            iteration=iteration,
            total_steps=total_steps,
            wall_time_ms=wall_ms,
            initial_pos=init_pos,
            final_pos=final_pos,
            target_pos=spec.target_position,
            steady_state_error=ss_error,
            max_actuator_force=round(max(forces), 2),
            actuator_saturation_pct=sat_pct,
            resistive_load_estimate_n=resistive_est,
            task_succeeded=succeeded,
            diagnostic_summary=diag,
            gif_path=gif_path,
            applied_gear=gear,
            applied_kp=spec.kp_gain,
            applied_ki=spec.ki_gain,
            applied_kd=spec.kd_gain,
        )
        out_json = os.path.join(out_dir, f"rollout_{iteration}.json")
        with open(out_json, "w", encoding="utf-8") as fp:
            fp.write(telemetry.model_dump_json(indent=2))
        return telemetry

    @staticmethod
    def compute_patch(job_id: str, proposed_gear: float | None = None, rationale: str = "") -> PhysicsPatchProposal:
        out_dir = os.path.join("artifacts", job_id)
        spec = UniversalSimSpec.model_validate_json(open(os.path.join(out_dir, "spec.json"), encoding="utf-8").read())
        r1 = RolloutTelemetry.model_validate_json(open(os.path.join(out_dir, "rollout_1.json"), encoding="utf-8").read())

        min_safe_gear = float(math.ceil(max(r1.resistive_load_estimate_n * 1.65, spec.actuator_gear * 2.5)))
        final_gear = max(float(proposed_gear), min_safe_gear) if proposed_gear else min_safe_gear
        kp = round(spec.kp_gain * 2.2, 1)
        ki = round(spec.ki_gain * 1.35, 1)
        kd = round(spec.kd_gain * 1.4, 1)
        patched_xml = re.sub(r'gear="[^"]+"', f'gear="{final_gear}"', spec.mjcf_xml, count=1)

        if not rationale:
            rationale = (
                f"Hold load + dry friction at {spec.target_position}{spec.unit_label} is "
                f"{r1.resistive_load_estimate_n}N, stalling initial {spec.actuator_gear}N motor at "
                f"{r1.final_pos}{spec.unit_label}. Upgrading gear to {final_gear}N (1.65x headroom)."
            )

        patch = PhysicsPatchProposal(
            patched_xml=patched_xml,
            proposed_gear=final_gear,
            proposed_kp=kp,
            proposed_ki=ki,
            proposed_kd=kd,
            exceeds_thermal_cap=final_gear > spec.thermal_force_cap,
            rationale=rationale,
        )
        with open(os.path.join(out_dir, "patch.json"), "w", encoding="utf-8") as fp:
            fp.write(patch.model_dump_json(indent=2))
        return patch

    @staticmethod
    def apply_approved_patch(job_id: str) -> UniversalSimSpec:
        out_dir = os.path.join("artifacts", job_id)
        spec_path = os.path.join(out_dir, "spec.json")
        patch_path = os.path.join(out_dir, "patch.json")
        spec = UniversalSimSpec.model_validate_json(open(spec_path, encoding="utf-8").read())
        patch = PhysicsPatchProposal.model_validate_json(open(patch_path, encoding="utf-8").read())

        spec.mjcf_xml = patch.patched_xml
        spec.actuator_gear = patch.proposed_gear
        spec.kp_gain = patch.proposed_kp
        spec.ki_gain = patch.proposed_ki
        spec.kd_gain = patch.proposed_kd

        with open(spec_path, "w", encoding="utf-8") as fp:
            fp.write(spec.model_dump_json(indent=2))
        return spec

    @staticmethod
    def _render_trajectory_gif(
        spec: UniversalSimSpec,
        iteration: int,
        positions: list[float],
        forces: list[float],
        gear: float,
        sat_pct: float,
        succeeded: bool,
        gif_path: str,
    ) -> None:
        frames: list[Image.Image] = []
        span = max(abs(spec.target_position - positions[0]), 1e-5)
        dx, dy = spec.motion_vector

        for idx in range(0, len(positions), 20):
            img = Image.new("RGB", (600, 320), color=(13, 17, 23))
            draw = ImageDraw.Draw(img)
            pos = positions[idx]
            force = forces[idx]
            progress = min(max((pos - positions[0]) / span, -0.2), 1.25)

            status_col = (52, 211, 153) if succeeded else (251, 113, 133)
            draw.text((16, 10), f"JOB [{spec.job_id}] // {spec.task_name.upper()[:38]}", fill=(88, 166, 255))
            draw.text(
                (16, 30),
                f"ITER #{iteration}: {'CONVERGED' if succeeded else 'STALLED'} | "
                f"Pos: {pos:.4f} -> Goal: {spec.target_position:.4f}{spec.unit_label} | Sat: {sat_pct}%",
                fill=status_col,
            )

            draw.rectangle([16, 60, 290, 300], outline=(48, 54, 61), width=1)
            draw.text((26, 66), "SPATIAL TRAJECTORY (START -> GOAL)", fill=(139, 148, 158))
            cx, cy = 153, 185
            start_pt = (int(cx - dx * 75), int(cy - dy * 75))
            goal_pt = (int(cx + dx * 75), int(cy + dy * 75))
            curr_pt = (
                int(start_pt[0] + (goal_pt[0] - start_pt[0]) * progress),
                int(start_pt[1] + (goal_pt[1] - start_pt[1]) * progress),
            )

            draw.line([start_pt, goal_pt], fill=(55, 65, 81), width=3)
            draw.ellipse([start_pt[0] - 8, start_pt[1] - 8, start_pt[0] + 8, start_pt[1] + 8], outline=(156, 163, 175), width=2)
            draw.text((start_pt[0] - 24, start_pt[1] + 12), f"START ({positions[0]:.2f})", fill=(156, 163, 175))
            draw.ellipse([goal_pt[0] - 14, goal_pt[1] - 14, goal_pt[0] + 14, goal_pt[1] + 14], outline=(245, 158, 11), width=2)
            draw.text((goal_pt[0] - 28, goal_pt[1] - 28), f"GOAL ({spec.target_position:.2f})", fill=(245, 158, 11))

            node_col = (16, 185, 129) if succeeded else (244, 63, 94)
            draw.ellipse([curr_pt[0] - 12, curr_pt[1] - 12, curr_pt[0] + 12, curr_pt[1] + 12], fill=node_col, outline=(255, 255, 255), width=2)

            draw.rectangle([306, 60, 584, 245], outline=(48, 54, 61), width=1)
            draw.text((316, 66), "JOINT CONVERGENCE q(t) vs TARGET", fill=(139, 148, 158))
            target_y, zero_y = 100, 225
            draw.line([316, target_y, 574, target_y], fill=(245, 158, 11), width=1)
            draw.text((515, target_y - 14), "TARGET", fill=(245, 158, 11))

            pts = []
            for s in range(0, idx + 1, 10):
                px = 316 + int((s / 1000.0) * 255)
                p_norm = min(max((positions[s] - positions[0]) / span, -0.1), 1.2)
                py = int(zero_y - p_norm * (zero_y - target_y))
                pts.append((px, py))
            if len(pts) > 1:
                draw.line(pts, fill=node_col, width=3)

            draw.rectangle([306, 255, 584, 300], outline=(48, 54, 61), width=1)
            fill_w = int(min(force / max(gear, 1e-5), 1.0) * 256)
            bar_col = (244, 63, 94) if force >= 0.98 * gear else (56, 189, 248)
            draw.rectangle([316, 276, 316 + fill_w, 290], fill=bar_col)
            draw.text((316, 260), f"Motor Output: {force:.1f}N / Max Gear: {gear:.1f}N", fill=(201, 209, 217))

            frames.append(img)

        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=40, loop=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="500Hz MuJoCo CLI for Switch Console Agents")
    sub = parser.add_subparsers(dest="cmd")

    # Subcommand 1: architect
    p_arch = sub.add_parser("architect")
    p_arch.add_argument("--job-id", required=True)
    p_arch.add_argument("--mass-kg", type=float, required=True)
    p_arch.add_argument("--target", type=float, required=True, help="Stroke in meters or angle in radians")
    p_arch.add_argument("--unit", choices=["m", "rad"], default="m")
    p_arch.add_argument("--joint", choices=["slide", "hinge"], default="slide")
    p_arch.add_argument("--direction", choices=["up", "left", "right"], default="up")
    p_arch.add_argument("--prompt", default="Robotics Mechanism")

    # Subcommand 2: rollout
    p_roll = sub.add_parser("rollout")
    p_roll.add_argument("--job-id", required=True)
    p_roll.add_argument("--iter", type=int, default=1)

    # Subcommand 3: tune
    p_tune = sub.add_parser("tune")
    p_tune.add_argument("--job-id", required=True)
    p_tune.add_argument("--proposed-gear", type=float, default=None)
    p_tune.add_argument("--rationale", default="")

    # Subcommand 4: approve
    p_app = sub.add_parser("approve")
    p_app.add_argument("--job-id", required=True)

    # Backwards-compatible root flags if called without subcommand
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--iter", type=int, default=1)

    args = parser.parse_args()
    engine = MuJoCoRolloutEngine()

    if args.cmd == "architect":
        s = engine.build_spec(
            args.job_id, args.prompt, args.mass_kg, args.target, args.unit, args.joint, args.direction
        )
        print(s.model_dump_json(indent=2))
    elif args.cmd == "tune":
        p = engine.compute_patch(args.job_id, args.proposed_gear, args.rationale)
        print(p.model_dump_json(indent=2))
    elif args.cmd == "approve":
        s = engine.apply_approved_patch(args.job_id)
        print(json.dumps({"status": "PATCH_APPLIED", "job_id": s.job_id, "new_gear": s.actuator_gear}, indent=2))
    else:
        jid = args.job_id
        spec_file = os.path.join("artifacts", jid, "spec.json")
        with open(spec_file, encoding="utf-8") as f:
            spec_obj = UniversalSimSpec.model_validate_json(f.read())
        res = engine.execute_rollout(spec_obj, iteration=args.iter)
        print(res.model_dump_json(indent=2))
