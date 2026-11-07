from typing import Optional
from pydantic import BaseModel


class UniversalSimSpec(BaseModel):
    job_id: str
    task_name: str = "Robotics Mechanism"
    user_prompt: str = ""
    mjcf_xml: str
    joint_type: str = "hinge"
    unit_label: str = "rad"
    initial_position: float = 0.0
    target_position: float = 1.20
    payload_mass_kg: float = 1.0
    actuator_gear: float = 10.0
    thermal_force_cap: float = 20.0
    kp_gain: float = 90.0
    ki_gain: float = 180.0
    kd_gain: float = 12.0
    motion_vector: tuple[float, float] = (0.75, -0.65)


class RolloutTelemetry(BaseModel):
    iteration: int
    total_steps: int
    wall_time_ms: float
    initial_pos: float
    final_pos: float
    target_pos: float
    steady_state_error: float
    max_actuator_force: float
    actuator_saturation_pct: float
    resistive_load_estimate_n: float
    task_succeeded: bool
    diagnostic_summary: str
    gif_path: str
    applied_gear: Optional[float] = None
    applied_kp: Optional[float] = None
    applied_ki: Optional[float] = None
    applied_kd: Optional[float] = None


class PhysicsPatchProposal(BaseModel):
    patched_xml: str
    proposed_gear: float
    proposed_kp: float = 198.0
    proposed_ki: float = 240.0
    proposed_kd: float = 16.8
    exceeds_thermal_cap: bool = True
    rationale: str = ""
