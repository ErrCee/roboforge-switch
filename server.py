import glob
import json
import os
import re
import xml.etree.ElementTree as ET
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI(title="RoboForge-Switch // Live Iteration & Telemetry Observer")
os.makedirs("artifacts", exist_ok=True)
app.mount("/artifacts", StaticFiles(directory="artifacts"), name="artifacts")


def _read_json_safe(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as fp:
            return json.load(fp)
    except Exception:
        return None


def _extract_gear_from_xml(xml_path: str) -> float | None:
    if not os.path.exists(xml_path):
        return None
    try:
        root = ET.parse(xml_path).getroot()
        motor = root.find(".//motor")
        if motor is not None and motor.get("gear"):
            return float(motor.get("gear"))
    except Exception:
        pass
    return None


def scan_switch_jobs() -> list[dict]:
    jobs = []
    folders = [f for f in glob.glob("artifacts/*") if os.path.isdir(f)]
    folders.sort(key=lambda p: os.path.getmtime(p), reverse=True)

    for folder in folders:
        job_id = os.path.basename(folder)
        spec = _read_json_safe(os.path.join(folder, "spec.json")) or {}
        patch = _read_json_safe(os.path.join(folder, "patch.json"))

        rollout_files = sorted(
            glob.glob(os.path.join(folder, "rollout_*.json")),
            key=lambda x: int(re.search(r"rollout_(\d+)\.json", x).group(1))
            if re.search(r"rollout_(\d+)\.json", x)
            else 99,
        )
        iterations = []
        for rf in rollout_files:
            m = re.search(r"rollout_(\d+)\.json", rf)
            if not m:
                continue
            it_num = int(m.group(1))
            r_data = _read_json_safe(rf)
            if not r_data:
                continue
            gif_file = os.path.join(folder, f"rollout_iter_{it_num}.gif")
            xml_file = os.path.join(folder, f"model_iter_{it_num}.xml")
            gif_url = (
                f"/{gif_file}?t={int(os.path.getmtime(gif_file) * 1000)}"
                if os.path.exists(gif_file)
                else None
            )
            if r_data.get("applied_gear") is None:
                r_data["applied_gear"] = _extract_gear_from_xml(xml_file) or spec.get("actuator_gear")
            r_data["gif_url"] = gif_url
            iterations.append(r_data)

        if not spec and not iterations:
            continue

        latest_iter = iterations[-1] if iterations else None
        spec_gear = spec.get("actuator_gear")
        patch_gear = patch.get("proposed_gear") if patch else None
        patch_applied_to_spec = (
            patch is not None
            and spec_gear is not None
            and patch_gear is not None
            and abs(float(spec_gear) - float(patch_gear)) < 1e-3
        )

        if latest_iter and latest_iter.get("task_succeeded"):
            stage_num = 5
            stage_label = f"CONVERGED (ITERATION #{latest_iter['iteration']})"
            active_agent = "Complete — Verified in MuJoCo"
        elif patch_applied_to_spec and len(iterations) < 2:
            stage_num = 4
            stage_label = "PATCH APPLIED — RUNNING ITERATION #2"
            active_agent = "@telemetry-observer (executing rollout #2)"
        elif patch is not None and len(iterations) == 1:
            stage_num = 3
            stage_label = "AWAITING HUMAN GOVERNANCE IN SWITCH"
            active_agent = f"@control-firmware (waiting for: approve {job_id})"
        elif len(iterations) == 1 and not iterations[0].get("task_succeeded"):
            stage_num = 2
            stage_label = "ITERATION #1 STALLED — TUNING"
            active_agent = "@physics-tuner (computing inverse-dynamics patch)"
        else:
            stage_num = 1
            stage_label = "MJCF SPEC SYNTHESIZED — RUNNING ITERATION #1"
            active_agent = "@telemetry-observer (executing rollout #1)"

        jobs.append({
            "job_id": job_id,
            "stage_num": stage_num,
            "stage_label": stage_label,
            "active_agent": active_agent,
            "prompt": spec.get("user_prompt") or spec.get("task_name") or job_id,
            "spec": spec,
            "patch": patch,
            "iterations": iterations,
            "last_updated": int(os.path.getmtime(folder) * 1000),
        })
    return jobs


@app.get("/api/jobs")
async def api_jobs() -> dict:
    return {"jobs": scan_switch_jobs()}


@app.get("/", response_class=HTMLResponse)
async def judge_dashboard() -> HTMLResponse:
    return HTMLResponse("""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>RoboForge-Switch // Live Telemetry & Iteration Observer</title>
<script src="https://cdn.tailwindcss.com"></script></head>
<body class="bg-slate-950 text-slate-100 font-mono p-6">
  <div class="max-w-7xl mx-auto space-y-6">
    <div class="flex flex-wrap justify-between items-center border-b border-slate-800 pb-4 gap-3">
      <div>
        <h1 class="text-xl font-bold text-emerald-400">ROBOFORGE-SWITCH // LIVE 500Hz MUJOCO ITERATION OBSERVER</h1>
        <p class="text-xs text-slate-400 mt-1">
          Real-time filesystem monitor watching <code class="text-cyan-400">artifacts/&lt;job_id&gt;/</code> generated by your 4 Switch Console agents
        </p>
      </div>
      <div class="flex items-center gap-3">
        <span class="inline-block w-2.5 h-2.5 rounded-full bg-emerald-400 animate-ping"></span>
        <span id="jobCounter" class="text-xs bg-slate-900 border border-slate-700 px-3 py-1.5 rounded text-cyan-300">Scanning artifacts/...</span>
      </div>
    </div>
    <div id="jobsContainer" class="space-y-6"></div>
  </div>
  <script>
    let lastSignature = "";
    const STAGES = [
      "1. @architect-agent (MJCF Spec)",
      "2. @telemetry-observer (Iter #1)",
      "3. @physics-tuner (Load Patch)",
      "4. @control-firmware (Human Sign-Off)",
      "5. @telemetry-observer (Verified)"
    ];

    async function pollDisk() {
      const res = await fetch('/api/jobs');
      const data = await res.json();
      const jobs = data.jobs || [];
      document.getElementById('jobCounter').innerText = `${jobs.length} Job(s) Tracked in artifacts/`;

      const sig = JSON.stringify(jobs.map(j => [j.job_id, j.stage_num, j.iterations.length, j.last_updated]));
      if (sig === lastSignature) return;
      lastSignature = sig;

      const root = document.getElementById('jobsContainer');
      if (jobs.length === 0) {
        root.innerHTML = `
          <div class="border border-dashed border-slate-800 bg-slate-900/30 rounded p-12 text-center space-y-2">
            <div class="text-sm text-slate-300 font-bold">No Switch Jobs Found in <code class="text-cyan-400">artifacts/</code> Yet</div>
            <div class="text-xs text-slate-500">Post a problem statement to <code class="text-emerald-400">@architect-agent</code> in your Switch room to watch live iterations appear here.</div>
          </div>`;
        return;
      }

      root.innerHTML = jobs.map(j => {
        const stepper = STAGES.map((label, idx) => {
          const step = idx + 1;
          const done = j.stage_num > step || (j.stage_num === 5 && step === 5);
          const current = j.stage_num === step && j.stage_num !== 5;
          const cls = done
            ? "bg-emerald-950/80 border-emerald-600 text-emerald-300"
            : current
            ? "bg-amber-950/80 border-amber-500 text-amber-300 animate-pulse"
            : "bg-slate-900 border-slate-800 text-slate-500";
          return `<div class="border px-2.5 py-1.5 rounded text-[11px] font-bold text-center ${cls}">${label}</div>`;
        }).join('');

        const iterCards = j.iterations.map(it => {
          const ok = it.task_succeeded;
          const borderCol = ok ? "border-emerald-700/70 bg-emerald-950/10" : "border-rose-800/70 bg-rose-950/10";
          const badgeCol = ok ? "bg-emerald-900 text-emerald-300" : "bg-rose-900 text-rose-300";
          return `
            <div class="border ${borderCol} p-3.5 rounded space-y-2.5">
              <div class="flex justify-between items-center">
                <span class="text-xs font-bold ${ok ? 'text-emerald-400' : 'text-rose-400'}">
                  ITERATION #${it.iteration} ${it.applied_gear ? `(Motor Gear: ${it.applied_gear}N)` : ''}
                </span>
                <span class="text-[10px] px-2 py-0.5 rounded font-bold ${badgeCol}">
                  ${ok ? 'PASS // CONVERGED' : 'FAIL // ACTUATOR STALLED'}
                </span>
              </div>
              ${it.gif_url ? `<img src="${it.gif_url}" class="w-full rounded border border-slate-800"/>` : '<div class="h-44 flex items-center justify-center text-xs text-slate-500">Rendering GIF...</div>'}
              <div class="grid grid-cols-3 gap-2 text-[11px] bg-slate-950 p-2.5 rounded border border-slate-800/80">
                <div><span class="text-slate-500">Trajectory:</span><br/><span class="text-slate-200 font-bold">${it.initial_pos} &rarr; ${it.final_pos} (Goal: ${it.target_pos})</span></div>
                <div><span class="text-slate-500">Saturation:</span><br/><span class="${it.actuator_saturation_pct > 80 ? 'text-rose-400' : 'text-emerald-400'} font-bold">${it.actuator_saturation_pct}% (Peak: ${it.max_actuator_force}N)</span></div>
                <div><span class="text-slate-500">Hold Load / Time:</span><br/><span class="text-cyan-300 font-bold">${it.resistive_load_estimate_n}N (${it.wall_time_ms}ms)</span></div>
              </div>
              <div class="text-[11px] text-slate-400">${it.diagnostic_summary}</div>
            </div>
          `;
        }).join('');

        const pendingCard = j.iterations.length < 2 ? `
          <div class="border border-dashed border-slate-800 bg-slate-950/60 p-4 rounded flex flex-col items-center justify-center text-center min-h-[260px] space-y-2">
            <div class="text-xs font-bold text-amber-400">${j.stage_label}</div>
            <div class="text-[11px] text-slate-400 max-w-md">Active in Switch Room: <span class="text-cyan-300">${j.active_agent}</span></div>
            ${j.patch ? `
              <div class="mt-2 text-left text-[11px] bg-slate-900 border border-amber-800/50 p-3 rounded w-full space-y-1">
                <div class="font-bold text-amber-300">Proposed Physics Patch (patch.json):</div>
                <div class="text-slate-300">Gear Upgrade: <code class="text-rose-400">${j.iterations[0]?.applied_gear ?? j.spec.actuator_gear}N</code> &rarr; <code class="text-emerald-400 font-bold">${j.patch.proposed_gear}N</code> (Thermal Cap: ${j.spec.thermal_force_cap ?? 'N/A'}N)</div>
                <div class="text-slate-400">${j.patch.rationale || ''}</div>
              </div>
            ` : ''}
          </div>
        ` : '';

        return `
          <div class="border border-slate-800 bg-slate-900/50 p-5 rounded-lg space-y-4">
            <div class="flex flex-wrap justify-between items-center border-b border-slate-800 pb-3 gap-2">
              <div>
                <span class="text-cyan-400 font-bold text-base">${j.job_id}</span>
                <span class="text-slate-200 text-xs ml-3">"${j.prompt}"</span>
              </div>
              <div class="flex items-center gap-3">
                <span class="text-[11px] text-slate-400">Next Step: <strong class="text-cyan-300">${j.active_agent}</strong></span>
                <span class="text-xs px-2.5 py-1 rounded font-bold ${
                  j.stage_num === 5 ? 'bg-emerald-900 text-emerald-300' :
                  j.stage_num === 3 ? 'bg-amber-900 text-amber-300 animate-pulse' :
                  'bg-sky-950 text-sky-300 border border-sky-800'
                }">${j.stage_label}</span>
              </div>
            </div>
            <div class="grid grid-cols-5 gap-2">${stepper}</div>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
              ${iterCards}
              ${pendingCard}
            </div>
          </div>
        `;
      }).join('');
    }

    setInterval(pollDisk, 1000);
    pollDisk();
  </script>
</body></html>""")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8090)
