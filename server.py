import threading
import queue
import inspect
import sys
import time
import base64
import io
from PIL import Image

import fastapi
from pydantic import BaseModel
import uvicorn

import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from motor.unitree_bridge import UnitreeBridge
from motor.arm_controller import ArmController
from motor.locomotion import LocomotionController
from motor.grasp_controller import GraspController
from motor.semantic_actions import SemanticActions
import mujoco
import mujoco.viewer

app = fastapi.FastAPI(title="Unitree Telemetry Server")

# Global state
class SimState:
    def __init__(self):
        self.model = None
        self.data = None
        self.viewer = None
        self.bridge = None
        self.semantic = None
        self.renderer = None
        self.tools = []
        self.method_map = {}

sim = SimState()
command_queue = queue.Queue()

def init_sim():
    print("====== INITIALIZING MUJOCO TELEMETRY SERVER ======")
    import os
    scene_xml = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mujoco_menagerie", "unitree_g1", "g1_obstacle_course.xml")
    sim.model = mujoco.MjModel.from_xml_path(scene_xml)
    sim.data = mujoco.MjData(sim.model)
    
    sim.viewer = mujoco.viewer.launch_passive(sim.model, sim.data)
    with sim.viewer.lock():
        sim.viewer.cam.trackbodyid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        sim.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        sim.viewer.cam.azimuth = -140
        sim.viewer.cam.elevation = -20
        sim.viewer.cam.distance = 3.0
        
    _og_step = mujoco.mj_step
    def sync_step(m, d):
        _og_step(m, d)
        if sim.viewer.is_running():
            sim.viewer.sync()
            time.sleep(sim.model.opt.timestep)
            
    mujoco.mj_step = sync_step

    sim.bridge = UnitreeBridge(sim.model, sim.data)
    sim.bridge.reset_to_stand()
    mujoco.mj_forward(sim.model, sim.data)
    
    sim.renderer = mujoco.Renderer(sim.model, 480, 640)
    
    loco = LocomotionController(sim.bridge)
    arm = ArmController(sim.bridge, hand="right")
    grasp = GraspController(sim.bridge)
    
    sim.bridge.pre_step_hooks.append(grasp.update)
    sim.bridge.pre_step_hooks.append(loco.update)
    
    sim.semantic = SemanticActions(sim.bridge, loco, arm, grasp)
    
    methods = inspect.getmembers(sim.semantic, predicate=inspect.ismethod)
    for name, func in methods:
        if name.startswith("_"): continue
        sig = inspect.signature(func)
        doc = inspect.getdoc(func) or f"Execute {name}"
        properties = {}
        required = []
        for param_name, param in sig.parameters.items():
            if param_name == "self": continue
            
            # Note: We intentionally map everything to type 'string' in the JSON schema.
            # Llama models frequently hallucinate strings for number inputs (e.g. {"dx": "2.0"}).
            # If we enforce 'number', Groq's API throws a strict HTTP 400 error and drops the call.
            # We accept strings and typecast them back to float/int inside the execute endpoint.
            ptype = "string" 
            properties[param_name] = {
                "type": ptype, 
                "description": f"({param.annotation.__name__ if hasattr(param.annotation, '__name__') else str(param.annotation)}) {param_name}"
            }
            if param.default == inspect.Parameter.empty:
                required.append(param_name)
        sim.tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": doc,
                "parameters": {"type": "object", "properties": properties, "required": required}
            }
        })
        sim.method_map[name] = func

@app.on_event("startup")
def startup_event():
    # We initialize here so that the FastAPI workers can read the static tools list.
    pass

@app.get("/discover")
def discover():
    return {"tools": sim.tools}

@app.get("/perception")
def get_perception():
    # Post a perception request to the main thread
    future = queue.Queue()
    command_queue.put(("perception", None, future))
    return future.get(timeout=10.0)

class CommandReq(BaseModel):
    action: str
    args: dict

@app.post("/execute")
def execute(req: CommandReq):
    if req.action not in sim.method_map:
        return {"success": False, "result": f"Error: Tool {req.action} not found."}
        
    # VLM Type Hallucination Fix: Auto-cast string arguments back to their required Python types
    func = sim.method_map[req.action]
    sig = inspect.signature(func)
    for k, v in req.args.items():
        if k in sig.parameters:
            anno = sig.parameters[k].annotation
            if anno in [float, 'float'] and isinstance(v, str):
                try: req.args[k] = float(v)
                except ValueError: pass
            elif anno in [int, 'int'] and isinstance(v, str):
                try: req.args[k] = int(float(v))
                except ValueError: pass
                
    future = queue.Queue()
    command_queue.put(("execute", req, future))
    try:
        # Wait for the main thread to execute the command (e.g. up to 300 seconds for long walks)
        return future.get(timeout=300.0)
    except queue.Empty:
         return {"success": False, "result": "Action failed: Server execution timeout"}

def main_thread_loop():
    init_sim()
    print("====== MUJOCO MAIN THREAD ENGAGED ======")
    while True:
        try:
            cmd_type, req, future = command_queue.get(timeout=0.01)
            
            if cmd_type == "perception":
                try:
                    # Use the ego-centric head_cam, not the static overhead god_cam
                    sim.renderer.update_scene(sim.data, camera="head_cam")
                    pixels = sim.renderer.render()
                    img = Image.fromarray(pixels)
                    buffered = io.BytesIO()
                    img.save(buffered, format="JPEG", quality=85)
                    img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    scene_text = sim.semantic.describe_scene()
                    future.put({
                        "image_b64": img_b64,
                        "scene_text": f"Current Scene:\n{scene_text}"
                    })
                except Exception as e:
                    future.put({"error": str(e)})
                    
            elif cmd_type == "execute":
                try:
                    result_text = sim.method_map[req.action](**req.args)
                    future.put({"success": True, "result": result_text})
                except Exception as e:
                    future.put({"success": False, "result": f"Action failed: {e}"})
                    
        except queue.Empty:
            # Idle simulation step to keep the viewer alive AND keep the RL policy running
            if sim.viewer.is_running():
                sim.bridge.step(1)
            else:
                break

if __name__ == "__main__":
    # Run uvicorn in a daemon thread
    server_thread = threading.Thread(
        target=lambda: uvicorn.run(app, host="0.0.0.0", port=8000, log_level="error"),
        daemon=True
    )
    server_thread.start()
    
    # Run MuJoCo exclusively on the main thread
    main_thread_loop()
