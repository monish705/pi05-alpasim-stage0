import time
import numpy as np
from PIL import Image
import mujoco
import mujoco.viewer
from motor.unitree_bridge import UnitreeBridge
from motor.locomotion import LocomotionController

def run_navigation_test():
    print("Initializing Navigation Test in Complex Environment...")
    
    scene_xml = r"C:\Users\brind\Documents\unitree_embodied_ai\mujoco_menagerie\unitree_g1\g1_obstacle_course.xml"
    model = mujoco.MjModel.from_xml_path(scene_xml)
    data = mujoco.MjData(model)
    
    viewer = mujoco.viewer.launch_passive(model, data)
    with viewer.lock():
        viewer.cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.azimuth = -140
        viewer.cam.elevation = -20
        viewer.cam.distance = 3.0

    video_frames = []
    step_count = 0
    renderer = mujoco.Renderer(model, 480, 640)
    
    _og_step = mujoco.mj_step
    def sync_step(m, d):
        nonlocal step_count
        _og_step(m, d)
        if viewer.is_running():
            viewer.sync()
            time.sleep(model.opt.timestep) 
            
        step_count += 1
        if step_count % 50 == 0:  # 10fps
            renderer.update_scene(d, camera="overview_cam")
            pixels = renderer.render()
            video_frames.append(Image.fromarray(pixels))

    mujoco.mj_step = sync_step

    bridge = UnitreeBridge(model, data)
    bridge.reset_to_stand()
    mujoco.mj_forward(model, data)
    
    loco = LocomotionController(bridge)
    bridge.pre_step_hooks.append(loco.update)
    
    # Waypoints to avoid the obstacles defined in g1_obstacle_course.xml
    waypoints = [
        (1.2, -0.4),  # Go right to avoid the left wall and middle block
        (2.5, 0.6),   # Go left to avoid the right wall
        (4.0, 0.0)    # Arrive at target zone
    ]
    
    for i, (wx, wy) in enumerate(waypoints):
        print(f"\n[NAV] Moving to Waypoint {i+1}: ({wx}, {wy})")
        res = loco.walk_to(wx, wy, tolerance=0.3, max_duration_s=20.0)
        if not res["success"]:
            print(f"[NAV] ❌ Failed to reach Waypoint {i+1}. Result: {res}")
            break
        print(f"[NAV] ✅ Reached Waypoint {i+1} successfully.")
    
    print("\nNavigation completed. Keeping viewer open for 2 seconds...")
    start = time.time()
    while time.time() - start < 2 and viewer.is_running():
        mujoco.mj_step(model, data)
    viewer.close()
    
    if video_frames:
        print("\n[Video] Exporting WebP demonstration...")
        out_path = r"C:\Users\brind\.gemini\antigravity\brain\5dea276f-5269-4079-95bd-0b55d4f846a7\navigation_demo.webp"
        video_frames[0].save(
            out_path,
            save_all=True,
            append_images=video_frames[1:],
            duration=100,
            loop=0
        )
        print(f"[Video] Saved to {out_path}")

if __name__ == "__main__":
    run_navigation_test()
