import sys
import os
import mujoco
from PIL import Image

sys.path.insert(0, r"C:\Users\brind\Documents\unitree_embodied_ai")

# Load model and initialize simulation state
xml_path = r"C:\Users\brind\Documents\unitree_embodied_ai\mujoco_menagerie\unitree_g1\g1_obstacle_course.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Forward kinematics to update geom positions
mujoco.mj_forward(model, data)

# Render the god_cam view (same resolution as VLM)
renderer = mujoco.Renderer(model, 480, 640)
renderer.update_scene(data, camera="god_cam")
pixels = renderer.render()

# Save the frame to the artifacts directory
out_path = r"C:\Users\brind\.gemini\antigravity\brain\5dea276f-5269-4079-95bd-0b55d4f846a7\god_cam_view.jpg"
img = Image.fromarray(pixels)
img.save(out_path, quality=95)
print(f"✅ Rendered VLM perception test frame to:\n   {out_path}")
