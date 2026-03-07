"""
motor/semantic_actions.py
==========================
ACTION GROUNDING LAYER — The missing piece between VLM semantic intent and motor commands.

This module solves the Perception-Action Semantic Gap:
  - VLMs cannot think in metric space (they hallucinate coordinates)
  - π0/RT-2 style: VLM does semantic reasoning, a separate layer does spatial math
  - This layer uses MuJoCo ground-truth (sim) or depth sensors (real) to convert
    semantic verbs ("walk toward the red mug") into grounded motor commands

The VLM should ONLY see methods from this class — never raw arm_move_to(x,y,z).
"""
import math
import numpy as np
import mujoco
from typing import Optional
from motor.unitree_bridge import UnitreeBridge, RIGHT_HAND_BODY, LEFT_HAND_BODY
from motor.locomotion import LocomotionController
from motor.arm_controller import ArmController
from motor.grasp_controller import GraspController


class SemanticActions:
    """Semantic action interface for VLM-driven robot control.
    
    Every method here is a grounded semantic verb that the VLM can call.
    No method exposes raw coordinates — all spatial math is handled internally
    using MuJoCo ground-truth state.
    """

    def __init__(self, bridge: UnitreeBridge,
                 locomotion: LocomotionController,
                 arm: ArmController,
                 grasp: GraspController):
        self.bridge = bridge
        self.model = bridge.model
        self.data = bridge.data
        self.loco = locomotion
        self.arm = arm
        self.grasp = grasp

        # Build a map of named scene objects (non-robot bodies with geoms)
        self._scene_objects = {}
        self._build_scene_map()

    # ------------------------------------------------------------------
    # Internal: Scene understanding via MuJoCo ground truth
    # ------------------------------------------------------------------

    def _build_scene_map(self):
        """Index all named non-robot bodies for semantic reference."""
        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name and name not in ("world", "pelvis") and not name.endswith("_link"):
                self._scene_objects[name] = i

        # Also index named geoms that belong to worldbody (walls, floors, targets)
        self._scene_geoms = {}
        for i in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name and name not in ("floor",):
                body_id = self.model.geom_bodyid[i]
                body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                if body_name == "world" or body_name in self._scene_objects:
                    self._scene_geoms[name] = i

        n_obj = len(self._scene_objects)
        n_geom = len(self._scene_geoms)
        print(f"[SemanticActions] Scene map: {n_obj} objects, {n_geom} named geoms")

    def _get_robot_pos(self):
        """Get robot base (x, y) position in world frame."""
        return self.bridge.get_base_pos()[:2]

    def _get_robot_heading(self):
        """Get robot yaw angle in radians."""
        quat = self.bridge.get_base_quat()
        # Extract yaw from quaternion (w, x, y, z)
        w, x, y, z = quat
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return np.arctan2(siny_cosp, cosy_cosp)

    def _get_object_pos(self, name: str) -> Optional[np.ndarray]:
        """Get world position of a named object or geom."""
        mujoco.mj_forward(self.model, self.data)
        if name in self._scene_objects:
            return self.data.xpos[self._scene_objects[name]].copy()
        if name in self._scene_geoms:
            return self.data.geom_xpos[self._scene_geoms[name]].copy()
        return None

    def _is_path_blocked(self, start_pos: np.ndarray, end_pos: np.ndarray) -> Optional[str]:
        """Returns the name of the first obstacle blocking the straight-line path, or None."""
        obstacles = []
        for name, gid in self._scene_geoms.items():
            if "wall" in name or "block" in name:
                pos = self.data.geom_xpos[gid]
                size = self.model.geom_size[gid]
                obstacles.append((name, pos[0]-size[0], pos[0]+size[0], pos[1]-size[1], pos[1]+size[1]))
                
        # Sample points along the segment (inflation = 0.2m for robot width)
        for t in np.linspace(0.1, 0.9, 15):
            x = start_pos[0] + t * (end_pos[0] - start_pos[0])
            y = start_pos[1] + t * (end_pos[1] - start_pos[1])
            
            for obs_name, mx, Mx, my, My in obstacles:
                if (mx - 0.2) <= x <= (Mx + 0.2) and (my - 0.2) <= y <= (My + 0.2):
                    return obs_name
        return None

    def _relative_description(self, target_pos: np.ndarray, target_name: str = "") -> str:
        """Describe a target position relative to the robot in natural language."""
        robot_pos = self._get_robot_pos()
        heading = self._get_robot_heading()

        # Vector from robot to target in world frame
        dx = target_pos[0] - robot_pos[0]
        dy = target_pos[1] - robot_pos[1]
        dist = np.sqrt(dx**2 + dy**2)

        # Angle to target in world frame, then relative to robot heading
        angle_to_target = np.arctan2(dy, dx)
        relative_angle = angle_to_target - heading
        # Normalize to [-pi, pi]
        relative_angle = (relative_angle + np.pi) % (2 * np.pi) - np.pi
        angle_deg = np.degrees(relative_angle)

        # Direction words with angle
        if abs(angle_deg) < 15:
            lr = "directly ahead"
        elif angle_deg > 0:
            lr = f"{abs(angle_deg):.0f}° to your left"
        else:
            lr = f"{abs(angle_deg):.0f}° to your right"

        desc = f"{dist:.1f}m away, {lr}"
        
        # Check if the path is physically blocked by an obstacle
        blocker = self._is_path_blocked(robot_pos, target_pos)
        if blocker and blocker != target_name:
            desc += f" ⚠️ [BLOCKING PATH: '{blocker}']"
            
        return desc

    # ------------------------------------------------------------------
    # SEMANTIC VERBS — These are exposed to the VLM
    # ------------------------------------------------------------------

    def describe_scene(self) -> str:
        """Look around and describe what you see. Returns a text description of all
        visible objects, obstacles, and landmarks with their relative positions.
        Call this first to understand your environment before taking action."""
        mujoco.mj_forward(self.model, self.data)
        lines = []

        robot_pos = self._get_robot_pos()
        heading_deg = np.degrees(self._get_robot_heading())
        lines.append(f"Robot at ({robot_pos[0]:.1f}, {robot_pos[1]:.1f}), heading {heading_deg:.0f}°")
        lines.append("")

        # Scene geoms (walls, obstacles, targets)
        for name, gid in self._scene_geoms.items():
            pos = self.data.geom_xpos[gid]
            desc = self._relative_description(pos, name)
            geom_type = self.model.geom_type[gid]
            type_name = {0: "plane", 2: "sphere", 3: "capsule", 5: "cylinder", 6: "box"}.get(geom_type, "shape")
            lines.append(f"  • {name} ({type_name}): {desc}")

        # Freejoint objects
        for name, bid in self._scene_objects.items():
            pos = self.data.xpos[bid]
            desc = self._relative_description(pos, name)
            lines.append(f"  • {name}: {desc}")

        return "\n".join(lines)

    def walk_forward(self, duration_s: float = 2.0) -> str:
        """Walk straight forward for the given duration in seconds.
        Use this to move toward something that is directly ahead of you.
        Typical duration: 1.0 to 3.0 seconds."""
        duration_s = float(np.clip(duration_s, 0.5, 5.0))
        self.loco.walk_velocity(vx=0.5, vy=0.0, wz=0.0, duration_s=duration_s)
        pos = self._get_robot_pos()
        return f"Walked forward {duration_s:.1f}s. Now at ({pos[0]:.1f}, {pos[1]:.1f})"

    def walk_backward(self, duration_s: float = 1.0) -> str:
        """Walk backward for the given duration. Use to back away from obstacles."""
        duration_s = float(np.clip(duration_s, 0.5, 3.0))
        self.loco.walk_velocity(vx=-0.3, vy=0.0, wz=0.0, duration_s=duration_s)
        pos = self._get_robot_pos()
        return f"Walked backward {duration_s:.1f}s. Now at ({pos[0]:.1f}, {pos[1]:.1f})"

    def turn_left(self, duration_s: float = 2.0) -> str:
        """Turn left (counter-clockwise) in place for the given duration.
        ~2 seconds = roughly 60-90 degree turn."""
        duration_s = float(np.clip(duration_s, 0.5, 5.0))
        # Back up briefly to break wall contact friction before rotating
        self.loco.walk_velocity(vx=-0.3, vy=0.0, wz=0.0, duration_s=0.5)
        self.loco.walk_velocity(vx=0.0, vy=0.0, wz=0.8, duration_s=duration_s)
        heading = np.degrees(self._get_robot_heading())
        return f"Turned left {duration_s:.1f}s. Heading now {heading:.0f}°"

    def turn_right(self, duration_s: float = 2.0) -> str:
        """Turn right (clockwise) in place for the given duration.
        ~2 seconds = roughly 60-90 degree turn."""
        duration_s = float(np.clip(duration_s, 0.5, 5.0))
        # Back up briefly to break wall contact friction before rotating
        self.loco.walk_velocity(vx=-0.3, vy=0.0, wz=0.0, duration_s=0.5)
        self.loco.walk_velocity(vx=0.0, vy=0.0, wz=-0.8, duration_s=duration_s)
        heading = np.degrees(self._get_robot_heading())
        return f"Turned right {duration_s:.1f}s. Heading now {heading:.0f}°"

    def walk_toward_object(self, object_name: str) -> str:
        """Walk toward a named object or landmark. The robot will automatically
        navigate to within 0.5m of the target. 
        Example: walk_toward_object('target_zone') or walk_toward_object('obj_red_mug')"""
        pos = self._get_object_pos(object_name)
        if pos is None:
            return f"Cannot find object '{object_name}' in the scene."
        
        result = self.loco.walk_to(pos[0], pos[1], tolerance=0.5, max_duration_s=20.0)
        robot_pos = self._get_robot_pos()
        if result["success"]:
            return f"Arrived near '{object_name}'. Robot at ({robot_pos[0]:.1f}, {robot_pos[1]:.1f})"
        else:
            # Tell VLM where the target is NOW relative to robot so it can self-correct
            desc = self._relative_description(pos, object_name)
            return (f"Could not reach '{object_name}' (got to {result['final_distance']:.1f}m away). "
                    f"Target is {desc}.")

    def walk_to_waypoint(self, dx: float, dy: float) -> str:
        """Walk to an exact waypoint relative to your current position and heading.
        If the direct path to the target is blocked by a large obstacle, use this to navigate AROUND the obstacle first.
        Args:
            dx (float): Forward distance in meters (use negative for backward).
            dy (float): Left distance in meters (use negative for right).
        Example: to bypass a wall blocking the right side, walk_to_waypoint(1.0, 2.0) walk 1m forward and 2m left."""
        robot_pos = self._get_robot_pos()
        heading = self._get_robot_heading()
        
        # Transform relative (dx, dy) back to global (x, y) world coordinates
        # dx is forward (local X), dy is left (local Y)
        world_dx = dx * math.cos(heading) - dy * math.sin(heading)
        world_dy = dx * math.sin(heading) + dy * math.cos(heading)
        
        target_x = robot_pos[0] + world_dx
        target_y = robot_pos[1] + world_dy
        
        result = self.loco.walk_to(target_x, target_y, tolerance=0.3, max_duration_s=20.0)
        new_pos = self._get_robot_pos()
        if result["success"]:
            return f"Arrived at waypoint. Robot now at ({new_pos[0]:.1f}, {new_pos[1]:.1f})"
        else:
            return f"Could not reach waypoint (got to {result['final_distance']:.1f}m away). Blocked by obstacle."


    def reach_for_object(self, object_name: str) -> str:
        """Extend arm to reach for a named object. The robot uses ground-truth position
        to compute the exact arm target — no coordinate guessing needed.
        Only works if the object is within arm's reach (~0.6m from shoulder)."""
        pos = self._get_object_pos(object_name)
        if pos is None:
            return f"Cannot find object '{object_name}' in the scene."

        result = self.arm.move_to(pos)
        if result["success"]:
            return f"Successfully reached '{object_name}' (error: {result['final_error']*100:.1f}cm)"
        else:
            reason = result.get("abort_reason", "unknown")
            return f"Could not reach '{object_name}': {reason} (error: {result['final_error']*100:.1f}cm)"

    def grasp_nearest(self, hand: str = "right") -> str:
        """Close the gripper and grasp the nearest object within reach.
        The robot automatically detects what is close enough to grab.
        Args: hand — 'right' or 'left'"""
        result = self.grasp.grasp(hand=hand)
        if result["success"]:
            return f"Grasped '{result['object']}' with {hand} hand (distance: {result['distance']*100:.0f}cm)"
        else:
            reason = result.get("reason", "unknown")
            return f"Grasp failed: {reason}"

    def release(self, hand: str = "right") -> str:
        """Open the gripper and release whatever the specified hand is holding."""
        self.grasp.release(hand=hand)
        return f"Released object from {hand} hand."

    def stop(self) -> str:
        """Stop all movement immediately and stand still."""
        self.loco.stand_still(duration_s=1.0)
        return "Stopped. Standing still."
