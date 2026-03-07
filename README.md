# 🦾 Unitree G1 Embodied AI: Auto-Discovery Control Pipeline

![Status](https://img.shields.io/badge/Status-Prototype-orange)
![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Sim](https://img.shields.io/badge/MuJoCo-3.1.2-red)
![FastAPI](https://img.shields.io/badge/FastAPI-Server-blue)
![License](https://img.shields.io/badge/License-MIT-green)

A cutting-edge, end-to-end embodied AI architecture for the Unitree G1 humanoid robot. This project demonstrates a state-of-the-art hierarchical control pipeline: translating high-level natural language semantic goals into dynamic waypoint routing, while a local solver executes real-time obstacle avoidance.

## 🧠 System Hierarchy

The system is built on a decoupled hierarchy prioritizing asynchronous execution:

1.  **Macro-Planner (VLM Orchestrator):** `vlm_navigator.py` queries endpoints to fetch the robot's Ego-centric Head Camera. If a massive obstacle blocks the path, the VLM dynamically replans and issues global `walk_to_waypoint()` commands.
2.  **Telemetry Micro-Server:** `server.py` hosts a headless FastAPI instance exposing robot actions. It manages command queueing so the intensive physics solver never blocks during API latency. 
3.  **Local Navigator (APF Field):** `motor/locomotion.py` utilizes an Artificial Potential Field to slide around walls, creep past corners, and execute stall-recovery maneuvers (all without crashing the robot).
4.  **Balance Engine (RL Policy):** An ONNX-exported neural network policy running at 50Hz that queries the `UnitreeBridge` to compute raw joint torques natively in MuJoCo to keep the G1 stable and walking.

## 🚀 Getting Started

### Prerequisites

*   Python 3.11+
*   FastAPI & Uvicorn
*   MuJoCo 3.1.2+
*   Groq API Key (for LLaMA Vision)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/unitree-embodied-ai.git
   cd unitree_embodied_ai
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the VLM E2E Simulation

1. **Boot the Headless Physics Server:**
   ```bash
   python server.py
   ```
   *The Unitree G1 will spawn and balance infinitely in real-time.*

2. **Engage the VLM Cognitive Loop:**
   In a separate terminal, launch the brain:
   ```bash
   python vlm_navigator.py
   ```
   *The VLM will map the obstacle course and begin issuing routing commands.*

## 🎥 Core Capabilities
*   **Decoupled Rate Limiting:** The robot won't collapse if the LLM API hits a rate limit; the background server autonomously continues balancing. 
*   **Dynamic Evasion:** The macroscopic VLM works in tandem with the microscopic APF navigation to flawlessly route around geometry snags.

## 📄 License
MIT License
