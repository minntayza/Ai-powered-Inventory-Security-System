# 5-Day Agile Project Execution Plan: VLM-Powered Smart Warehouse Security and Interactive Surveillance System

## 1. Project Overview
The "VLM-Powered Smart Warehouse Security and Interactive Surveillance System" is a highly innovative, zero-budget, software-driven security solution. It utilizes existing consumer hardware—a Lenovo Legion 5 laptop as the Edge Node, a smartphone as an IP Camera (via Iriun/DroidCam), laptop speakers, and a smartphone hotspot for localized networking. 

The system combines computer vision (CV) and vision-language models (VLMs) to track inventory in real time, detect theft automatically, and let operators ask it questions in plain language. Furthermore, it runs on edge hardware, so it is fast and works without a constant internet connection.

### Core Architectural Modules
* **Module A: Perception & Security Engine:** Uses a computer vision model (YOLOv8) to track shelf inventory and keep an accurate item count. To prevent GPU overload and ensure 100% reliability, it uses **OpenCV Visual Marker Detection (ArUco/QR Code)**. It scans individuals entering the frame for an authorized ID badge (marker). It classifies individuals wearing the badge as "Authorized Personnel" and those without as "Unknown/Intruder". It flags a "Theft" event when the inventory count drops while an unknown person is nearby.
* **Module B: Cognitive Vision-Language Layer:** Runs a lightweight Vision-Language Model on edge hardware, so it can process images and text together. Visual Question Answering (VQA) lets an operator ask questions about the warehouse in plain language (e.g., "What is the person in the video doing?"). Text-to-Speech (TTS) turns the VLM's text answers into spoken audio, so operators can interact hands-free.
* **Module C: Action & User Interface Layer:** Uses a centralized dashboard built with Streamlit that puts all monitoring and interactive features on one screen. When a theft event is detected, the UI switches into a Red Alert state, a local siren sounds, and a photo and alert are sent to the administrator's phone via the Telegram Bot API. Every event is logged to a secure database for review and compliance.

---

## 2. Team Member Roles & Responsibilities
* **1. Computer Vision (CV) Engineer:** Responsible for Module A. Focuses on YOLOv8 for inventory counting (e.g., bottles, backpacks) and implementing OpenCV ArUco marker detection to reliably differentiate between Authorized and Unknown personnel.
* **2. VLM & AI Integration Engineer:** Responsible for Module B. Manages the localized deployment of Florence-2/Moondream2, handles the VQA pipeline, and integrates `pyttsx3` for TTS audio output.
* **3. Frontend Developer:** Responsible for Module C's dashboard. Builds the Streamlit interface, ensuring a seamless layout containing the operational view (video feed, inventory count, bounding boxes) and cognitive assistant view (chat box, activity feed).
* **4. Backend & Alerting Engineer:** Responsible for core event logic. Manages the "Theft Event" condition, CSV logging, MP3 siren triggers, and the Telegram Bot API payload transmission.
* **5. Hardware & Network Specialist:** Responsible for physical infrastructure. Configures the smartphone IP camera, establishes the local hotspot LAN, ensures the CUDA/Python environment is fully optimized on the Lenovo Legion 5, and prints/prepares the physical ArUco ID badges and exhibition layout.
* **6. Project Manager (PM) & QA/Presentation Lead:** Responsible for agile coordination, strict timeline enforcement, end-to-end QA testing, system documentation, and final pitch presentation preparation.

---

## 3. Day-by-Day Task Breakdown

### Day 1: Environment Setup & Component Prototyping
* **CV Engineer:** Setup YOLOv8 environment. Generate and print test ArUco markers. Gather sample images of specific target items (bottles, backpacks) and begin preliminary model testing.
* **VLM Engineer:** Download and instantiate Florence-2/Moondream2 on the edge node. Verify basic prompt-response functionality with static test images.
* **Frontend Developer:** Initialize the Streamlit repository. Draft the basic dual-panel wireframe layout (Left: Operational View, Right: Cognitive Assistant & Event Log).
* **Backend Engineer:** Create the base Python event loop framework. Register the Telegram bot via BotFather and test basic text message delivery.
* **Hardware Specialist:** Configure the Lenovo laptop with required NVIDIA/CUDA drivers. Set up the smartphone hotspot and DroidCam/Iriun stream.
* **PM & QA Lead:** Run the project kickoff. Establish the Git repository, define the API contracts (how the YOLO outputs feed into the core logic), and draft the testing criteria.

### Day 2: Core Module Development (Independent Work)
* **CV Engineer:** Integrate OpenCV ArUco detection with YOLOv8. Ensure the system checks for a marker *inside* a detected person's bounding box to flag them as "Authorized". Write the script to output dynamic item counts.
* **VLM Engineer:** Build the integration between the VLM and `pyttsx3`. Ensure the VLM can accept a cropped image frame and a text prompt, outputting a verbalized text response within a reasonable latency window.
* **Frontend Developer:** Build the real-time widget showing the current inventory count. Develop the chat box for asking the VLM about the warehouse, by text or voice. 
* **Backend Engineer:** Code the automated theft detection logic that flags a "Theft" event when the inventory count drops while an "Unknown" person (no ArUco badge detected) is nearby. Connect the MP3 siren trigger to this logic.
* **Hardware Specialist:** Test thermal throttling on the edge node when running AI models. Finalize camera placement angles for the physical demonstration to ensure ArUco badges on the chest are clearly visible.
* **PM & QA Lead:** Conduct end-of-day module reviews. Identify integration blockers. Begin drafting the project documentation and slide deck.

### Day 3: System Integration (Merging Modules Together)
* **CV Engineer & Backend Engineer:** Link the YOLO bounding box and ArUco marker states to the backend logic engine. Ensure the Python script correctly tracks state changes.
* **VLM Engineer & Frontend Developer:** Connect the Streamlit chat interface directly to the VLM pipeline. Ensure VLM responses are routed to the Streamlit UI and the `pyttsx3` audio engine simultaneously.
* **Hardware Specialist:** Assist with threading and GPU memory allocation, ensuring YOLOv8, ArUco detection, and the VLM can all run concurrently on the laptop's GPU without OOM (Out of Memory) errors.
* **PM & QA Lead:** Facilitate the integration merge. Run the first end-to-end test (Camera -> YOLO/Marker -> Logic -> Alert). Document any latency bottlenecks.

### Day 4: Full System QA, Debugging, and Optimization
* **CV Engineer:** Refine tracking confidence thresholds to reduce false positives/negatives. Address lighting condition variances in the video feed to ensure ArUco markers are always readable.
* **VLM Engineer:** Optimize VLM inference speed. Implement prompt caching or frame skipping if necessary to ensure the interactive investigation operates smoothly.
* **Frontend Developer:** Ensure that when a theft event is detected, the UI switches into a Red Alert state. Finalize the activity feed, ensuring a running, timestamped log of events is displayed.
* **Backend Engineer:** Test the Telegram bot payload. Ensure a photo and alert are sent to the administrator's phone via the Telegram Bot API during a breach.
* **Hardware Specialist:** Stress-test the localized network. Simulate connection drops and measure recovery times.
* **PM & QA Lead:** Lead formal QA sessions. Execute a simulated incident where an intruder takes an item vs. an authorized user taking an item. Validate the dashboard, alert execution, and the ability to ask the VLM questions about the incident frame.

### Day 5: Final Polish, Physical Dry-Run, and Exhibition Prep
* **CV Engineer & VLM Engineer:** Final code freeze. Clean up scripts, remove debug print statements, and package the Python environment (`requirements.txt`).
* **Frontend Developer & Backend Engineer:** Finalize the comprehensive audit trail, ensuring every event (removals, intrusions, timestamped images) is logged to a secure database for review and compliance.
* **Hardware Specialist:** Set up the physical exhibition desk (bottles, backpacks). Tape down wires, calibrate the final camera focus, and secure the networking hotspot.
* **PM & QA Lead:** Run multiple full-scale dry runs of the exhibition pitch. Oversee the live demonstration flow: Continuous Surveillance (Authorized vs. Unauthorized) -> Intrusion & Theft Detection -> Alert Execution -> Interactive Investigation. Finalize all deliverables.

---

## 4. Risk Mitigation & Contingency Plan
* **ArUco Marker Occlusion (Badge Not Visible):** 
  * *Risk:* An authorized user's badge is briefly turned away from the camera, triggering a false alarm when they touch an item. 
  * *Contingency:* Implement a multi-frame buffer (e.g., 3-second grace period) in the logic engine to allow the system to re-detect the badge before confirming a Theft Event.
* **VLM Inference Latency (GPU Bottleneck):** 
  * *Risk:* Running YOLOv8 and Florence-2 simultaneously may cause lag. 
  * *Contingency:* Downscale the VLM input image resolution. If necessary, pause the YOLO inference momentarily while the VLM processes a conversational query.
* **Network Instability (Hotspot Drops):** 
  * *Risk:* The smartphone hotspot fails, disconnecting the IP camera. 
  * *Contingency:* Keep a wired USB webcam on standby as an immediate backup for the perception layer. 
* **Audio Feedback Loops:** 
  * *Risk:* The MP3 siren triggers the VLM voice input. 
  * *Contingency:* Implement software muting of the microphone input while the siren or TTS is actively playing.