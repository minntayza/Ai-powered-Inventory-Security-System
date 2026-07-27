# Module B: Cognitive Vision-Language Layer

**Owner:** VLM & AI Integration Engineer

## Folder
```
src/module_b_vlm_layer/
```

## Files to Create

| File | Purpose |
|---|---|
| `vlm_engine.py` | Florence-2 / Moondream2 model loading and inference on edge GPU |
| `vqa_pipeline.py` | Visual Question Answering: accept cropped frame + text prompt, return text answer |
| `tts_engine.py` | pyttsx3 Text-to-Speech: convert VLM text responses to spoken audio |
| `prompt_cache.py` | Prompt caching and frame-skipping logic to optimize inference latency |

## Key Integration Points

- **Input from Frontend (`src/module_c_ui_dashboard/`):** User text/voice questions from chat box
- **Input from `src/utils/frame_capture.py`:** Current camera frame (cropped if needed)
- **Output to Frontend:** VLM text answer to display in chat
- **Output to `src/utils/audio_control.py`:** Signal when TTS is playing (for mic muting)
- **Shared config:** `configs/model_config.yaml` — VLM model name, resolution, max tokens, device

## Day-by-Day Deliverables

| Day | Task |
|---|---|
| 1 | Download and instantiate Florence-2/Moondream2, verify basic prompt-response with static images |
| 2 | Build VLM + pyttsx3 pipeline, accept cropped frame + text prompt, output verbalized response |
| 3 | Connect VLM pipeline to Streamlit chat UI and pyttsx3 audio engine simultaneously |
| 4 | Optimize inference speed, implement prompt caching or frame skipping for smooth interaction |
| 5 | Code freeze, remove debug statements, finalize `requirements.txt` |

## Notes

- Target model: Florence-2 or Moondream2 (lightweight, edge-deployable)
- Must run concurrently with YOLOv8 on same GPU without OOM errors
- Coordinate with Hardware Specialist on GPU memory allocation
