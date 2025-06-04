[![Open in Github](assets\github-mark.svg)](https://github.com/AndyGongDS/Reduce_LVLMs_Hallucination)

# Reducing Hallucinations in Large Vision-Language Models

This research project explores techniques to reduce hallucinations in Large Vision-Language Models (LVLMs) through two main approaches:

1. **Object Detection Integration**: Using explicit object detection to ground LVLMs in visual reality
2. **Language-Based Methods**: Implementing reasoning techniques to improve reliability

## Project Overview

Large Vision-Language Models have shown impressive capabilities in multimodal tasks but often suffer from hallucinations - generating facts or objects not present in the images they analyze. This project implements and evaluates multiple approaches to reduce these hallucinations.

## Approaches

### Object Detection Integration

#### DINO-X & LLaVA (`dx_llava/`)

This module integrates DINO-X (a state-of-the-art object detection model) with LLaVA to enhance visual question answering:

- Detects objects in images using DINO-X API
- Uses detection results to enhance question prompts
- Provides a Gradio web interface for interactive demonstrations
- Evaluated using the POPE benchmark

#### YOLO-Grounded LLaVA (`yolo_llava/`)

This module combines YOLOv8 with LLaVA 1.5 to provide explicit object grounding:

- Uses YOLOv8x for object detection
- Incorporates detected objects in prompts to LLaVA
- Demonstrates significant improvements in accuracy (+6.5%), precision (+9.3%), and F1 score (+6.1%)
- Evaluated on the POPE dataset for object hallucination testing

### Language-Based Methods (`language_methods/`)

This module explores reasoning techniques to improve model reliability:

- **Chain of Thought (CoT)**: Guides the model through step-by-step reasoning processes
- **Self-Verification (SV)**: Enables models to verify and correct their own responses

The module supports multiple LVLMs:

- LLaVA
- OpenAI's multimodal models
- Google's Gemma 3
