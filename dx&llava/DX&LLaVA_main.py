# dds cloudapi for DINO-X
from dds_cloudapi_sdk import Config
from dds_cloudapi_sdk import Client
from dds_cloudapi_sdk.tasks.v2_task import V2Task, create_task_with_local_image_auto_resize

# using supervision for visualization
import os
import cv2
import numpy as np
import supervision as sv
from pathlib import Path
from pycocotools import mask as mask_utils
from rle_util import rle_to_array

from transformers import pipeline
import torch

from collections import Counter

import matplotlib.pyplot as plt

import gradio as gr
import getpass

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

model_id = "llava-hf/llava-1.5-7b-hf"

pipe = pipeline(
    task="image-text-to-text",
    model=model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

"""
Hyper Parameters
"""
API_TOKEN = getpass.getpass("Enter your API key: ")


#IMG_PATH = "./assets/demo.png" # Example image
#TEXT_PROMPT = "Is there a mouse and a car in the picture"
OUTPUT_DIR = Path("./outputs/prompt_free_detection_segmentation")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_description_prompt(data):
    category_counts = Counter(obj['category'] for obj in data)
    sorted_categories = sorted(category_counts.items())

    phrases = []
    for category, count in sorted_categories:
        name = category if count == 1 else category + 's'
        phrases.append(f"{count} {name}")

    if len(phrases) == 1:
        return f"There is {phrases[0]} in the picture."
    else:
        return f"There are {', '.join(phrases[:-1])}, and {phrases[-1]} in the picture."

def DINO_X_API(API_TOKEN, IMG_PATH, OUTPUT_DIR, Visualization=False, return_image=False):
    
    # Step 1: Prepare paths
    img_path = Path(IMG_PATH)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_stem = img_path.stem  # e.g., "dog_cat"
    output_image_path = output_dir / f"{image_stem}_annotated.jpg"

    # Step 2: Set up config and client
    config = Config(API_TOKEN)
    client = Client(config)

    # Step 3: Run detection task with local image
    v2_task = create_task_with_local_image_auto_resize(
        api_path="/v2/task/dinox/detection",
        api_body_without_image={
            "model": "DINO-X-1.0",
            "prompt": {
                "type": "universal"
            },
            "targets": ["bbox", "mask"],
            "bbox_threshold": 0.25,
            "iou_threshold": 0.8
        },
        image_path=str(img_path)
    )

    client.run_task(v2_task)
    objects = v2_task.result["objects"]

    # Step 4: Optional visualization
    annotated_img = None
    if Visualization and objects:
        class_names = sorted(set(obj["category"].lower().strip() for obj in objects))
        class_name_to_id = {name: idx for idx, name in enumerate(class_names)}

        boxes = [obj["bbox"] for obj in objects]
        masks = [
            rle_to_array(obj["mask"]["counts"], obj["mask"]["size"][0] * obj["mask"]["size"][1])
            .reshape(obj["mask"]["size"])
            for obj in objects
        ]
        confidences = [obj["score"] for obj in objects]
        class_ids = [class_name_to_id[obj["category"].lower().strip()] for obj in objects]

        labels = [
            f"{obj['category']} {score:.2f}" for obj, score in zip(objects, confidences)
        ]

        img = cv2.imread(str(img_path))
        detections = sv.Detections(
            xyxy=np.array(boxes),
            mask=np.array(masks).astype(bool),
            class_id=np.array(class_ids),
        )

        box_annotator = sv.BoxAnnotator()
        annotated = box_annotator.annotate(scene=img.copy(), detections=detections)

        label_annotator = sv.LabelAnnotator()
        annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

        if return_image:
            annotated_img = annotated
        else:
            # Only save if not returning image
            output_image_path = output_dir / f"{image_stem}_annotated.jpg"
            cv2.imwrite(str(output_image_path), annotated)

    # Step 5: Return prompt and optionally the annotated image
    if return_image:
        return generate_description_prompt(objects), annotated_img
    return generate_description_prompt(objects)


def llama_dino_pipeline(
    API_TOKEN, 
    IMG_PATH, 
    OUTPUT_DIR, 
    Visualization=False, 
    Dino_activated=True,
    original_prompt=None,
    max_new_tokens=60, 
    return_full_text=False):
    
    
    if Dino_activated:
        enhancer = DINO_X_API(API_TOKEN, IMG_PATH, OUTPUT_DIR, Visualization)

    enhanced_prompt = "From the object detector, we can see that, " + enhancer + " So based on the given information from the object detector and the orginal image, " + original_prompt
    
    original_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": IMG_PATH,
                },
                {"type": "text", "text": original_prompt},
            ],
        }
    ]

    enhanced_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": IMG_PATH,
                },
                {"type": "text", "text": enhanced_prompt},
            ],
        }
    ]

    if Dino_activated:
        messages = enhanced_messages
    else:
        messages = original_messages

    outputs = pipe(text=messages, 
                    max_new_tokens=max_new_tokens, 
                       return_full_text=return_full_text,
                   )
    
    return messages, outputs[0]["generated_text"]


title = "DINO-LLaVA: Visual Question Answering with Object Detection"

def process_image(question, image_path):
    try:
        # Run DINO once to get detection and visualization
        dino_output, annotated_img = DINO_X_API(API_TOKEN, image_path, OUTPUT_DIR, Visualization=True, return_image=True)
        
        # Convert BGR to RGB for display
        annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        
        # Create enhanced prompt
        enhanced_prompt = f"From the object detector, we can see that, {dino_output} So based on the given information from the object detector and the orginal image, {question}"
        
        # Prepare messages for original prompt
        original_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]
        
        # Prepare messages for enhanced prompt
        enhanced_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": enhanced_prompt},
                ],
            }
        ]
        
        # Run LLaVA pipeline with original prompt
        original_outputs = pipe(
            text=original_messages,
            max_new_tokens=60,
            return_full_text=False
        )
        
        # Run LLaVA pipeline with enhanced prompt
        enhanced_outputs = pipe(
            text=enhanced_messages,
            max_new_tokens=60,
            return_full_text=False
        )
        
        return (
            annotated_img_rgb,  # DINO output image
            dino_output, 
            question,  # Original prompt
            original_outputs[0]["generated_text"],  # Original response
            enhanced_prompt,  # Enhanced prompt
            enhanced_outputs[0]["generated_text"]  # Enhanced response
        )
    except Exception as e:
        error_message = f"Error occurred: {str(e)}"
        return None, error_message, error_message, error_message, error_message, error_message

# Create the Gradio interface
with gr.Blocks(title=title) as demo:
    gr.Markdown(
        """
        <div style="text-align: center; max-width: 700px; margin: 0 auto;">
            <div>
                <h1>DINO-LLaVA: Visual Question Answering with Object Detection</h1>
                <h4> Comparing original QA with enhanced prompt to reduce hallucination</h4>
                <h4> A UCLA CS263 Project</h4>
            </div>
        </div>
        """
    )
    
    
    with gr.Row():
        with gr.Column():
            input_question = gr.Textbox(
                lines=2,
                placeholder="Ask a question about the image...",
                label="Question"
            )
            input_image = gr.Image(type="filepath", label="Input Image", height=400)
            submit_btn = gr.Button("Analyze Image")
        
        with gr.Column():
            dino_image = gr.Image(label="DINO Detection Visualization", height=400)
            dino_output = gr.Textbox(label="DINO Object Detection Output")
         
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Original Prompt")
            original_prompt = gr.Textbox(label="Input Prompt")
            original_response = gr.Textbox(label="LLaVA Response")
        
        with gr.Column():
            gr.Markdown("### Enhanced Prompt")
            enhanced_prompt = gr.Textbox(label="Enhanced Prompt with DINO")
            enhanced_response = gr.Textbox(label="LLaVA Response")
    
    submit_btn.click(
        fn=process_image,
        inputs=[input_question, input_image],
        outputs=[
            dino_image,
            dino_output,
            original_prompt,
            original_response,
            enhanced_prompt,
            enhanced_response
        ]
    )

    gr.Markdown(
        """ 
        
        """
    )


if __name__ == "__main__":
    demo.queue()    # Queue before launch is good, but putting it here is also fine
    demo.launch(share=True)
