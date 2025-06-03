# YOLO-Grounded LLaVA: Reducing Hallucinations in Large Vision-Language Models

This project explores how explicit object detection can help reduce hallucinations in Large Vision-Language Models (LVLMs). We integrate YOLOv8, a state-of-the-art object detection model, with LLaVA 1.5 to provide grounding information that constrains the LVLM's responses to objects actually present in images.

## Project Overview

Large Vision-Language Models (LVLMs) like LLaVA have demonstrated impressive capabilities in understanding and describing images. However, they often "hallucinate" objects not present in images or provide incorrect information about scene contents.

This project tests a simple yet effective approach to mitigate hallucinations:

1. Use YOLOv8 to detect objects in images
2. Provide the list of detected objects as part of the prompt to LLaVA
3. Measure whether this explicit grounding reduces hallucinations

## Methodology

### Data

We use the POPE (Prompt Object Presence Evaluation) dataset, which contains yes/no questions about objects in COCO images, specifically designed to evaluate object hallucination.

### Models

- **Object Detection**: YOLOv8x (Ultralytics)
- **Vision-Language Model**: LLaVA 1.5 7B

### Process

1. Run YOLOv8 on each image to detect objects
2. For each image, two prompting strategies are compared:
   - **Normal Prompt**: Standard question-answering without object grounding
   - **YOLO-Grounded Prompt**: Question-answering with a list of detected objects

### Evaluation

We evaluate on key metrics:

- Accuracy (in answering yes/no questions)
- Precision, Recall, F1 Score
- Hallucination Rate (percentage of responses mentioning objects not detected by YOLO)

## Results

| Metric             | YOLO-Grounded LLaVA | Standard LLaVA |
| ------------------ | ------------------- | -------------- |
| Accuracy           | 0.885               | 0.820          |
| Precision          | 0.909               | 0.816          |
| Recall             | 0.856               | 0.826          |
| F1 Score           | 0.882               | 0.821          |
| Hallucination Rate | 0.394               | N/A            |

The results demonstrate that providing explicit object grounding significantly improves accuracy (+6.5%), precision (+9.3%), and overall F1 score (+6.1%) on the POPE dataset. While the hallucination rate is still non-zero, this approach provides a simple way to make LVLMs more reliable in visual question answering.

## Usage

The implementation is provided as Python scripts and Jupyter notebooks:

- `yolo_llava/yolo+llava_pope.py`: Main implementation
- `yolo_llava/YOLO_+_LLAVA_pope.ipynb`: Notebook with examples and results

### Requirements

```
ultralytics
tqdm
transformers
bitsandbytes
accelerate
jsonlines
torch
```

### Running the Code

1. Install the required dependencies
2. Set up paths to your POPE dataset and images
3. Run the notebook or script to evaluate the models

## Conclusion

This project demonstrates that explicit object grounding can substantially improve the reliability of Large Vision-Language Models. By simply providing a list of detected objects in the prompt, we can reduce hallucinations and improve accuracy on visual question answering tasks.

Future work could explore more sophisticated integration of object detection and vision-language models, potentially with end-to-end training or more structured grounding approaches.
