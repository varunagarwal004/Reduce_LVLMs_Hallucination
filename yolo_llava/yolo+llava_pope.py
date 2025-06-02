# %%
# --- Colab Setup (run this cell first) ---
!pip install ultralytics tqdm transformers bitsandbytes accelerate jsonlines --quiet

# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
import os
import json
import random
from ultralytics import YOLO
from tqdm import tqdm
from PIL import Image
import torch
import jsonlines
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig


# %%
model_id = "llava-hf/llava-1.5-7b-hf"
processor = AutoProcessor.from_pretrained(model_id)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="cuda"
)

# %%
# --- CONFIG ---
POPE_IMG_ROOT = '/content/drive/MyDrive/UCLA/cs 263/final_proj/val2014/'  # Images are in the same directory level or update to your Drive path
POPE_ANN_PATH = '/content/drive/MyDrive/UCLA/cs 263/final_proj/coco_pope_adversarial (1).json'  # Your annotation file
RESULTS_FILE = '/content/drive/MyDrive/UCLA/cs 263/final_proj/pope_yolo_llava_results.json'
NUM_EXAMPLES = 500  # Number of unique images to process

# %%
# --- 1. Load POPE Annotation Data (grouped by image, random sample) ---
def load_pope_annotations_grouped(ann_path, img_root, num_examples=100):
    """
    Loads POPE annotations and groups them by image filename.
    Randomly samples up to num_examples unique images.
    Returns a list of dicts: {img_path, questions: [annotation_dicts]}
    """
    from collections import defaultdict
    images = []
    image_to_questions = defaultdict(list)
    with open(ann_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            img_fn = item['image']
            img_path = os.path.join(img_root, img_fn)
            if os.path.exists(img_path):
                print('loading:', img_path)
                image_to_questions[img_path].append(item)
    all_img_paths = list(image_to_questions.keys())
    random.shuffle(all_img_paths)
    all_img_paths = all_img_paths[:num_examples]
    for img_path in all_img_paths:
        images.append({'img_path': img_path, 'questions': image_to_questions[img_path]})
    return images

# %%
  images = load_pope_annotations_grouped(POPE_ANN_PATH, POPE_IMG_ROOT, NUM_EXAMPLES)
  print(f"Loaded {len(images)} unique images from POPE dataset.")

# %%
print(images[0])

# %%
# --- 2. Run YOLO Object Detection ---
def run_yolo_on_images(images, yolo_model_path='yolov8x.pt'):
    yolo_model = YOLO(yolo_model_path)
    results_dict = {}
    for entry in tqdm(images, desc='Running YOLO'):
        img_path = entry['img_path']
        try:
            results = yolo_model(img_path)
            detected_objects = [yolo_model.model.names[int(cls)] for cls in results[0].boxes.cls]
        except Exception as e:
            print(f"YOLO failed on {img_path}: {e}")
            detected_objects = []
        results_dict[img_path] = detected_objects
    return results_dict

# %%
  yolo_results = run_yolo_on_images(images)
  print("YOLO detection complete.")

# %%
#yolo_results

# %%
# --- 3. Construct Prompts for LLaVA ---
def construct_yolo_prompt(detected_objects, question):
    prompt = "<image>\n"
    prompt += f"USER: This is a list of detected objects: {', '.join(detected_objects)}.\n"
    prompt += f"Look at the image and answer the following question:\n- {question}\n"
    prompt += "\nASSISTANT:"
    return prompt
def construct_normal_prompt(question):
    prompt = "<image>\n"
    prompt += f"USER: Look at the image and answer the following question:\n- {question}\n"
    prompt += "\nASSISTANT:"
    return prompt

# %%
def run_llava_on_images_yolo(images, yolo_results):
    results = {}
    for entry in tqdm(images, desc='Running LLaVA'):
        img_path = entry['img_path']
        detected_objects = yolo_results.get(img_path, [])
        image = Image.open(img_path).convert("RGB")
        per_question_results = []
        #print(img_path)
        #print('Detected objects:', detected_objects)
        for q in entry['questions']:
            question = q['text']
            prompt = construct_yolo_prompt(detected_objects, question)
            #print(question)
            try:
                inputs = processor(prompt, image, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    output = model.generate(**inputs, max_new_tokens=128)
                description = processor.batch_decode(output, skip_special_tokens=True)[0]
                # split on ASSISTANT:
                if "ASSISTANT:" in description:
                    description = description.split("ASSISTANT:")[-1].strip()
                #print(description)
            except Exception as e:
                print(f"LLaVA failed on {img_path} for question '{question}': {e}")
                description = "[ERROR] LLaVA inference failed."
            per_question_results.append({
                'question': question,
                'llava_answer': description,
                'label': q.get('label', None)
            })
        results[img_path] = {
            'detected_objects': detected_objects,
            'questions': per_question_results
        }
        #print('-'*50)
    return results

# %%
def run_llava_on_images_normal(images):
    results = {}
    for entry in tqdm(images, desc='Running LLaVA'):
        img_path = entry['img_path']
        image = Image.open(img_path).convert("RGB")
        per_question_results = []
        #print(img_path)
        for q in entry['questions']:
            question = q['text']
            prompt = construct_normal_prompt(question)
            #print(question)
            try:
                inputs = processor(prompt, image, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    output = model.generate(**inputs, max_new_tokens=128)
                description = processor.batch_decode(output, skip_special_tokens=True)[0]
                # split on ASSISTANT:
                if "ASSISTANT:" in description:
                    description = description.split("ASSISTANT:")[-1].strip()
                #print(description)
            except Exception as e:
                print(f"LLaVA failed on {img_path} for question '{question}': {e}")
                description = "[ERROR] LLaVA inference failed."
            per_question_results.append({
                'question': question,
                'llava_answer': description,
                'label': q.get('label', None)
            })
        results[img_path] = {
            'questions': per_question_results
        }
        #print('-'*50)
    return results

# %%
llava_results = run_llava_on_images_yolo(images, yolo_results)
with open('/content/drive/MyDrive/UCLA/cs 263/final_proj/pope_yolo_llava_results.json', 'w') as f:
    json.dump(llava_results, f, indent=2)

# %%



# %%
llava_normal_results = run_llava_on_images_normal(images)
with open('/content/drive/MyDrive/UCLA/cs 263/final_proj/pope_normal_llava_results.json', 'w') as f:
    json.dump(llava_normal_results, f, indent=2)


# %%



# %%
import json
import re

# %%
def extract_yes_no_from_answer(answer):
    """
    Extracts a yes/no prediction from the LLaVA answer.
    Returns 'yes', 'no', or None if unclear.
    """
    answer = answer.lower()
    # Look for 'yes' or 'no' as a standalone word at the start or after punctuation
    if re.match(r'^(yes)[\\W_]*', answer):
        return 'yes'
    if re.match(r'^(no)[\\W_]*', answer):
        return 'no'
    # Fallback: look for yes/no anywhere
    if 'yes' in answer:
        return 'yes'
    if 'no' in answer:
        return 'no'
    return None

# %%
def evaluate_results(results_path):
    with open(results_path, 'r') as f:
        results = json.load(f)
    total = 0
    correct = 0
    for img_path, entry in results.items():
        for q in entry['questions']:
            label = q.get('label', '').strip().lower()
            answer = q.get('llava_answer', '')
            pred = extract_yes_no_from_answer(answer)
            if label in ['yes', 'no'] and pred is not None:
                total += 1
                if pred == label:
                    correct += 1
    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy: {accuracy:.3f} ({correct}/{total})")
    return accuracy


# %%
import json, re

# %%
def extract_yes_no_from_answer(answer):
    """
    Extracts a yes/no prediction from the LLaVA answer.
    Returns 'yes', 'no', or None if unclear.
    """
    answer = answer.lower()
    # Look for 'yes' or 'no' as a standalone word at the start or after punctuation
    if re.match(r'^(yes)[\\W_]*', answer):
        return 'yes'
    if re.match(r'^(no)[\\W_]*', answer):
        return 'no'
    # Fallback: look for yes/no anywhere
    if 'yes' in answer:
        return 'yes'
    if 'no' in answer:
        return 'no'
    return None

# %%
def evaluate_results(results_path):
    with open(results_path, 'r') as f:
        results = json.load(f)

    # Metrics counters
    metrics = {
        'total': 0,
        'correct': 0,
        'tp': 0,  # true positives (predicted yes, label yes)
        'tn': 0,  # true negatives (predicted no, label no)
        'fp': 0,  # false positives (predicted yes, label no)
        'fn': 0,  # false negatives (predicted no, label yes)
        'hallucinated': 0,
        'hallucination_total': 0
    }

    # Stopwords for hallucination filtering
    stopwords = {
        'the','a','an','is','are','was','were','it','this','that','and','or','of','to','in','on','at','for','with','as','by','from','yes','no','not','but','if','then','so','because','about','into','over','after','before','between','during','without','within','through','above','below','up','down','out','off','again','further','once'
    }

    for img_path, entry in results.items():
        detected_objects = set(obj.lower() for obj in entry.get('detected_objects', []))
        for q in entry['questions']:
            label = q.get('label', '').strip().lower()
            answer = q.get('llava_answer', '')
            pred = extract_yes_no_from_answer(answer)
            # Classification metrics
            if label in ['yes', 'no'] and pred is not None:
                metrics['total'] += 1
                if pred == label:
                    metrics['correct'] += 1
                if label == 'yes' and pred == 'yes':
                    metrics['tp'] += 1
                elif label == 'no' and pred == 'no':
                    metrics['tn'] += 1
                elif label == 'no' and pred == 'yes':
                    metrics['fp'] += 1
                elif label == 'yes' and pred == 'no':
                    metrics['fn'] += 1
            # Hallucination metrics
            if 'detected_objects' in entry:
                metrics['hallucination_total'] += 1
                words = set(re.findall(r'\b\w+\b', answer.lower()))
                content_words = words - stopwords
                if len(detected_objects) > 0 and any(w not in detected_objects for w in content_words):
                    metrics['hallucinated'] += 1

    # Calculate scores
    accuracy = metrics['correct'] / metrics['total'] if metrics['total'] > 0 else 0
    precision = metrics['tp'] / (metrics['tp'] + metrics['fp']) if (metrics['tp'] + metrics['fp']) > 0 else 0
    recall = metrics['tp'] / (metrics['tp'] + metrics['fn']) if (metrics['tp'] + metrics['fn']) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    hallucination_rate = metrics['hallucinated'] / metrics['hallucination_total'] if metrics['hallucination_total'] > 0 else 0
    print(metrics)

    # Print results
    #print(f"Accuracy: {accuracy:.3f} ({metrics['correct']}/{metrics['total']})")
    #print(f"Precision: {precision:.3f}")
    #print(f"Recall: {recall:.3f}")
    #print(f"F1 Score: {f1:.3f}")
    #print(f"Hallucination Rate: {hallucination_rate:.3f} ({metrics['hallucinated']}/{metrics['hallucination_total']})")
    return accuracy, precision, recall, f1, hallucination_rate

# %%
yolo_acc, yolo_prec, yolo_rec, yolo_f1, yolo_hall = evaluate_results('/content/drive/MyDrive/UCLA/cs 263/final_proj/pope_yolo_llava_results.json')
normal_acc, normal_prec, normal_rec, normal_f1, normal_hall = evaluate_results('/content/drive/MyDrive/UCLA/cs 263/final_proj/pope_normal_llava_results.json')

print(f'YOLO-grounded accuracy: {yolo_acc:.3f}')
print(f'YOLO-grounded precision: {yolo_prec:.3f}')
print(f'YOLO-grounded recall: {yolo_rec:.3f}')
print(f'YOLO-grounded F1: {yolo_f1:.3f}')
print(f'YOLO-grounded hallucination rate: {yolo_hall:.3f}')

print(f'Normal accuracy: {normal_acc:.3f}')
print(f'Normal precision: {normal_prec:.3f}')
print(f'Normal recall: {normal_rec:.3f}')
print(f'Normal F1: {normal_f1:.3f}')
print(f'Normal hallucination rate: {normal_hall:.3f}')

# %%
yolo_acc = evaluate_results('/content/drive/MyDrive/UCLA/cs 263/final_proj/pope_yolo_llava_results.json')
normal_acc = evaluate_results('/content/drive/MyDrive/UCLA/cs 263/final_proj/pope_normal_llava_results.json')
print(f'YOLO-grounded accuracy: {yolo_acc:.3f}')
print(f'Normal accuracy: {normal_acc:.3f}')

# %%



