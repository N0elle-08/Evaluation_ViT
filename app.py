import os
import shutil
import zipfile
import csv
from pathlib import Path
from PIL import Image
import gradio as gr
import numpy as np
from transformers import ViTForImageClassification, ViTFeatureExtractor, AutoImageProcessor
from datasets import Dataset
import evaluate
from transformers import TrainingArguments, Trainer

#Define the models and tasks
model_name_1 = "google/vit-base-patch16-224"
model_name_2 = "jonathanfernandes/vit-base-patch16-224-finetuned-flower"

#Initialize ViT model and ImageProcessor
model_1 = ViTForImageClassification.from_pretrained(model_name_1)
feature_extractor_1 = AutoImageProcessor.from_pretrained(model_name_1)

model_2 = ViTForImageClassification.from_pretrained(model_name_2)
feature_extractor_2 = AutoImageProcessor.from_pretrained(model_name_2)

def load_and_preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return image

def prepare_dataset_from_path(dataset_dir, feature_extractor, model):
    data = []
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                image_path = os.path.join(root, file)
                image = load_and_preprocess_image(image_path)
                label = os.path.basename(os.path.dirname(image_path))
                label_id = model.config.label2id.get(label, 0)  
                data.append((image_path, image, label_id))

    encodings = feature_extractor(images=[img for _, img, _ in data], return_tensors="pt")
    return Dataset.from_dict({
        "image": [img for _, img, _ in data],
        "pixel_values": encodings["pixel_values"],
        "labels": [lab for _, _, lab in data]
    })

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    eval_steps=10,
)

def evaluate_model(model, feature_extractor, dataset_dir):
    data = prepare_dataset_from_path(dataset_dir, feature_extractor, model)

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=data,
        compute_metrics=compute_metrics,
    )

    eval_results = trainer.evaluate()
    return eval_results

def save_results_to_csv(results, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Model", "Metric", "Value"])
        for model_name, metrics in results.items():
            for metric_name, value in metrics.items():
                writer.writerow([model_name, metric_name, value])
    return os.path.abspath(file_path)

def upload_and_evaluate(uploaded_files):
    #extract files from zip
    file = uploaded_files[0]

    temp_dir = Path('temp_dataset')
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(file.name, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    #Evaluate the dataset with models
    results = {}
    results[model_name_1] = evaluate_model(model_1, feature_extractor_1, temp_dir)
    results[model_name_2] = evaluate_model(model_2, feature_extractor_2, temp_dir)
    
    for model_name in results:
        if "eval_accuracy" in results[model_name]:
            results[model_name]["eval_accuracy"] = round(results[model_name]["eval_accuracy"] * 100, 2)  

    
    shutil.rmtree(temp_dir)

    #Save results to CSV
    csv_file_path = save_results_to_csv(results, "results", "eval_results.csv")
    formatted_results = {
        model_name_1: results[model_name_1]["eval_accuracy"],
        model_name_2: results[model_name_2]["eval_accuracy"]
    }
    return formatted_results, csv_file_path
 

#Gradio interface
interface = gr.Interface(
    fn=upload_and_evaluate,
    inputs=gr.Files(),
    outputs=["json", "file"],
    title="Image Classification - Evaluation with ViT",
    description="Upload a zip file containing images organized by labels"
)

if __name__ == "__main__":
    try:
        interface.launch(debug=True)
    except GradioError as e:
        print(f"Error launching interface: {e}")
    except KeyboardInterrupt:
        print("Shutting down Gradio interface...")
        interface.close()
