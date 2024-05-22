import os
import shutil
import zipfile
import csv
from pathlib import Path
from PIL import Image
import gradio as gr
import numpy as np
import pandas as pd
from transformers import ViTForImageClassification, ViTFeatureExtractor, AutoImageProcessor
from datasets import Dataset
import evaluate
from transformers import TrainingArguments, Trainer
from sklearn.metrics import precision_recall_fscore_support

# Define the models
model_name_vit1 = "google/vit-base-patch16-224"
model_name_vit2 = "jonathanfernandes/vit-base-patch16-224-finetuned-flower"


# Initialize ViT and Vilt models and processors
model_vit1 = ViTForImageClassification.from_pretrained(model_name_vit1)
feature_extractor_vit1 = AutoImageProcessor.from_pretrained(model_name_vit1)

model_vit2 = ViTForImageClassification.from_pretrained(model_name_vit2)
feature_extractor_vit2 = AutoImageProcessor.from_pretrained(model_name_vit2)

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

def prepare_vilt_dataset_from_path(dataset_dir, processor, model):
    data = []
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                image_path = os.path.join(root, file)
                image = load_and_preprocess_image(image_path)
                label = os.path.basename(os.path.dirname(image_path))
                label_id = model.config.label2id.get(label, 0)
                data.append((image_path, image, label_id))

    encodings = processor(images=[img for _, img, _ in data], return_tensors="pt")
    return Dataset.from_dict({
        "image": [img for _, img, _ in data],
        "pixel_values": encodings["pixel_values"],
        "labels": [lab for _, _, lab in data]
    })

metric = evaluate.load("accuracy")

def compute_detailed_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = metric.compute(predictions=predictions, references=labels)["accuracy"]
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

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
        compute_metrics=compute_detailed_metrics,
    )

    eval_results = trainer.evaluate()
    return eval_results

def save_detailed_results_to_csv(results, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)
    
    # Extract models as rows and their keys as columns
    data = {}
    for model, metrics in results.items():
        data[model] = metrics

    # Convert to DataFrame
    df = pd.DataFrame.from_dict(data, orient='index')

    # Reset index to make "Model" a column
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Model'}, inplace=True)
    
    # Save DataFrame to CSV
    df.to_csv(file_path, index=False)
    return os.path.abspath(file_path)

def upload_and_evaluate(uploaded_files):
    #Extract files from zip
    file = uploaded_files[0]

    temp_dir = Path('temp_dataset')
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(file.name, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    #Evaluate the dataset with models
    results = {}
    results[model_name_vit1] = evaluate_model(model_vit1, feature_extractor_vit1, temp_dir)
    results[model_name_vit2] = evaluate_model(model_vit2, feature_extractor_vit2, temp_dir)
    
    for model_name in results:
        if "eval_accuracy" in results[model_name]:
            results[model_name]["eval_accuracy"] = round(results[model_name]["eval_accuracy"] * 100, 2)
    
    shutil.rmtree(temp_dir)

    #Save results to CSV
    csv_file_path = save_detailed_results_to_csv(results, "results", "eval_results.csv")
    formatted_results = {
        f'{model_name_vit1} Accuracy %': results[model_name_vit1]["eval_accuracy"],
        f'{model_name_vit2} Accuracy %': results[model_name_vit2]["eval_accuracy"]
    }
    return formatted_results, csv_file_path

#Gradio interface
interface = gr.Interface(
    fn=upload_and_evaluate,
    inputs=gr.Files(label="Upload Zip File"),
    outputs=[
        gr.JSON(label="Evaluation Results"),
        gr.File(label=" Download CSV Report")
    ],
    title="Image Classification - Evaluation of ViT models",
    description="Upload a zip file containing images organized by labels"
)

if __name__ == "__main__":
    try:
        interface.launch(debug=True)
    except Exception as e:
        print(f"Error launching interface: {e}")
    except KeyboardInterrupt:
        print("Shutting down Gradio interface...")
        interface.close()
