from pathlib import Path
import requests
import shutil
from PIL import Image, UnidentifiedImageError
from io import BytesIO
from transformers import AutoImageProcessor, ViTFeatureExtractor, ViTForImageClassification, ImageClassificationPipeline
import torch
import requests
import pandas as pd
import torch
from datasets import Dataset
from evaluate import evaluator
from transformers import TrainingArguments, Trainer
import evaluate
import numpy as np


term_1 = "rose" #@param {type:"string"}
term_2 = "dandelion" #@param {type:"string"}
term_3 = "sunflower" #@param {type:"string"}
term_4 = "daisy" #@param {type:"string"}
term_5 = "orchid" #@param {type:"string"}

search_terms = sorted([
    term_1,
    term_2,
    term_3,
    term_4,
    term_5
])

search_terms = [x for x in search_terms if x.strip() != '']


SEARCH_URL = "https://huggingpics-api-server.fly.dev/images/search"

def get_image_urls_by_term(search_term: str, count=10):
    params  = {"q": search_term, "license": "public", "imageType": "photo", "count": count}
    response = requests.get(SEARCH_URL, params=params)
    response.raise_for_status()
    response_data = response.json()
    image_urls = [img['thumbnailUrl'] for img in response_data['value']]
    return image_urls


def gen_images_from_urls(urls):
    num_skipped = 0
    for url in urls:
        response = requests.get(url)
        if not response.status_code == 200:
            num_skipped += 1
        try:
            img = Image.open(BytesIO(response.content))
            yield img
        except UnidentifiedImageError:
            num_skipped +=1

    print(f"Retrieved {len(urls) - num_skipped} images. Skipped {num_skipped}.")


def urls_to_image_folder(urls, save_directory):
    for i, image in enumerate(gen_images_from_urls(urls)):
        image.save(save_directory / f'{i}.jpg')