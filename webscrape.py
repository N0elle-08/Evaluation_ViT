from pathlib import Path
import requests
import shutil
from io import BytesIO
from PIL import Image, UnidentifiedImageError
import time

#Define search terms
term_1 = "rose"
term_2 = "dandelion"
term_3 = "sunflower"
term_4 = "daisy"
term_5 = "orchid"

search_terms = sorted([term_1, term_2, term_3, term_4, term_5])
search_terms = [x for x in search_terms if x.strip() != '']

#Search URL
SEARCH_URL = "https://huggingpics-api-server.fly.dev/images/search"

#Function to get image URLs by search term
def get_image_urls_by_term(search_term: str, count=10):
    params = {"q": search_term, "license": "public", "imageType": "photo", "count": count}
    response = requests.get(SEARCH_URL, params=params)
    response.raise_for_status()
    response_data = response.json()
    image_urls = [img['thumbnailUrl'] for img in response_data['value']]
    return image_urls

#fetch images from URLs 
def fetch_image(url, retries=3, backoff_factor=0.3):
    for i in range(retries):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
        except (requests.RequestException, UnidentifiedImageError) as e:
            if i < retries - 1:
                time.sleep(backoff_factor * (2 ** i))
            else:
                print(f"Failed to fetch image from {url}: {e}")
    return None

#Generator to yield images from URLs
def gen_images_from_urls(urls):
    num_skipped = 0
    for url in urls:
        img = fetch_image(url)
        if img is not None:
            yield img
        else:
            num_skipped += 1
    print(f"Retrieved {len(urls) - num_skipped} images. Skipped {num_skipped}.")

# Function to save images to a folder
def urls_to_image_folder(urls, save_directory):
    for i, image in enumerate(gen_images_from_urls(urls)):
        if image:
            image.save(save_directory / f'{i}.jpg')

#Directory to save images
data_dir = Path('images')
if data_dir.exists():
    shutil.rmtree(data_dir)

#Fetch and save images for each search term
for search_term in search_terms:
    search_term_dir = data_dir / search_term
    search_term_dir.mkdir(exist_ok=True, parents=True)
    urls = get_image_urls_by_term(search_term)
    print(f"Saving images of {search_term} to {str(search_term_dir)}...")
    urls_to_image_folder(urls, search_term_dir)


shutil.make_archive('images', 'zip', 'images')

