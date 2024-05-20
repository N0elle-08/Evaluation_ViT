# ViT_Evaluation
Evaluation system

Screenshots
![image](https://github.com/N0elle-08/Evaluation_ViT/assets/124637016/0bb45df0-7451-4aea-9410-59c9430deeec)

![image](https://github.com/N0elle-08/Evaluation_ViT/assets/124637016/60826367-4ce8-4ae9-940c-523f298166b3)

Overview
This project implements an image classification evaluation system using Vision Transformers (ViT). It consists of two main components:

Image Downloader (webscrape.py): This script downloads images from the web based on predefined search terms. The downloaded images are organized into folders corresponding to their respective search terms.
Gradio App (app.py): This is a Gradio app that allows users to upload a zip file containing images organized by labels. The app then evaluates the uploaded dataset using two pre-trained ViT models and provides the evaluation results as output along with a CSV file.

How to Run the App
Follow these steps to run the evaluation system:

Step 1: Clone the Repository
Clone this repository to your local machine using the following command:
>> git clone https://github.com/N0elle-08/Evaluation_ViT.git

Step 2: Install Dependencies
Navigate to the project directory and install the required dependencies:
>> cd Evaluation_ViT
>> 
>>pip install -r requirements.txt

Step 3: Run the Image Downloader
Run the image downloader script to download images based on predefined search terms:
>>python webscrape.py

This will create a folder named images containing subfolders for each search term, with downloaded images inside.

Step 4: Launch the Gradio App
Run the Gradio app script to start the web interface for evaluating image datasets:
>> python app.py

Follow the link generated.



