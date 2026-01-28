#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LaViC Code: Image Crawling (Category-Aware Version)
---------------------------------------------------
A helper script for downloading product images from URLs found
in JSON or JSONL metadata files. Primarily used to populate
train_images/ and valid_images/ directories with product images.

New Features:
-------------
- Takes category input via command line argument (--category)
- Dynamically loads item2meta_train_<category>.json and item2meta_valid_<category>.jsonl
- Saves images under category-specific folders:
      ../data/train_images/<category>/
      ../data/valid_images/<category>/

Usage Example:
--------------
python crawl_images.py --category office_products
"""

import os
import json
import argparse
import requests
from io import BytesIO
from PIL import Image


def load_json_data(file_name):
    """
    Load and return JSON data as a Python dictionary.

    Args:
        file_name (str): Path to the JSON file.

    Returns:
        dict: Deserialized data from the JSON file.
    """
    with open(file_name, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_jsonl_data(file_name):
    """
    Load and return a list of Python dict objects from a JSONL file.

    Each line in the file should contain a valid JSON object.

    Args:
        file_name (str): Path to the JSONL file.

    Returns:
        list of dict: Deserialized data for each line in the file.
    """
    data = []
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_image(url, path):
    """
    Download an image from a given URL and save it to the specified path.

    Args:
        url (str): URL of the image to download.
        path (str): Local file path where the image should be saved.

    Returns:
        bool: True if the image was saved successfully, False otherwise.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        image.save(path)
        return True
    except Exception as e:
        print(f"Failed to save image from {url}: {e}")
        return False


def download_images_json(item_data, folder_path):
    """
    Download all images from a JSON dictionary of items.

    The dictionary (item_data) maps item IDs to metadata. If 'images'
    is present, each 'large' field is downloaded.

    Args:
        item_data (dict): JSON dictionary containing item info, including 'images'.
        folder_path (str): Destination folder to save images.
    """
    os.makedirs(folder_path, exist_ok=True)

    count_exist = 0
    count_save = 0
    count_fail = 0

    for item_id, details in item_data.items():
        if 'images' in details:
            for index, img_obj in enumerate(details['images']):
                image_url = img_obj.get('large', '')
                if not image_url:
                    continue

                file_path = os.path.join(folder_path, f"{item_id}_{index}.jpg")

                if os.path.exists(file_path):
                    count_exist += 1
                else:
                    if save_image(image_url, file_path):
                        print(f"Saved: {file_path}")
                        count_save += 1
                    else:
                        print(f"Failed to save image for item: {item_id}")
                        count_fail += 1

    print(f"[JSON] Images already exist: {count_exist}, Newly saved: {count_save}, Failed: {count_fail}")


def download_images_jsonl(item_data, folder_path):
    """
    Download images from a JSONL list of items.

    Each element is a dict that may include 'image_name' and 'image'.

    Args:
        item_data (list of dict): List of items from a JSONL file.
        folder_path (str): Destination folder to save images.
    """
    os.makedirs(folder_path, exist_ok=True)

    count_exist = 0
    count_save = 0
    count_fail = 0

    for entry in item_data:
        image_url = entry.get('image', '')
        image_name = entry.get('image_name', '')

        if not image_url or not image_name:
            continue

        file_path = os.path.join(folder_path, image_name)

        if os.path.exists(file_path):
            print(f"Exists: {file_path}")
            count_exist += 1
        else:
            if save_image(image_url, file_path):
                print(f"Saved: {file_path}")
                count_save += 1
            else:
                print(f"Failed to save image for title: {entry.get('title','(unknown)')}")
                count_fail += 1

    print(f"[JSONL] Images already exist: {count_exist}, Newly saved: {count_save}, Failed: {count_fail}")


def main():
    """
    Main function to orchestrate image downloads for a given category.

    - Loads item2meta_train_<category>.json and downloads images to ../data/train_images/<category>
    - Loads item2meta_valid_<category>.jsonl and downloads images to ../data/valid_images/<category>
    """
    parser = argparse.ArgumentParser(description="Download product images for a given category.")
    parser.add_argument("--category", required=True, help="Category name (e.g., office_products)")
    args = parser.parse_args()

    category = args.category.strip()
    print(f"========== Starting image download for category: {category} ==========")

    # Construct paths dynamically based on category
    train_json_path = f"../data/item2meta_train_{category}.with_desc.json"
    valid_jsonl_path = f"../data/item2meta_valid_{category}.with_desc.jsonl"
    train_images_dir = os.path.join("../data/train_images", category)
    valid_images_dir = os.path.join("../data/valid_images", category)

    # 1) Download images for train data (JSON)
    if os.path.exists(train_json_path):
        print(f"Found training metadata: {train_json_path}")
        item_data = load_json_data(train_json_path)
        download_images_json(item_data, train_images_dir)
    else:
        print(f"No file found at {train_json_path}, skipping train images.")

    # 2) Download images for valid data (JSONL)
    if os.path.exists(valid_jsonl_path):
        print(f"Found validation metadata: {valid_jsonl_path}")
        item_data_list = load_jsonl_data(valid_jsonl_path)
        download_images_jsonl(item_data_list, valid_images_dir)
    else:
        print(f"No file found at {valid_jsonl_path}, skipping valid images.")

    print(f"========== Completed category: {category} ==========")


if __name__ == '__main__':
    main()
