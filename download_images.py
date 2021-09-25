The_amount_of_validate_code_images_you_wish_to_download = 500


# ================================================== #


import os
import requests


def download_validate_code_images():
    if not os.path.exists("dataset"): os.makedirs("dataset")
    img_url = "https://cos1s.ntnu.edu.tw/AasEnrollStudent/RandImage"
    session = requests.Session()
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    for i in range(The_amount_of_validate_code_images_you_wish_to_download):
        r = session.get(img_url, headers=headers)
        with open(f"dataset/image_{i:05}.jpg", "wb") as f:
            for chunk in r: f.write(chunk)


if __name__ == "__main__":
    download_validate_code_images()