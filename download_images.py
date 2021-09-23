import os
import requests


def download_image_v1():
    if not os.path.exists("dataset"): os.makedirs("dataset")
    img_url = "https://cos1s.ntnu.edu.tw/AasEnrollStudent/RandImage"
    session = requests.Session()
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    for i in range(500):
        r = session.get(img_url, headers=headers)
        with open(f"dataset/image_{i:05}.jpg", "wb") as f:
            for chunk in r: f.write(chunk)


# def download_image_v2():
#     pass


if __name__ == "__main__":
    download_image_v1()
    # download_image_v2()