import os
import requests


if __name__ == "__main__":

    if not os.path.exists("dataset"): os.makedirs("dataset")

    img_url = "https://cos1s.ntnu.edu.tw/AasEnrollStudent/RandImage"
    session = requests.Session()
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    for i in range(100, 200):
        r = session.get(img_url, headers=headers)
        with open(f"dataset/image_{i:05}.jpg", "wb") as f:
            for chunk in r: f.write(chunk)