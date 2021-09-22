for i in range(100, 200):
    while True:
        label = input(f"Enter label of image_{i:05}.jpg: ")
        if len(label) == 3: label += "="
        if len(label) == 4: break
        print("Input error!")
    with open(f"dataset/label_{i:05}.txt", "w") as txt_file:
        txt_file.write(label)