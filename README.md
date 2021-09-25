# NTNU-Validate-Code-AI

## Prepare for the Dataset
For downloading new validate code images, you can execute the program `download_images.py`.
It will download validate code images from the [NTNU course taking website](https://cos1s.ntnu.edu.tw/AasEnrollStudent/LoginCheckCtrl?language=TW) by sending GET requests to `https://cos1s.ntnu.edu.tw/AasEnrollStudent/RandImage`.
To change how many validate code images to download, just edit the variable in `download_images.py`.
After download validate code images from the website, you will need to label them by yourself.
For this task, I wrote a simple program `labeling.py`.

## Existing Dataset
[These](https://drive.google.com/file/d/15Iw5rXws4rhuizP7hrgDtCEuR7TT3PP_/view?usp=sharing) are 600 validate code images & labels I used.
Feel free to download and use them.

## The best model I have trained: 94.99% validation accuracy
The architecture of the best model I found is in the file `best_model.py`.
To use it, you can execute the program `predict.py`.
However, you will need the weights file (`.h5` file) that I have trained, which is at [here](https://drive.google.com/file/d/1qdB1SECI-cwqbUQNbJ834EcRAX07i4Z5/view?usp=sharing).
Remember to edit the weights file (`.h5` file) path in `best_model.py` to where you put my pretrained weights file.
