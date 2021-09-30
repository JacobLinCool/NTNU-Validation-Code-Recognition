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

## The best model I have trained: 100% validation accuracy
The architecture of the best model I found is in the file `best_model.py`.

To use it, you can try understanding how the program `predict.py` works, and do some adjustments to fit your need.

However, you will need the weights file (`val_loss.h5`) that I have trained, which is at [here](https://drive.google.com/file/d/16YL-915VVvY0bSMr2FiKhVnV19ipYF59/view?usp=sharing).

Remember to edit the weights file (`val_loss.h5`) path in the first line of `best_model.py` to where you put the pretrained weights file.
