# Basic Object Classifier
A program that uses the Tensorflow library to make a neural network for facial recognition/image classification.

## How to use
Run the following command to download the required libraries
```bash
pip3 install tensorflow opencv-python numpy scikit-learn
```

## Usage
Run the following command to make a model:
```bash
python3 train.py folder model.h5
```
And then the following command to have the program to classify the image:
```bash
python3 recognition.py model.h5 path/to/img.jpg
```

I have included two folders in the 'folder' directory -- one of which includes picures of cats and the other of dogs. The recogniition.py script will (with some probability) classify a particular image as either a cat or a dog.

All pictures have been taken from https://unsplash.com/
