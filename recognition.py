import numpy as np
import sys
import tensorflow as tf
import time
import cv2
# Check command-line arguments

if len(sys.argv) != 3:
    sys.exit("Usage: python3 recognition.py model path/to/img.jpg")
model = tf.keras.models.load_model(sys.argv[1])

def main():
    img = cv2.imread(sys.argv[2])
    img = cv2.resize(img, (30, 30))

    classification = model.predict(
        [np.array(img).reshape(1, 30, 30, 3)]
    ).argmax()


    print(classification)

if __name__ == "__main__":
    main()
