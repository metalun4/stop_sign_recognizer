import imutils
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import load_img, img_to_array


def plot_history(H):
    history = H.history

    plt.style.use("ggplot")
    plt.figure()

    plt.plot(np.arange(0, len(history['loss'])),
             history["loss"],
             label="train_loss")

    plt.plot(np.arange(0, len(history['loss'])),
             history["val_loss"],
             label="val_loss")

    plt.title("Bounding Box Regression Loss")
    plt.xlabel("Epoch №")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("stop_sign_model_loss_plot.png")


def show_predictions(img_paths, model):
    for img_path in img_paths:
        img = load_img(img_path, target_size=(224, 224))

        img = img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        # get the predictions
        preds = model.predict(img)[0]

        # load a test image again
        img = cv2.imread(img_path)
        img = imutils.resize(img, width=600)
        (h, w) = img.shape[:2]

        # scale the predicted bounding
        # box coordinates
        # based on the img dimensions
        preds[0] = int(preds[0] * w)
        preds[1] = int(preds[1] * h)
        preds[2] = int(preds[2] * w)
        preds[3] = int(preds[3] * h)

        # convert predictions to int
        preds = [int(p) for p in preds]

        # draw the predicted bounding box on the img
        cv2.rectangle(img,
                      (preds[0], preds[1]),
                      (preds[2], preds[3]),
                      (0, 255, 0), 3)

        # show the output img
        imgplot = plt.imshow(cv2.cvtColor(img,
                                          cv2.COLOR_BGR2RGB))
        plt.show()
