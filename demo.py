"""
Age estimation demo
Loads images from given folder and creates and saves results for each image in json format
Result ex:
{
  raw_probas:[],
  raw_logits:[],
  threshold:float,
  thresholded_probas:[],
  result_value:int
}
result_value : final estimated age
"""

import glob
import cv2
import argparse
import os

from utils.utils import load_custom_model, save_log_file, read_config_file


def process_image(image, height, width):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = img / 255
    img = cv2.resize(img, (height, width))
    img = img.reshape((1,) + img.shape)

    return img


def main(path_to_test_imgs):
    conf = read_config_file("configs/predict_config.json")
    model = load_custom_model(conf["model_path"], conf["n_classes"])

    if not os.path.exists("results"):
        os.mkdir("results")

    for path in glob.glob(path_to_test_imgs + "/*"):
        img = cv2.imread(path)
        name = os.path.basename(path)
        im_tensor = process_image(img, conf["img_size"], conf["img_size"])

        probas, logits = model.predict(im_tensor)
        save_log_file(name, conf["threshold"], probas, logits, conf["start_label"])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--test_imgs_path', type=str, required=True)

    config = parser.parse_args()

    main(config.test_imgs_path)
