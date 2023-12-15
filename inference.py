import argparse
import numpy as np
import torch
from mtcnn import MTCNN
from torchvision import transforms
import cv2
import os

from dataset import good_res_quality, resize_image, crop_face
from helper import predict
from model import BaseModel


def run_inference(path):
    """

    """
    print(f"Running inference on the data at: {path}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    face_detector = MTCNN()
    gender_mapping = {1: 'Female', 0: 'Male'}
    if path.lower().endswith(('.png', '.jpg')):
        face_image = crop_face(face_detector, path)

        if good_res_quality(face_image, threshold=18000):
            resized = resize_image(face_image)
        image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        model_age = BaseModel().to(device)
        model_age.load_state_dict(torch.load('./weights/Age_Model_BaseModel_Best.pt'))

        model_gender = BaseModel().to(device)
        model_gender.load_state_dict(torch.load('./weights/Gender_Model_BaseModel_Best.pt'))

        image = image.astype(np.float32)

        age_prediction = predict(model_age, image)
        gender_prediction = torch.sigmoid(predict(model_gender, image))

        print('Age of the person =', np.round(age_prediction.item()))
        print('Gender =', gender_mapping[int(np.round(gender_prediction.item()))])



def main():
    parser = argparse.ArgumentParser(description="Inference script")
    parser.add_argument("-p", "--path", type=str, required=True, help="Path to the data for inference")
    args = parser.parse_args()

    # Run the inference function with the provided path
    run_inference(args.path)

if __name__ == "__main__":
    main()