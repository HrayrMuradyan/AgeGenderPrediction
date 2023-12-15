import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt


def show_image(image):
    if os.path.isfile(image):
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.show()

class TorchDatasetCreator(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset[0])
    
    def __getitem__(self, idx):
        image = self.dataset[0][idx]
        image = transforms.ToTensor()(image)
        age = self.dataset[1][idx]
        gender = self.dataset[2][idx]

        return image, age, gender


def extract_labels(images_path):
    data = pd.DataFrame(columns=['Age', 'Gender', 'Race'])
    for index, path in enumerate(images_path):
        annotation = path.split('\\')[-1]
        info = annotation.split('_')
        data.loc[index, 'Age'] = info[0]
        data.loc[index, 'Gender'] = info[1]
        data.loc[index, 'Race'] = info[2]
    return data


def crop_face(detector, image_path, add_margin = 0.0, output_path='', save=False):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(img_rgb)
    if len(faces)!=1:
        return -1
    img_height, img_width, _ = img.shape
        
    for i, face in enumerate(faces):
        x, y, width, height = face['box']
        margin_h = int(height*add_margin)
        margin_w = int(width*add_margin)
        lower_bound_x = max(y-margin_h, 0)
        upper_bound_x = min(y+height+margin_h, img_height)
        lower_bound_y = max(x-margin_w, 0)
        upper_bound_y = min(x+width+margin_w, img_width)
        x, y = max(0, x), max(0, y)
        cropped_face = img[lower_bound_x:upper_bound_x, lower_bound_y:upper_bound_y]
        
    if save:
        cv2.imwrite(output_path, cropped_face)
        print('Saved', output_path)
        return
        
    return cropped_face

def crop_faces(path_images, detector, add_margin = 0.0, dir_to_save='./data/face_cropped/', save=False):
    if save:
        if not os.path.exists(dir_to_save):
            os.makedirs(dir_to_save) 
    
    for index, path in enumerate(path_images):
        if index % 100 == 0:
            print('-'*10 + str(index) + '-'*10)
        path_to_save = dir_to_save + path[6:-4] + '_cropped' + path[-4:]
        if os.path.exists(path_to_save):
            continue
        try:
            crop_face(detector, path, output_path=path_to_save, save=save)
        except:
            print(path)

def resize_image(image, to_size=(224, 224), fill_color=(255, 255, 255)):
    height, width = image.shape[:2]

    ratio = min(to_size[0] / width, to_size[1] / height)
    new_size = (int(width * ratio), int(height * ratio))

    img = cv2.resize(image, (new_size[0], new_size[1]), interpolation=cv2.INTER_CUBIC)

    new_img = np.full((to_size[1], to_size[0], 3), fill_color, dtype=np.uint8)

    x_center = (to_size[0] - new_size[0]) // 2
    y_center = (to_size[1] - new_size[1]) // 2

    new_img[y_center:y_center+new_size[1], x_center:x_center+new_size[0]] = img

    return new_img


def good_res_quality(image, threshold=15000):
    image_h, image_w, _ = image.shape
    if image_h*image_w <= threshold:
        return False
    return True



def resize_filter_images(images_path, to_size=(224, 224), fill_color=(255, 255, 255), threshold=15000, dir_to_save='./data/resized_cropped/', save=False):
    if save:
        if not os.path.exists(dir_to_save):
            os.makedirs(dir_to_save) 
    for index, path in enumerate(images_path):
        image = cv2.imread(path)
        if not good_res_quality(image, threshold=threshold):
            continue
        resized = resize_image(image)
        img_name = path.split('\\')[-1]
        cv2.imwrite(dir_to_save + img_name[:-4] + '_resized' + img_name[-4:], resized)
        if index % 100 == 0:
            print(f'{index} - Done!')



def save_in_np(images_path, batch_size=5000, path_to_save='./data/resized_data_path/'):
    n_batches = int(np.ceil(len(images_path)/batch_size))
    for batch in range(n_batches):
        container = []
        for path in images_path[batch*batch_size:(batch+1)*batch_size]:
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.expand_dims(image, 0)
            container.append(image)
        np_container = np.concatenate(container, axis=0)
        np.save(path_to_save + f'resized_cropped_np_{batch_size}_' + str(batch), np_container.astype(np.uint8))


def train_val_test_split(len_data, val_size=0.1, test_size=0):
    indices = np.arange(len_data)
    np.random.shuffle(indices)
    train_size = int((1-val_size-test_size)*len_data)
    test_size = int(test_size*len_data)
    
    train_indices = indices[:train_size]  # 
    test_indices = indices[train_size:train_size+test_size]
    val_indices = indices[train_size+test_size:]
    
    return train_indices, val_indices, test_indices