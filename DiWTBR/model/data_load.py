
from os import listdir
from os.path import join
from PIL import Image
import numpy as np
from torch.utils.data.dataset import Dataset
import random
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import zoom 

class DefocusData(Dataset):
    def __init__(self, trainset_input, trainset_output, transform, data_agu=True, is_training=True,crop1=768, crop2=1024):
        super(DefocusData, self).__init__()
        self.data_dir = trainset_input
        self.target_dir = trainset_output
        self.data_filenames = sorted([join(self.data_dir, x) for x in listdir(
            self.data_dir)])
        self.target_filenames = sorted([join(self.target_dir, x) for x in listdir(
            self.target_dir)])
        self.transform = transform
        self.data_agu = data_agu
        self.is_training = is_training
        self.crop1 = crop1
        self.crop2 = crop2

    def __getitem__(self, index):
        image = Image.open(self.data_filenames[index]).convert("RGB")
        target = Image.open(self.target_filenames[index]).convert("RGB")
        image_copy = image.copy()
        target_copy = target.copy()
        image_copy, target_copy = np.array(image_copy), np.array(target_copy)
        h, w = image_copy.shape[:2]
        zoom_factor = (self.crop1/h, self.crop2/w, 1)
        image_copy = zoom(image_copy, zoom_factor)
        target_copy = zoom(target_copy, zoom_factor)
            #print(image.shape)
        
        if self.data_agu:
            i = random.randint(0, 6)
            j = random.randint(0, 4)
            k = random.randint(0, 100)
            m = random.randint(0, 4)
            
            if self.is_training:
                if j == 0:
                    image_copy, target_copy =  np.flip(image_copy, axis=1), np.flip(target_copy, axis=1)
                elif j == 1:
                    image_copy, target_copy =   np.flip(image_copy, axis=0), np.flip(target_copy, axis=0)
                elif j == 2:
                    image_copy, target_copy = np.rot90(image_copy, 2), np.rot90(target_copy, 2)
                else:
                    image_copy, target_copy = image_copy, target_copy
           
            
        image_copy, target_copy = Image.fromarray(image_copy), Image.fromarray(target_copy)

        if self.transform:
            image_copy = self.transform(image_copy)
            target_copy = self.transform(target_copy)

        return [image_copy, target_copy,h,w]

    def __len__(self):
        return len(self.data_filenames)


class TestData(Dataset):
    def __init__(self, testset_input, testset_output, transform, data_agu=True, is_training=True,crop1=768, crop2=1024):
        super(TestData, self).__init__()
        self.data_dir = testset_input
        self.target_dir = testset_output
        self.data_filenames = sorted([join(self.data_dir, x) for x in listdir(
            self.data_dir)])
        self.target_filenames = sorted([join(self.target_dir, x) for x in listdir(
            self.target_dir)])
        self.transform = transform
        self.data_agu = data_agu
        self.is_training = is_training
        self.crop1 = crop1
        self.crop2 = crop2

    def __getitem__(self, index):
        image = Image.open(self.data_filenames[index]).convert("RGB")
        target = Image.open(self.target_filenames[index]).convert("RGB")
        image_copy = image.copy()
        target_copy = target.copy()
        image_copy, target_copy = np.array(image_copy), np.array(target_copy)
        # print(image_copy.shape)
        # print(target_copy.shape)
        h, w = image_copy.shape[:2]
        zoom_factor = (self.crop1/h, self.crop2/w, 1)
        image_copy = zoom(image_copy, zoom_factor)
        #target_copy = zoom(target_copy, zoom_factor)
        # image_copy = image_copy.resize((self.crop1, self.crop2), Image.LANCZOS)
        # target_copy = target_copy.resize((self.crop1, self.crop2), Image.LANCZOS)

        if self.transform:
            image_copy = self.transform(image_copy)
            target_copy = self.transform(target_copy)

        return [image_copy, target_copy,h,w]
    
    def __len__(self):
        return len(self.data_filenames)