from os import listdir
from os.path import join
from PIL import Image
import numpy as np
from torch.utils.data.dataset import Dataset
from scipy.ndimage import zoom

class DefocusData(Dataset):
    def __init__(self, dataset_dir, transform, data_agu=True, is_training=True,crop1=768, crop2=1024):
        super(DefocusData, self).__init__()
        self.data_dir = '/mnt/data0/qiuxs/EBB/train/original'
        self.target_dir = '/mnt/data0/qiuxs/EBB/train/boken'
        self.data_filenames = sorted([join(self.data_dir, x) for x in listdir(
            self.data_dir)])
        self.target_filenames = sorted([join(self.target_dir, x) for x in listdir(
            self.target_dir)])
        self.transform = transform
        self.data_agu = data_agu
        self.is_training = is_training
        self.crop1 = crop1
        self.crop2 = crop2
        #self.indices = random.sample(range(len(self.data_filenames)), 1000)
        self.count = 0

    def __getitem__(self, index):
        self.count += 1
        # print(self.count)
        #image_idx = self.indices[index]
        #img_file = self.data_filenames[image_idx] 
        #print("Reading image:", img_file)
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
        
        image_copy, target_copy = Image.fromarray(image_copy), Image.fromarray(target_copy)

        if self.transform:
            image_copy = self.transform(image_copy)
            target_copy = self.transform(target_copy)
        #print(image.shape)

        return [image_copy, target_copy,h,w]

    def __len__(self):
        return len(self.data_filenames)
        # return 1000
    
class TestData(Dataset):
    def __init__(self, dataset_dir, transform, data_agu=True, is_training=True,crop1=768, crop2=1024):
        super(TestData, self).__init__()
        self.data_dir = '/mnt/data0/qiuxs/EBB/test/original'
        self.target_dir = '/mnt/data0/qiuxs/EBB/test/bokeh'
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
        zoom_factor1 = (self.crop1/h*2, self.crop2/w*2, 1)
        image_copy = zoom(image_copy, zoom_factor)
        # target_copy = zoom(target_copy, zoom_factor1)

        if self.transform:
            image_copy = self.transform(image_copy)
            target_copy = self.transform(target_copy)

        return [image_copy, target_copy,h,w]
    
    def __len__(self):
        return len(self.data_filenames)