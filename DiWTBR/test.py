import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from model.DiWTBR_S import DiWTBR
import numpy as np
from torch.utils.data.dataset import Dataset
from os import listdir
from os.path import join, basename, splitext
from PIL import Image
from scipy.ndimage import zoom
import torch.nn as nn
import argparse

parser = argparse.ArgumentParser(description='DiWTBR Evaluation')
parser.add_argument('--checkpoint', type=str, default='./Checkpoint/S_EBB.pth', 
                    help='Path to model checkpoint file')
parser.add_argument('--test_datadir', type=str, default='/mnt/data0/qiuxs/EBB/test/original/', 
                    help='Path to test data directory')
parser.add_argument('--result_dir', type=str, default='./result/EBB-S/', 
                    help='Path to save result directory')
parser.add_argument('--gpu', type=str, default='0', 
                    help='GPU device ID')
opt = parser.parse_args()

class TestData(Dataset):
    def __init__(self, dataset_dir, transform, data_agu=True, is_training=True, crop1=768, crop2=1024):
        super(TestData, self).__init__()
        self.data_dir = dataset_dir
        self.data_filenames = sorted([join(self.data_dir, x) for x in listdir(
            self.data_dir)])
        self.transform = transform
        self.data_agu = data_agu
        self.crop1 = crop1
        self.crop2 = crop2

    def __getitem__(self, index):
        image = Image.open(self.data_filenames[index]).convert("RGB")
        image_copy = image.copy()
        image_copy = np.array(image_copy)
        h, w = image_copy.shape[:2]
        zoom_factor = (self.crop1/h, self.crop2/w, 1)
        image_copy = zoom(image_copy, zoom_factor)

        if self.transform:
            image_copy = self.transform(image_copy)

        filename = basename(self.data_filenames[index])
        return [image_copy, h, w, filename]
    
    def __len__(self):
        return len(self.data_filenames)

def evaluate(model):
    print("save defocus image")
    transform1 = transforms.Compose([
        transforms.ToTensor()
    ])
    test_set = TestData(opt.test_data, transform1,  data_agu=False, is_training=False, crop1=768, crop2=1024)
    test_data_loader = DataLoader(dataset=test_set, num_workers=3, batch_size=1, shuffle=False)
    result_folder = opt.result_dir
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    with torch.no_grad():
        #model.eval()
        for iteration, batch in enumerate(test_data_loader, 1):
            image= batch[0]
            label_h, label_w = batch[1][0],batch[2][0]
            filename = batch[3][0]
            image = image.cuda()
            defocus = model(image)
            defocus = defocus.cpu()
            defocus = F.interpolate(defocus, size=(label_h, label_w), mode='bilinear', align_corners=True)
            defocus = defocus.squeeze().clamp(0, 1)
            defocus = transforms.ToPILImage()(defocus)

            original_name = splitext(filename)[0]
            defocus.save('{}/{}.jpg'.format(result_folder, original_name))

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
model = DiWTBR()
# model = nn.DataParallel(model) ##Use this line of code when there is "module" in the ckpt dict
model = model.cuda()
checkpoint = torch.load(opt.checkpoint)
model.load_state_dict(checkpoint, strict=True)  
model.eval()
evaluate(model)