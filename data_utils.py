import os
import os.path as osp

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch



class SAR_DatasetLoader(Dataset):

    def __init__(self, setname, data_path):
        THE_PATH = data_path

        label_dict = {'2S1': 0, 'BMP2': 1, 'BRDM_2': 2, 'BTR70(SN_C71)': 3, 'BTR70': 3, 'BTR_60': 4, 'D7': 5,
                      'T62': 6, 'T72(SN_132)': 7, 'T72': 7, 'ZIL131': 8, 'ZSU_23_4': 9}
                      
        label_list = os.listdir(THE_PATH)

        data = []
        label = []

        if setname == 'OOD':
            for labelname in label_list:
                this_folder = osp.join(data_path, labelname)
                this_folder_images = os.listdir(this_folder)
                for image_path in this_folder_images:
                    data.append(osp.join(this_folder, image_path))
                    label.append(1000)

        skip_list=['SN_812', 'SN_S7', 'SN_9566', 'SN_C21']
        if setname in ['train', 'test', 'val']:
            for root, dirs, files in os.walk(data_path):
                # 跳过不想要的子目录
                if any(skip_name in root for skip_name in skip_list):
                    continue

                for file in files:
                    if os.path.splitext(file)[1].lower() in ['.jpeg', '.tif', '.png']:
                        image_path = os.path.join(root, file)
                        data.append(image_path)

                        label_assigned = False
                        for key, value in label_dict.items():
                            if key in root:
                                label.append(value)
                                label_assigned = True
                                break
                        if not label_assigned:
                            print(f"Warning: No label assigned for {image_path}")
        self.data = data
        self.label = label
        self.num_class = len(set(label))

        # Transformation
        image_size = 90
        self.transform = transforms.Compose([
            transforms.Grayscale(3),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label


def build_sar_transform(image_size: int = 90) -> transforms.Compose:
    """
    Same preprocessing as your labeled loader:
      - Grayscale to 3 channels
      - Resize to image_size 
      - ToTensor
      - Normalize with ImageNet stats
    """
    return transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize(image_size),  # force 90x * ( just shorter side)
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


class SingleImageDataset(Dataset):
    """
    只加载一张图片，返回 (tensor, path)
    """
    def __init__(self, img_path: str):
        label_dict = {'2S1': 0, 'BMP2': 1, 'BRDM_2': 2, 'BTR70(SN_C71)': 3, 'BTR70': 3, 'BTR_60': 4, 'D7': 5,
            'T62': 6, 'T72(SN_132)': 7, 'T72': 7, 'ZIL131': 8, 'ZSU_23_4': 9}
        self.path = img_path
        self.label = 1000

        for k,v in label_dict.items():
            if k in self.path:
                self.label = v
                break
            
        self.transform = build_sar_transform(image_size=90)

    def __len__(self):
        return 1

    def __getitem__(self, idx: int):
        image = self.transform(Image.open(self.path).convert('RGB'))
        return image, self.label

# 使用：
# x = load_image_as_tensor("path/to/image.png", image_size=90, keep_ratio=False)
