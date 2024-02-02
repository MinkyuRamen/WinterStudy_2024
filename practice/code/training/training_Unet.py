import glob
import torch
import numpy as np
from PIL import Image
import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.optim.adam import Adam

from torchvision import transforms
from torchvision.datasets.oxford_iiit_pet import OxfordIIITPet
from torchvision.transforms import Compose, ToTensor, Resize
import sys, os
sys.path.append("/root/limlab/mk/practice/code/model")
from UNet_model import UNet

dataset = OxfordIIITPet(root="/root/limlab/mk/practice/data", download=True)

path_to_annotation = "/root/limlab/mk/practice/data/oxford-iiit-pet/annotations/trimaps/"
path_to_image = "/root/limlab/mk/practice/data/oxford-iiit-pet/images/"

class Pets(Dataset):
    def __init__(self, path_to_img, path_to_anno, train=True, transforms=None, input_size=(128,128)):
        # gt image를 이름순으로 정렬
        self.images = sorted(glob.glob(path_to_img+"/*.jpg"))
        self.annotations = sorted(glob.glob(path_to_anno+"/*.png"))

        self.X_train = self.images[:int(len(self.images)*0.8)]
        self.X_test = self.images[int(len(self.images)*0.8):]
        self.Y_train = self.annotations[:int(len(self.annotations)*0.8)]
        self.Y_test = self.annotations[int(len(self.annotations)*0.8):]

        self.train = train
        self.transforms = transforms
        self.input_size = input_size

    def __len__(self): # 데이터 개수 반환
        if self.train:
            return len(self.X_train)
        else:
            return len(self.X_test)
    
    """
        다중분류를 이진분류로 바뀌도록, img에서 background=0 / foreground=1 로 변경하는 함수
    """
    def preprocess_mask(self, mask):
        mask = mask.resize(self.input_size)
        mask = np.array(mask).astype(np.float32)
        mask[mask!=2.0] = 1.0
        mask[mask==2.0] = 0.0
        return torch.tensor(mask)

    def __getitem__(self, idx): # idx th data와 label를 반환
        if self.train:
            X_train = Image.open(self.X_train[idx])
            X_train = self.transforms(X_train)
            Y_train = Image.open(self.Y_train[idx])
            Y_train = self.preprocess_mask(Y_train)

            return X_train, Y_train

        else:
            X_test = Image.open(self.X_test[idx])
            X_test = self.transforms(X_test)
            Y_test = Image.open(self.Y_test[idx])
            Y_test = self.preprocess_mask(Y_test)

            return X_test, Y_test
        

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

def transform_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image

transform = Compose([
    transforms.Lambda(lambda img: transform_image(img)),
    Resize((128,128)), # 원래는 256*256
    ToTensor(),
])

train_set = Pets(path_to_img=path_to_image,
                 path_to_anno=path_to_annotation,
                 transforms=transform)
test_set = Pets(path_to_img=path_to_image,
                path_to_anno=path_to_annotation,
                transforms=transform,
                train=False)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set)

model = UNet().to(device)

lr = 1e-4
optim = Adam(params=model.parameters(), lr=lr)

for epoch in range(1):
    iterator = tqdm.tqdm(train_loader)

    for data, label in iterator:
        optim.zero_grad()

        preds = model(data.to(device))
        loss = nn.BCEWithLogitsLoss()(preds, label.to(device))

        loss.backward()
        optim.step()

        iterator.set_description(f"Epoch: {epoch}, Loss: {loss.item()}")

torch.save(model.state_dict(), "/root/limlab/mk/practice/model_pth/unet_model.pt")

model.load_state_dict(torch.load("/root/limlab/mk/practice/model_pth/unet_model.pt"))
data, label = test_set[1]
pred = model(torch.unsqueeze(data.to(device), dim=0))>0.5 # pixel에 binary classification 진행
# unsqueeze >> HxW를 1xHxW로 변경

with torch.no_grad():
    plt.subplot(1,2,1)
    plt.title("Predicted")
    plt.imshow(pred)

    plt.subplot(1,2,2)
    plt.title("Label")
    plt.imshow(label)

    plt.show()