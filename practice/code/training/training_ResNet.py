import tqdm

import torch
import torch.nn as nn
from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import Compose, ToTensor
from torchvision.transforms import RandomHorizontalFlip, RandomCrop
from torchvision.transforms import Normalize
from torch.utils.data import DataLoader
from torch.optim.adam import Adam

import sys, os
sys.path.append("../model")
from ResNet_model import ResNet, BasicBlock

## training code
training_data = CIFAR10(
    root="/root/limlab/mk/practice/data",
    train=True,
    download=True,
    transform=ToTensor()
)

imgs = [item[0] for item in training_data]
imgs = torch.stack(imgs, dim=0).numpy() # tensor를 dim 방향으로 합쳐준다.
# (48,48) tensor를 dim=0 방향으로 3개 합치면, (3,48,48) size의 tensor가 된다.

mean_rgb = [imgs[:,i,:,:].mean() for i in range(3)]
std_rgb = [imgs[:,i,:,:].std() for i in range(3)]
# print(mean_rgb, std_rgb)

transforms = Compose([
    RandomCrop((32,32), padding=4),
    RandomHorizontalFlip(p=0.5),
    ToTensor(),
    Normalize(mean_rgb, std_rgb),
])

training_data = CIFAR10(
    root="/root/limlab/mk/practice/data",
    train=True,
    download=True,
    transform=transforms
)

test_data = CIFAR10(
    root="/root/limlab/mk/practice/data",
    train=False,
    download=True,
    transform=transforms
)

train_loader = DataLoader(training_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

model = ResNet(num_classes=10).to(device)

lr = 1e-4
optim = Adam(model.parameters(), lr=lr)

for epoch in range(10):
    iterator= tqdm.tqdm(train_loader)
    for data, label in iterator:
        optim.zero_grad()

        data = data.to(device)
        preds = model(data.to(device))

        loss = nn.CrossEntropyLoss()(preds, label.to(device))
        loss.backward()
        optim.step()

        iterator.set_description(f"Epoch {epoch} Loss {loss.item()}")

torch.save(model.state_dict(), "/root/limlab/mk/practice/model_pth/ResNet_model.pth")

model.load_state_dict(torch.load("/root/limlab/mk/practice/model_pth/ResNet_model.pth"))
num_corr = 0

with torch.no_grad():
    for data, label in test_loader:
        output = model(data.to(device))
        preds = torch.argmax(output, dim=1)

        corr = (preds == label.to(device)).sum().item()
        num_corr += corr

print(f"Accuracy: {num_corr/len(test_data)}")