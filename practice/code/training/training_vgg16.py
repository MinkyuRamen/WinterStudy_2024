import torch
import torch.nn as nn

from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import ToTensor, Compose, Resize, RandomHorizontalFlip, RandomCrop, Normalize
import torchvision.transforms as T

from torch.utils.data.dataloader import DataLoader
from torch.optim.adam import Adam

from torchvision.models import vgg16, VGG16_Weights
import tqdm

## pre process
training_data = CIFAR10(
    root="/root/limlab/mk/practice/data",
    train=True,
    download=True,
    transform=ToTensor()
)

imgs = [item[0] for item in training_data]
imgs = torch.stack(imgs, dim=0).numpy() # tensor를 dim 방향으로 합쳐준다.
# (224,224) tensor를 dim=0 방향으로 3개 합치면, (3,244,244) size의 tensor가 된다.

mean_rgb = [imgs[:,i,:,:].mean() for i in range(3)]
std_rgb = [imgs[:,i,:,:].std() for i in range(3)]
print(mean_rgb, std_rgb)

transforms = Compose([
    Resize(224),
    RandomCrop((224,224), padding=4),
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

## transfer learning
weights = VGG16_Weights.DEFAULT
model = vgg16(weights=weights)

# ImageNet은 class가 1000개이기 때문에 classifier의 출력값으로 1000개가 나온다. >> 10개로 줄여야 한다.
fc = nn.Sequential(
    nn.Linear(512*7*7, 4096), # 마지막층의 channel_size=512, HxW=7x7
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(4096, 4096),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(4096,10)
)

model.classifier = fc
model.to(device)

## training
lr = 1e-4

optim = Adam(model.parameters(), lr=lr)

for epoch in range(10):

    iterator = tqdm.tqdm(train_loader)
    for data, label in iterator:

        optim.zero_grad()

        preds = model(data.to(device))

        loss = nn.CrossEntropyLoss()(preds, label.to(device))
        loss.backward()
        optim.step()

        iterator.set_description(f"epoch: {epoch}, loss: {loss.item()}")

torch.save(model.state_dict(), "/root/limlab/mk/practice/model_pth/ResNet_model.pth")

## eval
model.load_state_dict(torch.load("./model/VGG_pretrained_model.pth"))
num_corr = 0

with torch.no_grad():
    for data, label in test_loader:

        output = model(data.to(device))
        preds = torch.argmax(output, dim=1)

        corr = torch.sum(preds==label.to(device)).item()
        num_corr += corr

    print(f"accuracy: {num_corr/len(test_data)}")