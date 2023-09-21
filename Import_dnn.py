import torch
from torchvision import transforms
import torchvision


train_data_path = "C:/Users/yarom/PycharmProjects/pythonNetWork/Data/Lifnes_Dataset/"
img_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((100,100))
    ])
train_data = torchvision.datasets.ImageFolder(root=train_data_path,transform=img_transforms)
train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
train_dataset = train_data_loader

val_data_path = "C:/Users/yarom/PycharmProjects/pythonNetWork/Test/Lifnes_Dataset/"
val_data = torchvision.datasets.ImageFolder(root=val_data_path,transform=img_transforms)
val_data_loader  = torch.utils.data.DataLoader(val_data, batch_size=128, shuffle=True)