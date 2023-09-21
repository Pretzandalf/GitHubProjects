from PIL import Image
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import torch.nn.functional as F
import torchvision
import torch.utils.model_zoo as model_zoo
import torch.optim as optim
import module_dnn as dnn
import matplotlib.pyplot as plt
from torch.autograd import Variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = dnn.MyNetwork()
model.to(device)

filepath = "C:/Users/yarom/PycharmProjects/pythonNetWork/NetWorks/Teacher2.pt"

model.load_state_dict(torch.load(filepath))
model.eval()

def image_loader(loader, image_name):
    image = Image.open(image_name)
    image = loader(image)
    image = image.clone().detach()
    image = image.unsqueeze(0)
    return image.to(device)

data_transforms = transforms.Compose([
    transforms.Resize((28,28)), # Input Shape!
    transforms.ToTensor() ])
path_to_img = "C:/Users/yarom/PycharmProjects/pythonNetWork/Test/Portrait_Dataset/Test/2.jpg"
x = image_loader(data_transforms, path_to_img)
z = model(x)
if z[0][0].item() > z[0][1].item():
    za = "Человек"
else:
    za = "Портрет"

a = Image.open(path_to_img)
plt.title("Estimation of our model = {}".format(za))
plt.imshow(a)