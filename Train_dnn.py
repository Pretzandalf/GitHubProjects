from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import module_dnn as dnn
import Import_dnn as imp
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = dnn.CNN()
model.to(device)

train_data_loader = imp.train_data_loader
val_data_loader = imp.val_data_loader

loss_func = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=1e-4)

def metrics_batch(target, output):
    pred = output.argmax(dim=1, keepdim=True)
    corrects=pred.eq(target.view_as(pred)).sum().item()
    return corrects

def loss_batch(loss_func, xb, yb,yb_h, opt=None):
    loss = loss_func(yb_h, yb)
    metric_b = metrics_batch(yb,yb_h)
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), metric_b

def loss_epoch(model,loss_func,dataset_dl,opt=None):
    loss=0.0
    metric=0.0
    len_data=len(dataset_dl.dataset)
    for xb, yb in dataset_dl:
        xb=xb.type(torch.float).to(device)
        yb=yb.to(device)
        yb_h=model(xb)
        # print(yb_h)
        loss_b,metric_b=loss_batch(loss_func, xb, yb,yb_h, opt)
        loss+=loss_b
        if metric_b is not None:
            metric+=metric_b
    loss/=len_data
    metric/=len_data
    return loss, metric

def train_val(epochs, model, loss_func, opt, train_dl, val_dl):
    for epoch in range(epochs):
        model.train()
        train_loss, train_metric=loss_epoch(model,loss_func,train_dl,opt)
        model.eval()
        with torch.no_grad():
            val_loss, val_metric=loss_epoch(model,loss_func,val_dl)
        print("epoch: %d, train loss: %.6f, val loss: %.6f, train accuracy: %.6f, val accuracy: %.6f" %(epoch, train_loss, val_loss, train_metric, val_metric))