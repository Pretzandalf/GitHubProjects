import torch
import torch.nn as nn
import torch.optim as optim
import module_dnn as dnn
import Import_dnn as imp
import Train_dnn as Tranin
import onnx
import datetime

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = dnn.CNN()
print(model)
model.to(device)

num_epochs = 2

train_data_loader = imp.train_data_loader
val_data_loader = imp.val_data_loader

loss_func = nn.CrossEntropyLoss()

opt = optim.Adam(model.parameters(), lr=1e-4)


Tranin.train_val(num_epochs, model, loss_func, opt, train_data_loader, val_data_loader)




Save_model= model
filepath = "C:/Users/yarom/PycharmProjects/pythonNetWork/NetWorks/LifnesDephtCNN8.pt"
torch.save(model.state_dict(), filepath)


# ONNX_MODEL_FILE = "C:/Users/yarom/PycharmProjects/pythonNetWork/NetWorks/LifnesDephtCNN2.onnx"
# print(model)
# def export_to_onnx(model, sample_input):
#    # sample_input = torch.randn(32, 3, 3, 3).to(cuda)
#    print(type(sample_input))
#    torch.onnx.export(model,            # model being run
#                      sample_input,     # model input (or a tuple for multiple inputs)
#                      ONNX_MODEL_FILE,  # where to save the model
#                      input_names = ['input'],   # the model's input names
#                      output_names = ['output']) # the model's output names
#
#    # Set metadata on the model.
#    onnx_model = onnx.load(ONNX_MODEL_FILE)
#    meta = onnx_model.metadata_props.add()
#    meta.key = "creation_date"
#    meta.value = datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
#    meta = onnx_model.metadata_props.add()
#    meta.key = "author"
#    meta.value = 'keithpij'
#    onnx_model.doc_string = 'MNIST model converted from Pytorch'
#    onnx_model.model_version = 3  # This must be an integer or long.
#    onnx.save(onnx_model, ONNX_MODEL_FILE)
#
# export_to_onnx(model, x)