import torch.nn as nn
import torch
import torchvision
import onnx
import onnxruntime

class PilotNetUs(nn.Module):
  
  def __init__(self, n_input_channels=3, n_outputs=1, n_branches=1):
    super(PilotNetUs, self).__init__()
    #264*68*3

    #conv layers
    self.conv1 = nn.Conv2d(3,24,4,stride=2)
    #131*33*24
    self.conv2 = nn.Conv2d(24,36,5,stride=2)
    #64*15*36
    self.conv3 = nn.Conv2d(36,48,(4,3),stride=2)
    #31*7*48
    self.conv4 = nn.Conv2d(48,64,3,stride=2)
    #15*3*64
    self.conv5 = nn.Conv2d(64,64,3,stride=2)
    #7*1*64


    #linear layers
    self.fc1 = nn.Linear(448,100)
    self.fc2 = nn.Linear(100,50)
    self.fc3 = nn.Linear(50,10)
    self.fc4 = nn.Linear(10,3)

    #batch norm
    self.bn1 = nn.BatchNorm2d(24)
    self.bn2 = nn.BatchNorm2d(36)
    self.bn3 = nn.BatchNorm2d(48)
    self.bn4 = nn.BatchNorm2d(64)
    self.bn5 = nn.BatchNorm2d(64)
    self.bn6 = nn.BatchNorm1d(100)
    self.bn7 = nn.BatchNorm1d(50)

    #Relu
    self.relu1 = nn.LeakyReLU()
    self.relu2 = nn.LeakyReLU()
    self.relu3 = nn.LeakyReLU()
    self.relu4 = nn.LeakyReLU()
    self.relu5 = nn.LeakyReLU()
    self.relu6 = nn.LeakyReLU()
    self.relu7 = nn.LeakyReLU()
    self.relu8 = nn.LeakyReLU()
    self.relu9 = nn.LeakyReLU()

    #flatten
    self.flatten = nn.Flatten()

    #softmax
    self.softmax = nn.Softmax()

  def forward(self, x):

    out = self.conv1(x)
    print(out.size())
    out = self.bn1(out)
    out = self.relu1(out)
    out = self.conv2(out)
    print(out.size())
    out = self.bn2(out)
    out = self.relu2(out)
    out = self.conv3(out)
    print(out.size())
    out = self.bn3(out)
    out = self.relu3(out)
    out = self.conv4(out)
    print(out.size())
    out = self.bn4(out)
    out = self.relu4(out)
    out = self.conv5(out)
    print(out.size())
    out = self.bn5(out)
    out = self.relu5(out)
    out = self.flatten(out)
    ## after that we should create a sequential thing to make the thing work but i don't know why 
    
    out = self.fc1(out)
    out = self.bn6(out)
    out = self.relu6(out)
    out = self.fc2(out)
    out = self.bn7(out)
    out = self.relu7(out)
    out = self.fc3(out)
    out = self.relu3(out)
    out = self.fc4(out)
    out = self.relu4(out)
    
    print(out.size())

    return out


model = PilotNet()
dummy_input = torch.randn(1, 3, 264, 68)
torch.onnx.export(model, dummy_input, "pilotnet.onnx", input_names=["input"], output_names=["output"])
