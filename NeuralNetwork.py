from __future__ import print_function
import torch
import torchvision
import os
import numpy as np
import cv2
from tqdm import tqdm
from torchvision import transforms ,datasets
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

if torch.cuda.is_available():

    device = torch.device("cuda:0")
    print("running on GPU")
else:
    device = torch.device("cpu")
    print("running on CPU")
print(device)

x_size=80
y_size=65
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(3,32,5)
        self.conv2=nn.Conv2d(32,64,5)
        self.conv3=nn.Conv2d(64,128,5)

        x=torch.randn(x_size,y_size,3).view(-1,3,x_size,y_size)
        self._to_linear = None
        self.convs(x)
        self.fc1= nn.Linear(self._to_linear,248)
        self.fc2 = nn.Linear(248,2)

    def convs(self,x):
        x=F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x=F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x=F.max_pool2d(F.relu(self.conv3(x)), (2,2))
        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x

    def forward(self, x):
        x=self.convs(x)
        x= x.view(-1,self._to_linear)
        x= F.relu(self.fc1(x))
        x = self.fc2(x)  
        return  F.softmax(x,dim=1)



training_data=np.load("training_data.npy",allow_pickle=True)        #loading data
X = torch.from_numpy(np.array([i[0] for i in training_data])).view(-1,3,x_size,y_size)      #loading images into tensor X
#print(X.shape)
X = X/255. #scaling pixel values from 0-255 to 0-1

Y= torch.Tensor([i[1] for i in training_data])          #loading one-hot vectors int tensor Y
training_percent = 0.5 # determines how much data will be used for training network
training_size= int(len(X)*training_percent)
print("total images:",len(X))


train_X = X[-training_size:] 
train_Y = Y[-training_size:]
print("training set:",len(train_X))
test_X = X[:-training_size] 
test_Y = Y[:-training_size]
print("testing set:",len(test_X))
net = Net().to(device)      #creating the network
optimizer = optim.Adam(net.parameters(), lr=0.004)
loss_function = nn.MSELoss()

#fuction used for first training, a then testing the model
def train_only(net):
    test_X=X[:-training_size]
    test_Y=Y[:-training_size]
    BATCH_SIZE=20
    EPOCHS = 8
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    loss_function = nn.MSELoss()
    for epoch in range(EPOCHS):
        for i in tqdm(range(0,len(train_X),BATCH_SIZE)):
            #print(i,i+BATCH_SIZE)
            batch_X=train_X[i:i+BATCH_SIZE].view(-1,3,x_size,y_size)
            batch_Y =train_Y[i:i+BATCH_SIZE]
            batch_X,batch_Y=batch_X.to(device),batch_Y.to(device)
            net.zero_grad()
            outputs=net(batch_X)
            loss=loss_function(outputs,batch_Y)
            loss.backward()
            optimizer.step()
        print(loss)

def test_only(net):
    correct = 0
    total =0
    with torch.no_grad():
        for i in tqdm(range(len(test_X))):
            real_class=torch.argmax(test_Y[i]).to(device)
            net_out = net(test_X[i].view(-1,3,x_size,y_size).to(device))[0]
            predicted_class=torch.argmax(net_out)
            if predicted_class == real_class:
                correct+=1
            total +=1
    print("Accuraccy:", round(correct/total,6))

#testing and displaying custom images
def cutstomTest(net):
    imgName=""
    #input(imgName)
    while imgName != "end":
        imgName= input()
        img= cv2.imread(imgName, cv2.IMREAD_COLOR)
        if img is not None:
            img=cv2.resize(img,(x_size,y_size))
            tensorImage =torch.Tensor( np.array(img)).view(-1,3,x_size,y_size)
            with torch.no_grad():
                result=net(tensorImage.to(device))
                print("This image shows an ")
                if torch.argmax(result):
                    print("orange")
                else:
                    print("apple")
            cv2.imshow("image",img)
            cv2.waitKey(0)
 

def forward_pass(X,Y,train=False):
    if train:
        net.zero_grad()
    outputs = net(X)
    matches = [torch.argmax(i)==torch.argmax(j) for i,j in zip(outputs,Y)]
    accuracy = matches.count(True)/len(matches)
    loss = loss_function(outputs,Y)
    if train:
        loss.backward()
        optimizer.step()
    return accuracy,loss
     
def test(size=35):
    random_start = np.random.randint(len(test_X)-size)
    X,Y = test_X[random_start:random_start+size],test_Y[random_start:random_start+size]
    with torch.no_grad():
        vall_acc, val_loss = forward_pass(X.view(-1,3,x_size,y_size).to(device),Y.to(device))
    return vall_acc, val_loss


#trains the network and tests it at the same time, makes a log of it
def train():
    MODEL_NAME = f"model-{int(time.time())}"
    BATCH_SIZE=80
    EPOCHS = 15
    with open("model(80,15,20,0.004).log","a") as f:
        for epoch in range(EPOCHS):
            for i in tqdm(range(0,len(train_X),BATCH_SIZE)):
                batch_X=train_X[i:i+BATCH_SIZE].view(-1,3,x_size,y_size).to(device)
                batch_Y =train_Y[i:i+BATCH_SIZE].to(device)
                acc,loss=forward_pass(batch_X,batch_Y,train=True)
                if i % 20 == 0:
                    val_acc,val_loss = test(size=54)
                    f.write(f"{MODEL_NAME},{round(time.time(),3)},{epoch},{round(float(val_acc),3)},{round(float(val_loss),4)},{round(float(acc),3)},{round(float(loss),4)}\n")



train()
#cutstomTest(net)

#train(net)
#test(net)
