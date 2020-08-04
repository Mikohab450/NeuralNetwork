from __future__ import print_function
import torch
import torchvision
import os
import numpy as np
import cv2
from tqdm import tqdm
from torchvision import transforms ,datasets
import time

x_size=80
y_size=65
REBUILD_DATA=True
class BuildData():
    apples ="Data/Apples"
    orange  ="Data/Oranges"
    labels ={apples :0,orange:1}
    training_data=[]
    apple_count=0
    orange_cout=0
    def make_training_data(self):
        for label in self.labels:
            print(label)
            for f in tqdm(os.listdir(label)):
                try:
                    path = os.path.join(label,f)
                    img= cv2.imread(path,cv2.IMREAD_COLOR)
                    img=cv2.resize(img,(x_size,y_size))
                    self.training_data.append([ np.array(img),np.eye(2)[self.labels[label]]])
                    
                    if label == self.apples:
                        self.apple_count +=1
                    if label== self.orange:
                         self.orange_cout += 1
                except Exception as e:
                    print(e)
        np.random.shuffle(self.training_data)
        np.save("training_data.npy",self.training_data)
        print("apples:",self.apple_count)
        print("orange:",self.orange_cout)
if REBUILD_DATA:
    Builder=BuildData()
    Builder.make_training_data()
