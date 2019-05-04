import torch.nn as nn
import cv2
import numpy as np
import torch
from torchvision import transforms


class SmileNetwork(nn.Module):
    
    def __init__(self,output_layer):
        super(SmileNetwork,self).__init__()
        self.conv1 = nn.Conv2d(3,32,5,padding=1)
        self.bn_conv1 = nn.BatchNorm2d(32)
        self.do_conv1 = nn.Dropout2d(0.25)
        self.conv2 = nn.Conv2d(32,64,3,padding=1)
        self.bn_conv2 = nn.BatchNorm2d(64)
        self.do_conv2 = nn.Dropout2d(0.25)
        self.conv3 = nn.Conv2d(64,64,2,padding=1)
        self.bn_conv3 = nn.BatchNorm2d(64)
        self.do_conv3 = nn.Dropout2d(0.25)
        
        
        self.fc1 = nn.Linear(64*10*10,1024)
        self.bn_fc1 = nn.BatchNorm1d(1024)
        self.do_fc1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024,512)
        self.bn_fc2 = nn.BatchNorm1d(512)
        self.do_fc2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(512,output_layer)

        self.pool = nn.MaxPool2d(kernel_size=2,stride=2,padding=1)

    def forward(self,x):
        #3x64x64
        x = torch.relu(self.conv1(x))
        x = self.bn_conv1(x)
        x = self.do_conv1(x)
        #32X62X62
        x = self.pool(x)
        #32X32X32
        x = torch.relu(self.conv2(x))
        x = self.bn_conv2(x)
        #64X32X32
        x = self.pool(x)
        #64X17X17
        x = torch.relu(self.conv3(x))
        x = self.bn_conv3(x)
        x = self.do_conv3(x)
        # 64X18X18
        x = self.pool(x)
        # 64X10X10
        
        x = x.view(x.size(0),-1)
        
        x = torch.relu(self.fc1(x))
        x = self.bn_fc1(x)
        x = self.do_fc1(x)
        x = torch.relu(self.fc2(x))
        x = self.bn_fc2(x)
        x = self.do_fc2(x)
        
        x = torch.relu(self.fc3(x))
        
        return x


def model_test(model,test_image_np):
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2023, 0.1994, 0.2010])
    valid_transform = transforms.Compose([transforms.ToPILImage(),transforms.transforms.Grayscale(3),transforms.ToTensor(),normalize])
    test_image_tensor = valid_transform(test_image_np)
    test_image_tensor = test_image_tensor.unsqueeze(0)
    test_image_tensor = test_image_tensor.float()

    model.eval()
    preds = model(test_image_tensor)
    pred = preds.data.max(1)[1]
    pred = pred.data.numpy()[0]    
    if pred:
        return "Smiling"
    else:
        return "Not Smiling"


camera = cv2.VideoCapture(0)

detector = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

model = SmileNetwork(2)

model.load_state_dict(torch.load("DeepSmile.pt"))


while(True):
    (grabbed,frame) = camera.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    for (fX, fY, fW, fH) in rects:
        face = gray[fY:fY + fH, fX:fX + fW]
        face = cv2.cvtColor(face,cv2.COLOR_GRAY2BGR)        
        face = cv2.resize(face, (64, 64))        
        label = model_test(model,face)
        cv2.putText(frame, label, (fX, fY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(frame,(fX,fY),(fX+fW,fY+fH),(0,255,0),3)
    cv2.imshow('LIVE...',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
camera.release()
cv2.destroyAllWindows()