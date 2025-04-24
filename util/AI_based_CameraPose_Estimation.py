import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image


class PPNet:
    def __init__(self,
                 model_path:str):
        self.model = torch.jit.load(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        
    def preprocess_frame(self,frame):
        # numpy.ndarray -> PIL.Image 변환
        frame_pil = Image.fromarray(frame)
        
        # 전처리 수행
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
        return transform(frame_pil).unsqueeze(0)
    
    def convison_output(self,shape,prediction):
        x,y,tx,ty = prediction
        s = np.sqrt(tx**2+ty**2)
        d = np.rad2deg(np.arccos(tx/s))
        if d < 0:
            d = d + 180
        elif d > 180:
            d = d - 180
        else:
            d = d
            
        if ty<0:
            d = -d
        else:
            d = d
            
        x = prediction[0]*shape[1]/100
        y = prediction[1]*shape[0]/100 # y 좌표
        angle = d  # 각도
        step = s/100
        if step < 0:
            step = 0.01
        return [x,y,angle,step]
        
        
    def run(self,frame):
        input_tensor = self.preprocess_frame(frame).to(self.device)
        img_y, img_x, _ = frame.shape
        with torch.no_grad():
            output = self.model(input_tensor)
            prediction = output.squeeze(0).tolist()
            
        Pose = self.convison_output(frame.shape,prediction)
        return {"Pose":Pose,"shape":frame.shape}