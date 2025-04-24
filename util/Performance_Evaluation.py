from .Mapping_Algorithm import mapping
from .AI_based_CameraPose_Estimation import PPNet
from .Camera_Parameter import Camera_intrinsic_parameter, Pipe_dimensions
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


class Performance:
    def __init__(self,
                 AI_model_path:str,
                 Directory_path:str,
                 **kwargs
                 ):
        """
        -----
        **kwargs
        -----
        water = 만관 상태에서 촬영되었는지에 대한 여부(기본값:False)\n
        pipe_type = 관 종류에 대한 정보(ex:"PVC")(기본값:None)\n
        """
        self.Directory_path = Directory_path
        self.water = kwargs.get("water") if isinstance(kwargs.get('water'), bool) else False
        self.Camera_intr_parms = kwargs.get("pipe_type") if isinstance(kwargs.get('pipe_type'), str) else None
        self.Pipe_diameter = Pipe_dimensions[self.Camera_intr_parms]
        self.Camera_parms = Camera_intrinsic_parameter
        self.model = PPNet(AI_model_path)
        self.mapping_alg = {"HD" : mapping(water=self.water,
                                           f=self.Camera_parms["HD"]["f"],
                                           cx=self.Camera_parms["HD"]["cx"],
                                           cy=self.Camera_parms["HD"]["cy"],
                                           pipe_diameter = self.Pipe_diameter
                                           ),
                            "FHD": mapping(water=self.water,
                                           f=self.Camera_parms["FHD"]["f"],
                                           cx=self.Camera_parms["FHD"]["cx"],
                                           cy=self.Camera_parms["FHD"]["cy"],
                                           pipe_diameter = self.Pipe_diameter
                                           )}

        self.data_set = self.get_image_mask_pairs()
        self.mapping_map = []
        self.mapping_result = []

    def get_image_mask_pairs(self):
        files = os.listdir(self.Directory_path)
        png_files = [f for f in files if f.endswith('.png')]

        image_set = set()
        mask_set = set()

        for file in png_files:
            if '_mask' in file:
                mask_set.add(file)
            else:
                image_set.add(file)

        pairs = []
        for image in image_set:
            name, _ = os.path.splitext(image)
            mask_name = f"{name}_mask.png"
            if mask_name in mask_set:
                pairs.append({"RGB_image_path" : os.path.join(self.Directory_path, image),
                              "bin_image_path" : os.path.join(self.Directory_path, mask_name),
                              "RGB_image" : cv2.cvtColor(cv2.imread(os.path.join(self.Directory_path, image)),cv2.COLOR_BGR2RGB),
                              "bin_image" :  cv2.imread(os.path.join(self.Directory_path, mask_name))})

        return pairs
    

    def Perfomance_eval(self):
        for data in self.data_set:
            frame_name = data["RGB_image_path"]
            RGB_img = data["RGB_image"]
            y,x,_ = RGB_img.shape
            bin_img = data["bin_image"]
            Pose = self.model.run(RGB_img)

            image_resolution = None
            for key, params in self.Camera_parms.items():
                if params["size"] == (x,y):
                    image_resolution = key
                    break
            
            mapping_img = self.mapping_alg[image_resolution].run(bin_img,Pose["Pose"])
            mapping_img = cv2.cvtColor(mapping_img,cv2.COLOR_BGR2GRAY)
            mapping_img = np.where(mapping_img>0,255,0).astype(np.uint8)
            self.mapping_map.append(mapping_img)
            self.mapping_result.append(np.count_nonzero(mapping_img))

    def MAPE(self):
        answer = np.zeros(self.mapping_map[0].shape, dtype=np.uint8)
        radius = 190 // 2 
        center = (self.mapping_map[0].shape[0]//2,self.mapping_map[0].shape[1]//2)
        cv2.circle(answer, center, radius, color=255, thickness=-1)
        answer_area = np.count_nonzero(answer)
        if answer_area == 0:
            answer_area = 28345
        predicted = np.array(self.mapping_result)
        Mean_Absolute_Percentage_Error = np.mean(np.abs((answer_area-predicted)/answer_area))*100
        return Mean_Absolute_Percentage_Error