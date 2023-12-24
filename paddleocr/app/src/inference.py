""" inference """
import os
import sys
# import glob
# import copy
# import time
# import re

# import cv2
# import numpy as np
import torch

os.environ['KMP_DUPLICATE_LIB_OK']='True'
from paddleocr import PaddleOCR
from console_logging.console import Console
console=Console()

class InferenceModel:
    """
    PaddleOCR inference
    """
    def __init__(self, model_path=None, gpu=False, logger=None):
        """
        Initialize PaddleOCR inference
        
        Args:
            model_path (str): path of the downloaded and unzipped model
            gpu=True, if the system have NVIDIA GPU compatibility
        """
        if gpu:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"
        self.log = logger
        self.model_path = model_path #+ "/" + os.listdir(self.model_path)[0]
        self.det_model_path = model_path + "/" + os.listdir(model_path)[0]
        self.rec_model_path = model_path + "/" + os.listdir(model_path)[1]
        console.info(f" detection model path {self.det_model_path}")
        self.log.info(f" detection model path {self.det_model_path}")
        self.ocr = None
        self.model = None
        self.augment = None
        self.object_confidence = 0.01
        self.iou_threshold = 0.01
        self.classes = None
        self.agnostic_nms = False
        self.max_det = 1000
        self.half = False

    def initializeVariable(
        self,
        conf=0.01,
        iou_threshold=0.01,
        classes=None,
        agnostic_nms=False,
        max_det=1000,
        half=False,
        augment=False,
    ):
        """
        This will initialize the model parameters of inference. This configuration is specific to camera group
        Args:
            conf (float): confidence of detection
            iou_threshold (float): intersection over union threshold of detection
            classes (obj): classes of the detection
            agnostic_ms (boolean): Non max supression
            half (boolean): precision of the detection
            augment (boolean): Augmentation while detection
        """
        self.object_confidence = conf
        self.iou_threshold = iou_threshold
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.max_det = max_det
        self.half = half

    def loadmodel(self):
        '''
        This will load PaddleOCR model
        '''
        if os.path.exists(self.det_model_path) and os.path.exists(self.rec_model_path):
            self.model = PaddleOCR(use_angle_cls=False,lang='en',show_log=False,det_model_dir=self.det_model_path,rec_model_dir=self.rec_model_path)

        # if  os.path.exists(self.model_path):
        #     self.model = YOLO(self.model_path)
        else:
            print("MODEL NOT FOUND")
            self.log.error("MODEL NOT FOUND")
            console.error("MODEL NOT FOUND")
            sys.exit()
        # self.stride = int(self.model.stride.max())
        # self.names = (
        #     self.model.module.names
        #     if hasattr(self.model, "module")
        #     else self.model.names
        # )

    def getClasses(self):
        """
        Get the classes of the model

        """
        return "characters"

    def infer(self, image, model_config=None):
        """
        This will do the detection on the image
        Args:
            image (array): image in numpy array
            model_config (dict): configuration specific to camera group for detection
        Returns:
            results (list): list of dictionary. It will have all the detection result.
        """
        image_height, image_width, _ = image.shape
        self.log.info(f"image shape===={image.shape}")
        console.info(f"image shape===={image.shape}")
        # raw_image = copy.deepcopy(image)
        # img0 = copy.deepcopy(image)
        # img = copy.deepcopy(image)

        result = self.model.ocr(image, cls=False)
        self.log.info(f"result {result}")
        console.info(f"result {result}")
        if len(result[0]):
            for idx in range(len(result)):
                res = result[idx]
                self.log.info(f"res===  {res}")
                console.info(f"res===  {res}")
                txts = [line[1][0] for line in res if len(line[1][0]) > 2]
                np_pred = "".join(txts)
                # np_pred = re.sub(r'[^\w]', '',np_pred)
                # if len(np_pred) > 10:
                #     np_pred = np_pred.replace('IND','')
                # if len(np_pred) > 10:
                #     np_pred = np_pred.replace('IN','')
                # if len(np_pred) > 10:
                #     np_pred = np_pred.replace('I','')
                # if len(np_pred) > 10:
                #     np_pred = np_pred[1:]
        else:
            np_pred = ''
        return np_pred