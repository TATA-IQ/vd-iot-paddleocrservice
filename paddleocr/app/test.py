import cv2
import requests
import random
import base64
import numpy as np
from datetime import datetime
from PIL import Image
from io import BytesIO
import io


# url = "http://127.0.0.1:6504/detect/"
url = "http://0.0.0.0:6504/detect/"

img = cv2.imread("image3.jpg")
# print(img.shape)
img_str = cv2.imencode(".jpg", img)[1].tobytes().decode("ISO-8859-1")
# # img_str = cv2.imencode(".jpg", image)[1].tobytes().decode("ISO-8859-1")
# # img_str = base64.b64encode(cv2.imencode(".jpg", image)[1])#.tobytes()
# # image_bytes = cv2.imencode('.jpg', image)[1].tostring()
# # img_str = base64.b64encode(image_bytes)
# # print(img_str)
# print("*"*100)
# print(type(img_str))

# stream = BytesIO(img_str.encode("ISO-8859-1"))
# image = Image.open(stream).convert("RGB")
# open_cv_image = np.array(image) 
# # print(open_cv_image)

# # imgdata = base64.b64decode(img_str)
# # img = Image.open(io.BytesIO(imgdata))
# # opencv_img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
# print(open_cv_image.shape)
# print("*"*100)

# # original_image = base64.b64decode(img_str)
# # jpg_as_np = np.frombuffer(original_image, dtype=np.uint8)
# # image = cv2.imdecode(jpg_as_np, flags=1) 
# # print(str(img_str))
np_coord = {1:[150,95,455,145], 2:[590,385,950,450]} # xmin, ymin, xmax, ymax

query = {"image":img_str, "image_name":"test.jpg", "np_coord":np_coord, "model_config":{}}

# # query={"model_name": "vehicle.zip","model_path": "/object_detection/usecase4/model_id_4/vehicle.zip",  "model_id": "model_id_4",  "model_framework": "yolov8",}

r = requests.post(url, json=query)
data = r.json()
print(data)

