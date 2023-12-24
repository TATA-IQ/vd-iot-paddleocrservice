""" paddle ocr app"""
# import cv2
# import subprocess
import os
import socket
from zipfile import ZipFile
from io import BytesIO
from urllib.parse import urlparse
import requests
import time
import consul

from PIL import Image
# import io
import torch
# import base64


from fastapi import FastAPI
import uvicorn
import numpy as np

from sourcelogs.logger import create_rotating_log
from sftpdownload.download import SFTPClient
from src.inference import InferenceModel
from src.configParser import Config
from querymodel.imageModel import Image_Model
from utils_download.model_download import DownloadModel
from console_logging.console import Console
console=Console()
os.makedirs("logs", exist_ok=True)
log = create_rotating_log("logs/logs.log")

def get_local_ip():
        '''
        Get the ip of server
        '''
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # doesn't even have to be reachable
            s.connect(("192.255.255.255", 1))
            IP = s.getsockname()[0]
        except:
            IP = "127.0.0.1"
        finally:
            s.close()
        return IP

def register_service(consul_conf,port):
    name=socket.gethostname()
    # local_ip=socket.gethostbyname(socket.gethostname())
    local_ip=get_local_ip()
    consul_client = consul.Consul(host=consul_conf["host"],port=int(consul_conf["port"]))
    consul_client.agent.service.register(
    "yolov8validate",service_id=name+"-yolov8-"+consul_conf["env"],
    port=int(port),
    address=local_ip,
    tags=["python","yolov8",consul_conf["env"]]
)

def get_service_address(consul_client,service_name,env):
    while True:
        
        try:
            print("===service_name===", service_name)
            services=consul_client.catalog.service(service_name)[1]
            print(services)
            for i in services:
                if env == i["ServiceID"].split("-")[-1]:
                    return i
        except Exception as ex:
            print(ex)
            time.sleep(10)
            continue




def get_confdata(consul_conf):
    consul_client = consul.Consul(host=consul_conf["host"],port=consul_conf["port"])
    pipelineconf=get_service_address(consul_client,"pipelineconfig",consul_conf["env"])

    
    
    env=consul_conf["env"]
    
    endpoint_addr="http://"+pipelineconf["ServiceAddress"]+":"+str(pipelineconf["ServicePort"])
    print("endpoint addr====",endpoint_addr)
    while True:
        
        try:
            res=requests.get(endpoint_addr+"/")
            endpoints=res.json()
            log.info(f"===got endpoints==={endpoints}")
            console.info(f"===got endpoints==={endpoints}")
            break
        except Exception as ex:
            log.error(f"endpoint exception==>{ex}")
            console.error(f"endpoint exception==>{ex}")
            time.sleep(10)
            continue
    
    while True:
        try:
            res=requests.get(endpoint_addr+endpoints["endpoint"]["model"])
            modelconf=res.json()
            log.info(f"modelconf===>{modelconf}")
            console.info(f"modelconf===>{modelconf}")
            break
            

        except Exception as ex:
            log.error(f"modelconf exception==>{ex}")
            console.error(f"modelconf exception==>{ex}")
            time.sleep(10)
            continue
    console.info("=======searching for dbapi====")
    while True:
        try:
            log.info("=====consul search====")
            console.info("=====consul search====")
            dbconf=get_service_address(consul_client,"dbapi",consul_conf["env"])
            # print("****",dbconf)
            dbhost=dbconf["ServiceAddress"]
            dbport=dbconf["ServicePort"]
            res=requests.get(endpoint_addr+endpoints["endpoint"]["dbapi"])
            dbres=res.json()
            console.info(f"===got db conf==={ dbres}")
            break
        except Exception as ex:
            log.error("db discovery exception==={0}".format(ex))
            console.error("db discovery exception==={0}".format(ex))
            time.sleep(10)
            continue
    for i in dbres["apis"]:
        print("====>",i)
        dbres["apis"][i]="http://"+dbhost+":"+str(dbport)+dbres["apis"][i]

    
    console.info("======dbres======")
    log.info(dbres)
    log.info(modelconf)
    console.info(dbres)
    console.info(modelconf)
    return  dbres,modelconf

class SetupModel():
    '''
    Class to Setup the Inference Model
    '''

    def __init__(self,config_path="config/config.yaml",model_config_path="config/model.yaml",logger = log):       
        config = Config.yamlconfig(config_path)
        dbapi,minio_conf=get_confdata(config[0]["consul"])
        self.modelconf=Config.yamlModel(model_config_path)[0]
        self.apis = dbapi["apis"]
        # self.sftp = conf["sftp"]
        self.minio=minio_conf["minio"]
        self.log = logger
        # conf = Config.yamlconfig(config_path)[0]
        # modelconf=Config.yamlModel(model_config_path)[0]
        # self.apis = conf["apis"]
        # # self.sftp = conf["sftp"]
        # self.minio=conf["minio"]
        # self.modelconf=modelconf


    def get_local_ip(self):
        '''
        Get the ip of server
        '''
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # doesn't even have to be reachable
            s.connect(("192.255.255.255", 1))
            IP = s.getsockname()[0]
        except:
            IP = "127.0.0.1"
        finally:
            s.close()
        return IP


    # subprocess.getoutput("docker ps -aqf name=containername")
    '''
    hostid = subprocess.getoutput("cat /etc/hostname")
    cgroupout = subprocess.getoutput("cat /proc/self/cgroup")
    print(cgroupout)
    print("===>", cgroupout.split(":")[2][8:20])
    hostid = cgroupout.split(":")[2][8:20]
    # hostid="ce09255b0b13"
    query = {"container_id": hostid}
    print("query====>", query)
    responseContainer = requests.get(self.apis["container"], json=query)
    print(responseContainer.json())
    responseModel = requests.get(self.apis["container_model"], json=query)
    print("found====>", responseModel.json())
    '''
    def getModelMasterCofig(self):
        '''
        Get the Master Configuration of model
        '''
        modelmaster = requests.get(
            self.apis["model_master_config"], json={"model_id": self.modelconf["model_id"]}
        ).json()
        console.info(f"======Model Master Response======{modelmaster}")
        self.log.info(f"======Model Master Response======{modelmaster}")
        return modelmaster['data']
        # model_path = responseModel.json()["data"][0]["model_path"]
        # time.sleep(10)

    # def downloadSftp(self,model_path):
    #     '''
    #     This function will download the model from the server

    #     Args:
    #         model_path (dict): sftp server connection configuration
    #     '''

    #     print("downloading from server")

    #     sf = SFTPClient(self.sftp["host"], self.sftp["port"], self.sftp["username"], self.sftp["password"])

    #     sf.downloadyolov5(model_path, "model/test.zip")
    #     print("downloaded from server")

    #     model_nm = model_path.split(".")[0]
    #     with ZipFile("model/test.zip", "r") as zObject:
    #         zObject.extractall(path="model/temp")
    #         print(model_nm)
    def downloadMinio(self,modelmaster):
        '''
        Args:
            modelmaster (dict): configuration to connect with minio
        '''
        local_path="model"
        yolodownload = DownloadModel(modelmaster["model_framework"], self.minio, logger = self.log)
        yolodownload.createLocalFolder(local_path)
        yolodownload.save_data(modelmaster["model_path"], local_path)
        # downloadData(data.model_path,local_path)
        modelpathparse=urlparse(modelmaster["model_path"])
        model_name=os.path.basename(modelpathparse.path)
        yolodownload.unzip(local_path + "/" + model_name, local_path, model_name)



    def createIP(self):
        '''
        This function will create the api and update the endpoint url of model
        '''
        ip = self.get_local_ip()
        url = (
            "http://"
            + str(ip)
            + ":"
            + str(self.modelconf["port"])
            + "/detect"
        )
        model_id = self.modelconf["model_id"]
        print("updating for model id===>", model_id)

        responseupdate = requests.post(
            self.apis["update_endpoint"], json={"model_end_point": url, "model_id": model_id}
        )
        print(responseupdate.json())





    def getModelConfig(self):
        '''
        This function call the model configuration api
        '''
        model_id = self.modelconf["model_id"]
        model_config = requests.get(self.apis["model_config"], json={"model_id": model_id})
        return model_config



    def startModel(self):
        '''
        This function does the model setup
        '''
        modelmasterdata=self.getModelMasterCofig()
        self.downloadMinio(modelmasterdata[0])
        gpu=False
        model_name=modelmasterdata[0]["model_name"]
        framework=modelmasterdata[0]["model_framework"]

        if torch.cuda.is_available():
            gpu = True
        model_list = os.listdir("model/")
        print("model/" + model_list[0])
        im = InferenceModel(model_path="model/" + model_list[0], gpu=gpu, logger = self.log)
        im.loadmodel()
        console.info("====Model Loaded====")
        console.info("Running api on GPU {}".format(gpu))
        self.log.info("Running api on GPU {}".format(gpu))
        self.createIP()
        return im,self.modelconf,model_name, framework


st=SetupModel()
im,modelconf,model_name, framework=st.startModel()
app = FastAPI()
def strToImage(imagestr):
    # print("*"*100)
    stream = BytesIO(imagestr.encode("ISO-8859-1"))
    image = Image.open(stream).convert("RGB")
    open_cv_image = np.array(image)
    return open_cv_image

# def strToImage(imagestr):
#     imagencode = imagestr.encode()
#     decodedimage = base64.b64decode(imagencode)
#     nparr = np.frombuffer(decodedimage, np.uint8)
#     img_np = cv2.imdecode(nparr, flags=1)
#     return img_np


@app.get("/test")
def test_fetch():
    '''
    Test Api: Just for testing if model is running
    '''
    return {"status":"active","message":f"Model Name: {0} Framework {1}".format(model_name, framework) }


@app.post("/detect")
def detection(data: Image_Model):
    '''
    Args:
        data (Image_Model): Accepts image, image name and configuration specific to the camera group
    
    Returns:
        dict: inferred result of the images
    '''
    image = strToImage(data.image)
    image_name = data.image_name
    np_coordinates_dict  = data.np_coord
    log.info(f"model config is {data.model_config}")
    console.info(f"model config is {data.model_config}")
    log.info(f"image_name is {image_name}, np_coordinates_dict are : {np_coordinates_dict}")
    console.info(f"image_name is {image_name}, np_coordinates_dict are : {np_coordinates_dict}")

    if data.model_config is None:
        log.info("===model config is None===")
        console.info("===model config is None===")
        res = []
        for x,y in np_coordinates_dict.items():
            xmin = y[0]
            ymin = y[1]
            xmax = y[2]
            ymax = y[3]
            image1 = image[y[1]:y[3],y[0]:y[2]]
            inter_res = im.infer(image1)
            print(inter_res)
            # np_coords = [xmin,ymin,xmax,ymax]
            res.append({
                "id":x,
                "image_name":image_name,
                "score":0,
                "xmin":xmin, 
                "ymin":ymin,
                "xmax":xmax,
                "ymax":ymax,
                "classname":"numberplate",
                "np":inter_res,                
            })
    else:
        log.info("===model config is not None===")
        console.info("===model config is not None===")
        res = []
        if np_coordinates_dict is not None:
            for x,y in np_coordinates_dict.items():

                xmin = y[0]
                ymin = y[1]
                xmax = y[2]
                ymax = y[3]

                image1 = image[y[1]:y[3],y[0]:y[2]]

                inter_res = im.infer(image1)
                print(inter_res)
                # np_coords = [xmin,ymin,xmax,ymax]
                res.append({
                    "id":x,
                    "image_name":image_name,
                    "score":0,
                    "xmin":xmin, 
                    "ymin":ymin,
                    "xmax":xmax,
                    "ymax":ymax,
                    "classname":"numberplate",
                    "np":inter_res             
                })
        # res = im.infer(image, data.model_config)
    console.info("======inference done**********")
    log.info("======inference done**********")
    log.info(res)
    console.info(res)
    log.info(type(res))
    return {"data": res}


@app.get("/classes")
def detection():
    '''
    This function return all the classes of the model
    '''
    res = im.getClasses()
    return {"data": res}
    
if __name__ == "__main__":
    console.info("=====inside main************")
    log.info("=====inside main************")
    uvicorn.run(app, host="0.0.0.0", port=int(modelconf["port"]))


# app = FastAPI()
# @app.get("/test")
# async def ClassMaster_fetch(ImageModel):
#     return {"data":"Hi From TesorFlow "+responseModel.json()["data"][0]["model_usecase"]}

#         ##im.infer(ImageModel.image)
