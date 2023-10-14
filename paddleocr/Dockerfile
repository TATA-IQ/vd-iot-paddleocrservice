FROM python:3.9.2
RUN pip install pandas
RUN pip install kafka-python
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install opencv-python
RUN pip install numpy
RUN pip install matplotlib
RUN pip install pandas
RUN pip install requests
RUN pip install sqlalchemy
#RUN pip3 install torch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install imutils
RUN pip install PyYAML
RUN pip install tqdm
RUN pip install seaborn
RUN pip install scipy
RUN pip install glob2
RUN pip install paramiko
RUN pip install fastapi==0.99.1
RUN pip install "uvicorn[standard]"
RUN pip install torch
RUN pip install torchvision
RUN pip install protobuf==3.20.*
RUN pip install ipython
RUN pip install psutil
RUN pip install minio
RUN pip install paddlepaddle==2.4.2
RUN pip install shapely
RUN pip install scikit-image
RUN pip install pyclipper
RUN pip install lmdb
RUN pip install imgaug
copy app /app
WORKDIR /app
RUN mkdir /app/logs
RUN mkdir /app/model
RUN mkdir /app/model/tensorflow
RUN mkdir /app/model/pytorch
RUN mkdir /app/model/yolov5
RUN mkdir /app/model/yolov8
RUN mkdir /app/model/temp
cmd ["python3","app.py"]