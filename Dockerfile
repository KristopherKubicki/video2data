FROM gcr.io/tensorflow/tensorflow:latest-gpu

#VOLUME ["/app"]

RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install -y python3-pip python3-dev 
RUN pip3 install --upgrade pip
RUN pip3 install tensorflow-gpu

ADD . /app
WORKDIR /app

RUN apt-get install -y libdlib-data libdlib-dev
RUN apt-get install -y cmake
RUN apt-get install -y python3-numpy libopencv-dev python-opencv

RUN pip3 install -r requirements.txt
#RUN pip3 install .
