FROM ubuntu:22.04

WORKDIR /text2code
RUN mkdir text2code
RUN apt-get update -y && \
  apt-get install -y python3 python3-pip

COPY requirements.txt /text2code/requirements.txt
# install all required packages
RUN pip3 install -r /text2code/requirements.txt
# copy the app directory
COPY app /text2code/app
COPY run.py /text2code/run.py

# start the container
CMD ["python3", "/text2code/run.py"]