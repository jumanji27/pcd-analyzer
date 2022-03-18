FROM continuumio/miniconda3:4.10.3p1

WORKDIR /pcd-analyzer

RUN conda config --append channels conda-forge
RUN conda install numpy pandas pyntcloud

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD python app.py
