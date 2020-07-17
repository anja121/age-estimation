FROM tensorflow/tensorflow:latest-gpu
COPY requirements.txt /age-estimation/requirements.txt
WORKDIR age-estimation
COPY . /age-estimation
RUN pip install -r requirements.txt
CMD ["python", "train.py"]