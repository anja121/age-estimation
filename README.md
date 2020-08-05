### Consistent Rank Logits Age Estimation
[https://arxiv.org/abs/1901.07884](https://arxiv.org/abs/1901.07884)

Train and Demo code for the paper above using tensorflow 2.0 and tf.keras.

#### Data
Morph Dataset

* Image preprocessing: face alignment (dlib)
* original labels (Age) 16-70
* scaled labels (scaled) 0-54
* csv annotation file example:

|        | ID           | Path  | Age | scaled |
| ------------- |:-------------:| -----:|-----:|-----:|
| 0      | 134083 | Album2/134083_05M41.JPG | 41 | 25 |


#### Models

* VGG
* ResNet

#### Configuration
Config files are in  folder /configs 

* train_config
```
{
  "img_dir": "/path/to/images/dir/dataset/",
  "csv_path": "/path/to/annotation/file.csv",
  "task_weighting": true,  # include sample weights in loss fn
  "checkpoint": true,  # save checkpoints while training
  "img_size": 197,  # img height & width
  "logs": true,    # save tensorboard logs
  "batch_size" : 16, 
  "num_epochs" : 30,
  "model_prefix": "age_estimator_", # model name prefix
  "arch_type": "resnet",  # "resnet" or "vgg" (for now)
  "arch_subtype": 50     # for resnet (18,34,50,101 or 152)
}
```

* predict_config
```
{
  "n_classes": 50, # number of different age values 
  "start_label": 16, # lowest age value (original labels)
  "end_label": 65, # highest age value (original labels)
  "img_size": 197, # img height & width
  "threshold": 0.4, # probability threshold 
  "model_path": "/path/to/trained/model.h5"
}
```
#### Usage

With tensorflow docker:

prerequisite: tensorflow-gpu docker image [https://www.tensorflow.org/install/docker](https://www.tensorflow.org/install/docker)

```console
docker build -t age-estimation .
docker run --gpus all -v /path/to/dataset:/age-estimation/dataset/
```

With tensorflow:

prerequisite: tensorflow-gpu 2.0
```console
pip install -r requirements.txt
python train.py
```

Train and save trained model (.h5 format).

Training configuration can be set in configs/train_config.json

#### Demo

```
python demo.py -p /path/to/test/images/dir/
```
Loads images from given folder path and creates and saves results for 
each image in json format. 

Demo configuration can be set in configs/predict_config.json

Result json:
```
{
  raw_probas:[],
  raw_logits:[],
  threshold:float,
  thresholded_probas:[],
  result_value:int
}
```
