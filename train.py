"""
Train age estimation model
model and training configuration should be set in configs/train_config.json
"""
from utils.utils import read_config_file, get_callbacks, get_model
from utils.datasets_utils import load_csv, train_valid_split
from utils.morph_dataset import MorphDataset
from datetime import datetime


def main():
    # read train config
    conf = read_config_file("configs/train_config_copy.json")
    model_suffix = datetime.now().strftime("%Y%m%d-%H%M%S")

    # get datasets
    data_df = load_csv(csv_path=conf["csv_path"],
                       shuffle=True,
                       seed=42)

    train_data_df, valid_data_df = train_valid_split(data_df=data_df,
                                                     train_percent=0.8,
                                                     seed=42)

    train_data = MorphDataset(data_df=train_data_df,
                              img_dir=conf["img_dir"],
                              img_size=(conf["img_size"], conf["img_size"]),
                              channels=3,
                              seed=42)
    valid_data = MorphDataset(data_df=valid_data_df,
                              img_dir=conf["img_dir"],
                              img_size=(conf["img_size"], conf["img_size"]),
                              channels=3,
                              seed=42)

    train_size = train_data.size
    valid_size = valid_data.size
    label_array = train_data.label_array if conf["task_weighting"] else None
    n_classes = len(train_data.classes)

    train_data = train_data.batch(batch_size=conf["batch_size"])
    valid_data = valid_data.batch(batch_size=conf["batch_size"])

    # get model
    model = get_model(conf=conf,
                      n_classes=n_classes,
                      label_array=label_array)

    model.summary()

    # get train callback functions
    callbacks = get_callbacks(conf, model_suffix)

    # train model
    model.fit(train_data,
              epochs=conf["num_epochs"],
              steps_per_epoch=train_size//conf["batch_size"],
              validation_data=valid_data if valid_data is not None else None,
              validation_steps=valid_size//conf["batch_size"] if valid_data is not None else None,
              callbacks=callbacks)

    # save model
    model_name_prefix = str(conf["model_prefix"]) + str(conf["arch_type"])
    if conf["arch_type"] == "resnet":
        model_name_prefix += str(conf["arch_subtype"])

    model.save(model_name_prefix + "_" + model_suffix + '.h5')


if __name__ == '__main__':
    main()

