import os

import tensorflow as tf
from mediapipe_model_maker import gesture_recognizer

assert tf.__version__.startswith('2')


def train():
    dataset_path = os.path.join(bundle_dir, train_config.get("data_dir"))

    data = gesture_recognizer.Dataset.from_folder(
        dirname=dataset_path,
        hparams=gesture_recognizer.HandDataPreprocessingParams()
    )
    train_data, rest_data = data.split(0.8)
    validation_data, test_data = rest_data.split(0.5)

    hparams = gesture_recognizer.HParams(export_dir=train_config.get("export_dir"))
    options = gesture_recognizer.GestureRecognizerOptions(hparams=hparams)
    model = gesture_recognizer.GestureRecognizer.create(
        train_data=train_data,
        validation_data=validation_data,
        options=options
    )

    loss, acc = model.evaluate(test_data, batch_size=1)
    print(f"Test loss: {loss}, Test accuracy: {acc}")
    model.export_model()
