"""
Example using Optuna, Dask, and Dask-Optuna to tune hyperparameters for a
Keras convolutional neural network.

Adapted from https://github.com/optuna/optuna/blob/master/examples/tfkeras_simple.py
"""
import warnings

from tensorflow.keras.backend import clear_session
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop

import optuna
import joblib
from dask.distributed import Client
import dask_optuna

# Ignore some warnings coming from Tensorflow
warnings.simplefilter("ignore", category=(FutureWarning, RuntimeWarning))


N_TRAIN_EXAMPLES = 3000
N_VALID_EXAMPLES = 1000
BATCHSIZE = 128
CLASSES = 10
EPOCHS = 10


def objective(trial):
    # Clear clutter from previous tf.keras session graphs.
    clear_session()

    (x_train, y_train), (x_valid, y_valid) = mnist.load_data()
    img_x, img_y = x_train.shape[1], x_train.shape[2]
    x_train = (
        x_train.reshape(-1, img_x, img_y, 1)[:N_TRAIN_EXAMPLES].astype("float32") / 255
    )
    x_valid = (
        x_valid.reshape(-1, img_x, img_y, 1)[:N_VALID_EXAMPLES].astype("float32") / 255
    )
    y_train = y_train[:N_TRAIN_EXAMPLES]
    y_valid = y_valid[:N_VALID_EXAMPLES]
    input_shape = (img_x, img_y, 1)

    model = Sequential()
    model.add(
        Conv2D(
            filters=trial.suggest_categorical("filters", [32, 64]),
            kernel_size=trial.suggest_categorical("kernel_size", [3, 5]),
            strides=trial.suggest_categorical("strides", [1, 2]),
            activation=trial.suggest_categorical("activation", ["relu", "linear"]),
            input_shape=input_shape,
        )
    )
    model.add(Flatten())
    model.add(Dense(CLASSES, activation="softmax"))

    # We compile our model with a sampled learning rate.
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=RMSprop(lr=lr),
        metrics=["accuracy"],
    )

    model.fit(
        x_train,
        y_train,
        validation_data=(x_valid, y_valid),
        shuffle=True,
        batch_size=BATCHSIZE,
        epochs=EPOCHS,
        verbose=False,
    )

    # Evaluate the model accuracy on the validation set.
    score = model.evaluate(x_valid, y_valid, verbose=0)
    return score[1]


if __name__ == "__main__":

    with Client() as client:
        print(f"Dask dashboard is available at {client.dashboard_link}")

        storage = dask_optuna.DaskStorage()
        study = optuna.create_study(storage=storage, direction="maximize")
        with joblib.parallel_backend("dask"):
            study.optimize(objective, n_trials=20, n_jobs=-1)

        print("Number of trials: {}".format(len(study.trials)))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: {}".format(trial.value))

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
