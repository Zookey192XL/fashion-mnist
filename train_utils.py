
###################################################################################################

from time import time

import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler

from data_utils import Augmenter

###################################################################################################

def lr_scheduler_1(epoch):
    return 0.001 * min(1, np.exp(0.05 * (19 - epoch)))

def lr_scheduler_2(epoch):
    return 0.001 / 10 ** (epoch // 50)

###################################################################################################

def train_model(model, X_train, y_train, X_val, y_val, batch_size,
                epochs, aug_args, lr_scheduler=None, verbose=0, weights_dir="weights"):
    
    weights_file = f"{weights_dir}/{model.name}"
    
    if aug_args["mixup"]:
        weights_file += "_mix"

    if aug_args["re"]:
        weights_file += "_re"
        
    weights_file += ".h5"
        
    callbacks = []
    
    saver = ModelCheckpoint(
        weights_file, monitor="val_accuracy",
        save_best_only=True, save_weights_only=True)
    
    callbacks.append(saver)
    
    if lr_scheduler is not None:
        callbacks.append(LearningRateScheduler(lr_scheduler))
    
    augmenter = Augmenter(**aug_args)
    data_generator = augmenter.flow(X_train, y_train, batch_size)

    steps = int(len(X_train) / batch_size + 0.5)
    
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam", metrics=["accuracy"])
    
    start = time()
    
    hist = model.fit(
        data_generator,
        epochs=epochs,
        steps_per_epoch=steps,
        verbose=verbose,
        callbacks=callbacks,
        validation_data=(X_val, y_val))
    
    elapsed = time() - start
    
    train_info = hist.history
    train_info["model_name"] = model.name
    train_info["training_speed"] = elapsed / epochs
    
    for key, value in train_info.items():
        if isinstance(value, list):
            train_info[key] = [float(x) for x in value]
    
    model.load_weights(weights_file)
    
    return train_info

###################################################################################################

