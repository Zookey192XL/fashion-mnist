
###################################################################################################

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json

from time import time
from collections import OrderedDict

from data_utils import load_fashion_mnist

from train_utils import train_model 
from train_utils import lr_scheduler_1, lr_scheduler_2

from models import SimpleNetV1, SimpleNetV2, ResNet18_Light, ResNet18_Full

###################################################################################################

WEIGHTS_DIR = "weights"
RESULTS_DIR = "results"

BATCH_SIZE = 128
EPOCHS_1, EPOCHS_2 = 100, 150

###################################################################################################

(X_train, y_train), (X_val, y_val), (X_test, y_test) = load_fashion_mnist()

input_shape = X_train.shape[1:]
num_classes = y_train.shape[1]

###################################################################################################

# base aug args
aug_base = dict(x_flip=True, dx=1, dy=1)

# mixup aug args
aug_mixup = dict(**aug_base, mixup=True, re=False)

# random erasing aug args
aug_re = dict(**aug_base, mixup=False, re=True)

# mixup + random erasing aug args
aug_both = dict(**aug_base, mixup=True, re=True)

###################################################################################################

exp_id = 0

for aug_args in [aug_mixup, aug_re, aug_both]:
    
    models = [
        SimpleNetV1(input_shape, num_classes),
        SimpleNetV2(input_shape, num_classes),
        ResNet18_Light(input_shape, num_classes),
        ResNet18_Full(input_shape, num_classes)]
    
    for i, model in enumerate(models):
        
        exp_id += 1
        print(f"Running experiment {exp_id:2}/12", end="\r")
        
        epochs = EPOCHS_1 if i < 2 else EPOCHS_2
        lr_scheduler = lr_scheduler_1 if i < 2 else lr_scheduler_2
        
        train_info = train_model(
            model, X_train, y_train, X_val, y_val,
            batch_size=BATCH_SIZE, epochs=epochs,
            lr_scheduler=lr_scheduler, aug_args=aug_args,
            weights_dir=WEIGHTS_DIR)
        
        y_true = y_test.argmax(axis=1)
        
        start = time()
        y_pred = model.predict(X_test, batch_size=128).argmax(axis=1)
        elapsed = time() - start
        
        train_info["params"] = model.count_params()
        train_info["inference_speed"] = elapsed / len(X_test)
        train_info["test_accuracy"] = float((y_pred == y_true).sum() / len(X_test))
        
        train_info_file = f"{RESULTS_DIR}/{model.name}"
    
        if aug_args["mixup"]:
            train_info_file += "_mix"

        if aug_args["re"]:
            train_info_file += "_re"

        train_info_file += ".json"
        
        info = OrderedDict()
        
        keys = ["model_name", "params", "test_accuracy",
                "training_speed", "inference_speed",
                "loss", "accuracy", "val_loss", "val_accuracy"]
        
        for key in keys:
            info[key] = train_info[key]
        
        with open(train_info_file, "w") as file:
            json.dump(info, file, indent=4)
            
###################################################################################################

