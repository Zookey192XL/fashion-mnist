
###################################################################################################

from tensorflow.keras import layers as L
from tensorflow.keras import Sequential, Model

###################################################################################################

def SimpleNetV1(input_shape, num_classes, weights_file=None):
    
    model = Sequential([
        
        L.InputLayer(input_shape),
        
        L.Conv2D(128, 3, activation="relu"),
        L.Conv2D(128, 3, activation="relu"),
        L.MaxPooling2D(),
        
        L.Conv2D(128, 3, activation="relu"),
        L.Conv2D(128, 3, activation="relu"),
        L.MaxPooling2D(),
        
        L.Flatten(),
        
        L.Dense(128, activation="relu"),
        L.Dense(num_classes, activation="softmax")],
        
        name="simple_net_v1")
    
    if weights_file is not None:
        model.load_weights(weights_file)
    
    return model

###################################################################################################

def SimpleNetV2(input_shape, num_classes, weights_file=None):
    
    model = Sequential([
        
        L.InputLayer(input_shape),
         
        L.Conv2D(32, 3, strides=1, padding="same"),
        L.BatchNormalization(),
        L.Activation("relu"),
        
        L.Conv2D(32, 3, strides=2, padding="same"),
        L.BatchNormalization(),
        L.Activation("relu"),
        
        L.Conv2D(64, 3, strides=1, padding="same"),
        L.BatchNormalization(),
        L.Activation("relu"),
        
        L.Conv2D(64, 3, strides=2, padding="same"),
        L.BatchNormalization(),
        L.Activation("relu"),
        
        L.Conv2D(128, 3, strides=1, padding="same"),
        L.BatchNormalization(),
        L.Activation("relu"),
        
        L.Conv2D(256, 3, strides=2, padding="same"),
        L.BatchNormalization(),
        L.Activation("relu"),
        
        L.GlobalAveragePooling2D(),
        
        L.Dense(num_classes, activation="softmax")],
        
        name="simple_net_v2")
    
    if weights_file is not None:
        model.load_weights(weights_file)
    
    return model

###################################################################################################

_RN_CONV_OPTS = dict(padding="same", use_bias=False)

def _res_net_preact_block(x, filters, stride):
    
    shortcut_1 = x
    
    x = L.BatchNormalization()(x)
    x = L.Activation("relu")(x)
    
    shortcut_2 = x
    
    x = L.Conv2D(filters, 3, stride, **_RN_CONV_OPTS)(x)
    x = L.BatchNormalization()(x)
    x = L.Activation("relu")(x)
    
    x = L.Conv2D(filters, 3, 1, **_RN_CONV_OPTS)(x)
    
    shortcut = shortcut_1
    input_channels = shortcut.get_shape().as_list()[-1]
    
    if (stride != 1) or (filters != input_channels):
        shortcut = L.Conv2D(filters, 1, stride, **_RN_CONV_OPTS)(shortcut_2)
        
    x = L.Add()([x, shortcut])
    
    return x

def _res_net_18(input_shape, num_classes,
                initial_filters, filter_sizes, name, weights_file):
    
    # input
    x = input_ = L.Input(input_shape)
    
    # initial convolution
    x = L.Conv2D(initial_filters, 3, **_RN_CONV_OPTS)(x)
    
    # res net blocks
    
    for i, filters in enumerate(filter_sizes):
    
        s = 2 if i > 0 else 1
        x = _res_net_preact_block(x, filters, stride=s)
        x = _res_net_preact_block(x, filters, stride=1)
    
    # final pooling
    x = L.GlobalAveragePooling2D()(x)
    
    # output
    output = L.Dense(num_classes, activation="softmax")(x)
    
    # create model
    model = Model(inputs=input_, outputs=output, name=name)
    
    if weights_file is not None:
        model.load_weights(weights_file)
    
    return model

def ResNet18_Light(input_shape, num_classes, weights_file=None):
    
    return _res_net_18(input_shape, num_classes,
                       initial_filters=32,
                       filter_sizes=[32, 64, 128, 256],
                       name="res_net_18_light",
                       weights_file=weights_file)

def ResNet18_Full(input_shape, num_classes, weights_file=None):
    
    return _res_net_18(input_shape, num_classes,
                       initial_filters=64,
                       filter_sizes=[64, 128, 256, 512],
                       name="res_net_18_full",
                       weights_file=weights_file)

###################################################################################################

