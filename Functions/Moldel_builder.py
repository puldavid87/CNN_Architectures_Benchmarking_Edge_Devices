import tensorflow as tf
import keras

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications import MobileNetV2
from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications import InceptionV3


img_height = 224
img_width = 224



img_augmentation = tf.keras.models.Sequential(
    [
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.20),
        tf.keras.layers.experimental.preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        #tf.keras.layers.RandomContrast(factor=0.1),
    ],
    name="img_augmentation",
)

def build_Efficient_model(num_classes,aprov_pre):
    """Builds a transfer learning model for image classification
    Args:
        num_classes: {Number of classes for the classification task} 
        aprov_pre: {Whether to use preprocessing augmentation}
    Returns: 
        model: {The Keras model}
    Processing Logic:
        - Builds the EfficientNetB0 model with pretrained weights
        - Freezes the pretrained weights
        - Adds global average pooling, dropout and dense output layers
        - Compiles the model with Adam optimizer and categorical crossentropy loss
    """
    inputs = tf.keras.layers.Input(shape=(img_height,img_width,3))
    if aprov_pre==True:
        x = img_augmentation(inputs)
        model = EfficientNetB0(include_top=False, input_tensor=x, weights="imagenet")
        print("preprocessing:",aprov_pre)
    else:
        model = EfficientNetB0(include_top=False, input_tensor=inputs, weights="imagenet")
        print("preprocessing:",aprov_pre)
    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = tf.keras.layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = tf.keras.layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="pred")(x)

    # Compile
    model = tf.keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model

def build_Mobilenet_model(num_classes,aprov_pre):
    """Builds a transfer learning model for image classification
    Args:
        num_classes: {Number of classes for the classification task} 
        aprov_pre: {Whether to use preprocessing augmentation}
    Returns: 
        model: {The Keras model}
    Processing Logic:
        - Builds the EfficientNetB0 model with pretrained weights
        - Freezes the pretrained weights
        - Adds global average pooling, dropout and dense output layers
        - Compiles the model with Adam optimizer and categorical crossentropy loss
    """
    inputs = tf.keras.layers.Input(shape=(img_height,img_width,3))
    if aprov_pre==True:
        x = img_augmentation(inputs)
        model = MobileNetV2(include_top=False, input_tensor=x, weights="imagenet")
        print("preprocessing:",aprov_pre)
    else:
        model = MobileNetV2(include_top=False, input_tensor=inputs, weights="imagenet")
        print("preprocessing:",aprov_pre)
    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = tf.keras.layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = tf.keras.layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="pred")(x)

    # Compile
    model = tf.keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def build_VGG16_model(num_classes,aprov_pre):
    """Builds a transfer learning model for image classification
    Args:
        num_classes: {Number of classes for the classification task} 
        aprov_pre: {Whether to use preprocessing augmentation}
    Returns: 
        model: {The Keras model}
    Processing Logic:
        - Builds the VGG16 model with pretrained weights
        - Freezes the pretrained weights
        - Adds global average pooling, dropout and dense output layers
        - Compiles the model with Adam optimizer and categorical crossentropy loss
    """
    inputs = tf.keras.layers.Input(shape=(img_height,img_width,3))
    if aprov_pre==True:
        x = img_augmentation(inputs)
        model = VGG16(include_top=False, input_tensor=x, weights="imagenet")
        print("preprocessing:",aprov_pre)
    else:
        model = VGG16(include_top=False, input_tensor=inputs, weights="imagenet")
        print("preprocessing:",aprov_pre)
    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = tf.keras.layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = tf.keras.layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="pred")(x)

    # Compile
    model = tf.keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def build_Resnet_model(num_classes,aprov_pre):
    inputs = tf.keras.layers.Input(shape=(img_height,img_width,3))
    inputs_re = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1./127.5,offset=-1)(inputs)    
    if aprov_pre==True:
        x = img_augmentation(inputs_re)
        model_input = ResNet50V2(include_top=False, input_tensor=x, weights="imagenet")
        print("preprocessing:",aprov_pre)
    else:
        model_input = ResNet50V2(include_top=False, input_tensor=inputs_re, weights="imagenet")
        print("preprocessing:",aprov_pre)
    # Freeze the pretrained weights
    model_input.trainable = False
    model=tf.keras.Sequential([
	model_input,
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(num_classes, activation="softmax")
    ])
    # Compile
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def build_Inception_model(num_classes,aprov_pre):
    inputs = tf.keras.layers.Input(shape=(img_height,img_width,3))
    inputs_re = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1./255)(inputs)    
    if aprov_pre==True:
        y = img_augmentation(inputs_re)
        model_input = InceptionV3(include_top=False, input_tensor=y, weights="imagenet")
        print("preprocessing:",aprov_pre)
    else:
        model_input = InceptionV3(include_top=False, input_tensor=inputs_re, weights="imagenet")
        print("preprocessing:",aprov_pre)
    # Freeze the pretrained weights
    model_input.trainable = False
    x = model_input.output
    x = tf.keras.layers.Flatten()(x)
    predictions = tf.keras.layers.Dense(num_classes,activation = "softmax")(x)
    model=tf.keras.models.Model(inputs,predictions)
    # Compile
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def unfreeze_model(model,num):
    # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
    for layer in model.layers[num:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
