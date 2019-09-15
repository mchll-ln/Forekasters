from tensorflow.python.keras import Sequential
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.layers import GlobalAveragePooling2D, Dense


def get_model(input_shape=(128, 128, 16), classes=4, activation='sigmoid'):
    resnet = ResNet50(weights=None, include_top=False, input_shape=input_shape)
    model = Sequential()
    model.add(resnet)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(units=classes, activation=activation, kernel_initializer="he_normal"))

    return model


def get_model_vgg(input_shape=(128, 128, 16), classes=4, activation='sigmoid'):
    resnet = VGG16(weights=None, include_top=False, input_shape=input_shape)
    resnet.summary()
    model = Sequential()
    model.add(resnet)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(units=classes, activation=activation, kernel_initializer="he_normal"))

    return model


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
