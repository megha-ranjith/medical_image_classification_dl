import tensorflow as tf
import numpy as np

def load_feature_extractor():
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        pooling='avg',
        input_shape=(224,224,3)
    )
    return base_model

def extract_features(model, images):
    images = np.stack([np.stack((img,)*3, axis=-1) for img in images])
    return model.predict(images)
