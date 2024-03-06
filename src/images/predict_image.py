import numpy as np
from keras.preprocessing import image
from src.images.data_augmentation import IMG_SIZE, train_generator
from src.images.load_models import models

NUMBER_OF_CLASSES = 53


def predict_image_class(img):
    img = image.load_img(img, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.

    predictions = np.zeros((1, NUMBER_OF_CLASSES))
    for model in models:
        model_predictions = model.predict(img_array)
        predictions += model_predictions

    ensemble_predictions = predictions / len(models)
    predicted_class_index = np.argmax(ensemble_predictions)
    predicted_class_label = list(train_generator.class_indices.keys())[predicted_class_index]

    return predicted_class_label
