from keras.models import load_model
import numpy as np

class Recognizer():
    def __init__(self):
        self.model = load_model('mnist_cnn.h5')


    def predict(self, X):
        self.model._make_predict_function()
        return np.argmax(self.model.predict(X))