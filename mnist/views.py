from django.views import generic
from django.shortcuts import render
from io import BytesIO
import base64
from PIL import Image

import numpy as np
from .recognizer import Recognizer
import tensorflow as tf

from .models import Predict

graph = tf.get_default_graph()
recognizer = Recognizer()

class PaintView(generic.TemplateView):
    template_name = 'mnist/paint.html'

    def __init__(self):
        data = Predict.objects.get(pred_id=0)
        self.pred = data.pred
        data.pred = 10
        data.save()

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["pred"] = self.pred
        return context



    def post(self, request):
        base_name = request.POST['img-src']
        # data:image/png;base64,ファイル名 が格納されている
        base_name = base_name.replace("data:image/png;base64", "")

        file = BytesIO(base64.b64decode(base_name))

        img = Image.open(file).resize((28, 28)).convert('L')

        img = np.array(img) / 255
        img = img.reshape((1, 28, 28, 1))

        with graph.as_default():
            params = {
                'predict': recognizer.predict(img),
            }
        return render(request, 'mnist/paint.html', params)

