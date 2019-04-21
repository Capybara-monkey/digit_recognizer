from django.db import models

# Create your models here.
class Predict(models.Model):
    pred_id = models.IntegerField(default=0)
    pred = models.IntegerField()