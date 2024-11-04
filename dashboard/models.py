from django.db import models


# Create your models here.
class RECORD(models.Model):
    # User, Requirement, Result, date
    title = models.CharField(max_length=255)
    content = models.TextField()
    date = models.DateField(auto_now_add=True)
    user = models.ForeignKey("auth.user", on_delete=models.CASCADE, null=True)
    prediction = models.CharField(max_length=255, default="Unknown")
    group = models.CharField(max_length=100,default="Unknown")


    def __str__(self):
        return self.title


class Feedback(models.Model):
    user = models.ForeignKey("auth.user", on_delete=models.CASCADE)
    date = models.DateField(auto_now_add=True)
    feedback = models.TextField()

    def __str__(self):
        return self.feedback
