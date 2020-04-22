from django.db import models

class Image(models.Model):
    name = models.CharField(unique=True, max_length=127)
    data = models.BinaryField(blank=True)
    def __str__(self):
        return self.name

class RectifyRequest(models.Model):
    brief = models.CharField(max_length=255)
    request_time = models.DateTimeField(auto_now_add=True)
    def __str__(self):
        return "Rectify Request {0} ({1})".format(self.id, self.request_time)

class RectifyRequestImage(models.Model):
    request = models.ForeignKey(RectifyRequest, on_delete=models.CASCADE)
    img = models.ForeignKey(Image, on_delete=models.CASCADE)
    def __str__(self):
            return "Image {0} from Rectify Request {1}".format(self.img.name, self.request.id)
