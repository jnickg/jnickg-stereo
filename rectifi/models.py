from django.db import models
from django.utils.translation import gettext_lazy as _

class Image(models.Model):
    name = models.CharField(max_length=127)
    data = models.ImageField(blank=False, editable=True)

    def __str__(self):
        return "Image {0} - {1}".format(self.id, self.name)

class RectifyRequest(models.Model):
    class RequestStatus(models.TextChoices):
        SUBMITTED = 'SUB', _('Submitted')
        PROCESSING = 'PRC', _('Processing')
        ABORTED = 'BAD', _('Aborted due to input error')
        COMPLETE = 'YAY', _('Completed')

    brief = models.CharField(max_length=255)
    request_time = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=32, choices=RequestStatus.choices, default=RequestStatus.SUBMITTED)

    def __str__(self):
        return "Rectify Request {0} ({1})".format(self.id, self.request_time)

    def done_processing(self):
        return self.status in {
            self.RequestStatus.ABORTED,
            self.RequestStatus.COMPLETE
        }

class RectifyRequestImage(models.Model):
    request = models.ForeignKey(RectifyRequest, on_delete=models.CASCADE)
    img = models.ForeignKey(Image, on_delete=models.CASCADE)

    def __str__(self):
            return "Image {0} from Rectify Request {1}".format(self.img.name, self.request.id)
