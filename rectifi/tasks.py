from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.db import transaction
from workers import task
from time import sleep
from .models import Image, RectifyRequest, RectifyRequestImage, RectifyRequestResult
from .algorithm.fun import average_pels
import numpy as np

@task()
def go_and_rectify(request_id):
  req = None
  try:
    req = RectifyRequest.objects.get(pk=request_id)
  except:
    print("Could not retrieve request")
    return

  req.status = RectifyRequest.RequestStatus.PROCESSING
  req.save()

  images = []
  try:
    images =  RectifyRequestImage.objects.all().filter(request=req)
  except:
    print("Did not find associated images")
    req.status = RectifyRequest.RequestStatus.ABORTED
    req.save()
    return

  dest_fmt = ".bmp"
  dest = "request_{0}_average{1}".format(request_id, dest_fmt)

  #try:
  image_files = [np.fromfile(default_storage.open(i.img.name), dtype=np.uint8) for i in images]
  avg_output = average_pels(image_files, fmt=dest_fmt)
  avg_output_f = ContentFile(avg_output, name=dest)
  #default_storage.save(dest, avg_output_f)

  rslt_image = Image(name=avg_output_f.name, data=avg_output_f)
  result = RectifyRequestResult(request=req, img=rslt_image, notes="The average of interpolated pixels")
  req.status = RectifyRequest.RequestStatus.COMPLETE

  with transaction.atomic():
    rslt_image.save()
    result.save()
    req.save()
  #except ValueError as e:
  #  req.status = RectifyRequest.RequestStatus.ABORTED
  #  req.save()
  #  raise e
