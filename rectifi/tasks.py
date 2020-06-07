from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.db import transaction
from workers import task
from time import sleep
from .models import Image, RectifyRequest, RectifyRequestImage, RectifyRequestResult
from .algorithm.average_pels import average_pels
from .algorithm.stereo import rectify
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

  try:
    image_files = [np.fromfile(default_storage.open(i.img.data.name), dtype=np.uint8) for i in images]
    parameters = {
      "fmg":".bmp",
      "debug": False,
      "do_resize": True,
      "max_wide_len": 1000
    }
    outputs = rectify(image_files, params=parameters)

    db_objects = []
    for filename in outputs:
      print(f"Handling rectification output: {filename}...")
      local_output_file = open(filename, "rb")
      data_file = ContentFile(local_output_file.read(), name=filename)
      local_output_file.close()
      rslt_image = Image(name=data_file.name, data=data_file)
      db_objects.append(rslt_image)
      result = RectifyRequestResult(request=req, img=rslt_image, notes=filename)
      db_objects.append(result)

    req.status = RectifyRequest.RequestStatus.COMPLETE

    with transaction.atomic():
      for obj in db_objects:
        obj.save()
      req.save()
  except ValueError as e:
   req.status = RectifyRequest.RequestStatus.ABORTED
   req.save()
   raise e
