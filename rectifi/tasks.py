from workers import task
from time import sleep
from .models import Image, RectifyRequest, RectifyRequestImage

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

  # DO SOMETHING WITH IMAGES!!!
  sleep(20)
  for img in images:
    pass
  
  req.status = RectifyRequest.RequestStatus.COMPLETE
  req.save()
