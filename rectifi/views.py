from django.shortcuts import render, get_object_or_404, redirect
from django.http import HttpResponse, HttpResponseRedirect
from django.urls import reverse
from django.template import loader
from django.db import transaction
import requests

from .models import Image, RectifyRequest, RectifyRequestImage, RectifyRequestResult
from .tasks import go_and_rectify

# Create your views here.
def index(request):
    print("trying to grab requests...")
    latest_requests = RectifyRequest.objects.order_by('-request_time')[:5]
    context = {
        'latest_requests': latest_requests
    }
    return render(request, 'index.html', context)

def db(request):
    requests = RectifyRequest.objects.all()
    return render(request, "db.html", {"rectifyRequests": requests})

def detail(request, request_id):
    this_req = get_object_or_404(RectifyRequest, pk=request_id)
    images = []
    try:
        images = RectifyRequestImage.objects.all().filter(request=this_req)
    except:
        print("Did not find associated images")

    results = []
    try:
        results = RectifyRequestResult.objects.all().filter(request=this_req)
    except:
        print("Did not find associated results")
    
    context = {
        'request': this_req,
        'images': images,
        'results': results,
    }
    return render(request, 'detail.html', context)

def rectify(request):
    if (request.method != 'POST'):
        print("Redirect")
        return redirect('/rectifi', permanent=True)

    req = RectifyRequest()
    req.brief = request.POST.get('brief', '(None given)')
    print("Created rectifi request {0}".format(req.id))

    print("Including {0} files uploaded".format(len(request.FILES.getlist('images'))))
    images = []
    rect_images = []
    for posted_image in request.FILES.getlist('images'):
        print("Saving image {0}".format(posted_image.name))
        print("Image is size {0} and type {1}".format(posted_image.size, posted_image.content_type))
        image = Image(data=posted_image,
                      name=posted_image.name)
        rect_img = RectifyRequestImage(request=req,
                                       img=image)

        images.append(image)
        rect_images.append(rect_img)

    with transaction.atomic():
        req.save()
        for i in images: i.save()
        for r in rect_images: r.save()

    go_and_rectify(req.id)

    return HttpResponseRedirect(reverse('rectifi:detail', args=(req.id,)))

def results(request, request_id):
    return HttpResponse('Request {0} results page!'.format(request_id))
