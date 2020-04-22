from django.shortcuts import render, get_object_or_404, redirect
from django.http import HttpResponse, HttpResponseRedirect
from django.urls import reverse
from django.template import loader
import requests

from .models import Image, RectifyRequest, RectifyRequestImage

# Create your views here.
def index(request):
    print("trying to grab requests...")
    latest_requests = RectifyRequest.objects.order_by('-request_time')[:5]
    context = {
        'latest_requests': latest_requests
    }
    return render(request, 'index.html', context)

def db(request):
    req = RectifyRequest()
    req.brief = "A Rectify request about nothing."
    req.save()
    requests = RectifyRequest.objects.all()
    return render(request, "db.html", {"rectifyRequests": requests})

def detail(request, request_id):
    this_req = get_object_or_404(RectifyRequest, pk=request_id)
    images = []
    try:
        images =  RectifyRequestImage.objects.all().filter(request=this_req)
    except:
        print("Did not find associated images")
    
    context = {
        'request': this_req,
        'images': images
    }
    return render(request, 'detail.html', context)

def rectify(request):
    if (request.method != 'POST'):
        print("Redirect")
        return redirect('/rectifi', permanent=True)

    req = RectifyRequest()
    req.brief = request.POST.get('brief', '(None given)')
    req.save()

    print("Created rectifi request {0}".format(req.id))

    print("There are {0} files uploaded".format(len(request.FILES.getlist('images'))))
    for posted_image in request.FILES.getlist('images'):

        print("Saving image {0}".format(posted_image.name))
        print("Image is size {0} and type {1}".format(posted_image.size, posted_image.content_type))
        image = Image.objects.create(data=posted_image,
                                     name=posted_image.name)
        # image = Image(name=posted_image.name,
        #             data=posted_image.read())
        image.save()
        rect_img = RectifyRequestImage(request=req,
                                       img=image)
        rect_img.save()

    return HttpResponseRedirect(reverse('rectifi:detail', args=(req.id,)))

def results(request, request_id):
    return HttpResponse('Request {0} results page!'.format(request_id))
