from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse
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
    rect_request = get_object_or_404(RectifyRequest, pk=request_id)
    images = []
    try:
        images =  RectifyRequestImage.objects.get(rect_request=request_id)
    except:
        print("Did not find associated images")
    
    context = {
        'request': rect_request,
        'images': images
    }
    return render(request, 'detail.html', context)

def rectify(request, request_id):
    return HttpResponse('Hello from Python!')

def results(request, request_id):
    return HttpResponse('Request {0} results page!'.format(request_id))
