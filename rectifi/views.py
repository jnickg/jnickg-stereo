from django.shortcuts import render
from django.http import HttpResponse
import requests

from .models import Image, RectifyRequest

# Create your views here.
def index(request):
    #r = requests.get('http://httpbin.org/status/420')
    #print(r.text)
    #return HttpResponse('<pre>' + r.text + '</pre>')
    return HttpResponse('Hello from Python!')
    # return render(request, "index.html")


def db(request):

    req = RectifyRequest()
    req.brief = "A Rectify request about nothing."
    req.save()

    requests = RectifyRequest.objects.all()

    return render(request, "db.html", {"rectifyRequests": requests})
