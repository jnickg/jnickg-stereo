from django.contrib import admin

# Register your models here.
from .models import Image, RectifyRequest, RectifyRequestImage

admin.site.register(Image)
admin.site.register(RectifyRequest)
admin.site.register(RectifyRequestImage)
