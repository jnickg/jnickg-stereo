from django.urls import path, include
from django.shortcuts import redirect
from django.views.generic import RedirectView
from django.contrib import admin

admin.autodiscover()

import rectifi.views

# To add a new path, first import the app:
# import blog
#
# Then add the new path:
# path('blog/', blog.urls, name="blog")
#
# Learn more here: https://docs.djangoproject.com/en/2.1/topics/http/urls/

urlpatterns = [
    path("", RedirectView.as_view(pattern_name='rectifi/', permanent=False)),
    path("admin/", admin.site.urls),
    path("rectifi/", include("rectifi.urls"))
]
