from django.urls import path

from . import views

app_name = 'rectifi'

urlpatterns = [
    path('', views.index, name='index'),
    path('db/', views.db, name='db'),
    path('<int:request_id>/', views.detail, name='detail'),
    path('rectify/', views.rectify, name='rectify'),
    path('<int:request_id>/results/', views.results, name='results')
]
