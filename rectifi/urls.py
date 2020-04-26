from django.urls import path

from . import views

app_name = 'rectifi'

urlpatterns = [
    path('', views.index, name='index'),
    path('about/', views.about, name='about'),
    path('db/', views.db, name='db'),
    path('requests/<int:request_id>/detail', views.detail, name='detail'),
    path('rectify/', views.rectify, name='rectify'),
    path('requests/', views.requests, name='requests'),
    path('requests/<int:request_id>/results/', views.results, name='results')
]
