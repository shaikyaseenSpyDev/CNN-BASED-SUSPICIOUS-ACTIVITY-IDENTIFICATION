from django.urls import path

from . import views

urlpatterns = [path("index.html", views.index, name="index"),
               path("AdminLogin.html", views.AdminLogin, name="AdminLogin"),	      
               path("AdminLoginAction", views.AdminLoginAction, name="AdminLoginAction"),
	       path("TrainCNN", views.TrainCNN, name="TrainCNN"),
	       path("DetectActivity.html", views.DetectActivity, name="DetectActivity"),
	       path("DetectActivityAction", views.DetectActivityAction, name="DetectActivityAction"),	      
]
