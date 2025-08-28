from django.contrib import admin
from django.urls import path,include
from django.conf.urls.static import static
from django.shortcuts import render

def root_index(request):
    return render(request, "index.html")   # templates/index.html 사용

urlpatterns = [
    path('admin/', admin.site.urls),
    # path('bbs/', include('bbs.urls')),
    path('', include('bbs.urls')),

]