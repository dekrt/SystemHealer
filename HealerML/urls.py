from django.conf.urls.static import static
from django.conf import settings
from django.urls import path
from . import views

urlpatterns = [
    # path('admin/', admin.site.urls),
    path("", views.index, name="index"),
    path("upload", views.data, name="data"),
    path('download/<str:file_name>/', views.download, name='download'),
    path('download_model/<str:file_name>/', views.download_model, name='download_model')
]
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
