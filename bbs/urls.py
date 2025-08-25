from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

app_name = "bbs"
urlpatterns = [
    # 메인 페이지
    path("", views.MainPage.as_view(), name="index"),

    # 새로운 페이지를 위한 URL 패턴
    path('monthly/', views.MonthlyForecastPage.as_view(), name='monthly_forecast'),
    path('district/', views.DistrictForecastPage.as_view(), name='district_forecast'),
    path('price-range/', views.PriceRangeForecastPage.as_view(), name='price_range_forecast'),
    path('deposit/', views.DepositForecastPage.as_view(), name='deposit_forecast'),
]

# 개발 환경에서 정적 파일을 서빙하기 위한 설정 추가
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    # 또한, 미디어 파일을 위한 설정도 추가합니다.
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    # 게시판
  
