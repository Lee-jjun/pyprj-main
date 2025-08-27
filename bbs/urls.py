from django.urls import path
from . import views
from .views_controller import pfpa_views
from django.conf import settings
from django.conf.urls.static import static

app_name = "bbs"
urlpatterns = [
    # 메인 페이지
    path("", views.MainPage.as_view(), name="index"),

    # 기존 페이지를 위한 URL 패턴 (기존 코드 유지)
    path('monthly/', views.MonthlyForecastPage.as_view(), name='monthly_forecast'),
    path('district/', views.DistrictForecastPage.as_view(), name='district_forecast'),
    path('price-range/', views.PriceRangeForecastPage.as_view(), name='price_range_forecast'),
    path('deposit/', views.DepositForecastPage.as_view(), name='deposit_forecast'),

    #path('data-source/', views.DataSourcePage.as_view(), name='data_source'),
    path('pfpa/', pfpa_views.PriceForPerAreaPage.as_view(), name='price_for_per_area_page'),
    path('pfpa/api/', pfpa_views.PriceForPerArea.as_view(), name='price_for_per_area'),
    path('pfpa-test/api/', pfpa_views.PriceForPerAreaTestPage.as_view(), name='price_for_per_area_test'),
    
    # data_source 관련 URL 패턴은 제거하고 새로운 URL 패턴으로 대체합니다.
    # path('data-source/', views.data_source_view, name='data_source'),
    # path('data-source-data/', views.get_data_source_data, name='get_data_source_data'),
    
    # ✨ 새로운 뷰에 대한 URL 패턴을 추가합니다.
    #path('data_visualize/', views.DataVisualizePage.as_view(), name='data_visualize'),
    #path('get_data_visualize_data/', views.get_data_visualize_data, name='get_data_visualize_data'),

]

# 개발 환경에서 정적 파일을 서빙하기 위한 설정 추가
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    # 또한, 미디어 파일을 위한 설정도 추가합니다.
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
