# views.py

import logging
from django.shortcuts import render
from django.urls import reverse_lazy
from django.views.generic import ListView, DetailView, CreateView, UpdateView, TemplateView, FormView
from django.db import transaction
from django import forms
from django.core.paginator import Paginator
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

# Plotly 관련 라이브러리
import plotly.graph_objs as go
import json
import plotly

# Django 모델 및 비즈니스 로직 임포트
from .forms import PostForm
from bbs.dao.bbs_models import Bbs, BbsFile
from bbs.db.bbs_mysql import get_bbs_with_rownum, get_total_bbs_count
from bbs.biz.real_estate_price_forecast import engine
from bbs.biz.price_range_forecast import run_analysis

logger = logging.getLogger(__name__)

#####################################
# 파일 저장 헬퍼
#####################################
def save_files(bbs_instance, files):
    """
    게시글에 파일을 첨부하는 헬퍼 함수
    """
    for f in files:
        BbsFile.objects.create(
            bbs=bbs_instance,
            file=f,
            orig_name=f.name,
        )

#####################################
# 게시판 목록 조회
#####################################
class BbsLV(ListView):
    model = Bbs
    template_name = 'bbs/bbs_list.html'
    context_object_name = 'bbs_list'
    paginate_by = 20

    def dispatch(self, request, *args, **kwargs):
        logger.debug("BbsLV dispatch 호출됨")
        return super().dispatch(request, *args, **kwargs)

    def get_queryset(self):
        page = self.request.GET.get('page', 1)
        try:
            page = int(page)
        except ValueError:
            page = 1
        offset = (page - 1) * self.paginate_by
        return get_bbs_with_rownum(offset=offset, limit=self.paginate_by)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        total = get_total_bbs_count()
        page_obj = context['page_obj']
        current_page = page_obj.number
        start_index = total - ((current_page - 1) * self.paginate_by)
        context['total'] = total
        context['start_index'] = start_index

        paginator = context['paginator']
        total_pages = paginator.num_pages
        block_size = 3
        start_page = ((current_page - 1) // block_size) * block_size + 1
        end_page = min(start_page + block_size - 1, total_pages)

        context.update({
            'start_page': start_page,
            'end_page': end_page,
            'has_prev_block': start_page > 1,
            'has_next_block': end_page < total_pages,
            'prev_block_page': start_page - 1,
            'next_block_page': end_page + 1,
            'page_range': range(start_page, end_page + 1),
        })
        return context

#####################################
# 게시판 상세
#####################################
class BbsDetailView(DetailView):
    model = Bbs
    template_name = 'bbs/bbs_detail.html'
    context_object_name = 'bbs'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['file_list'] = self.object.files.all()
        return context

#####################################
# 게시판 신규 등록
#####################################
class BbsCreateView(CreateView):
    model = Bbs
    form_class = PostForm
    template_name = 'bbs/bbs_form.html'
    success_url = reverse_lazy('bbs:index')

    @transaction.atomic
    def form_valid(self, form):
        response = super().form_valid(form)
        bbs_instance = self.object
        if not bbs_instance.group_id:
            bbs_instance.group_id = bbs_instance.id
        bbs_instance.save()
        save_files(bbs_instance, self.request.FILES.getlist('file'))
        return response

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['is_edit'] = False
        return context

#####################################
# 게시판 수정
#####################################
class BbsUpdateView(UpdateView):
    model = Bbs
    form_class = PostForm
    template_name = 'bbs/bbs_form.html'
    success_url = reverse_lazy('bbs:index')

    @transaction.atomic
    def form_valid(self, form):
        response = super().form_valid(form)
        bbs_instance = self.object
        if not bbs_instance.group_id:
            bbs_instance.group_id = bbs_instance.id
            bbs_instance.save()
        save_files(bbs_instance, self.request.FILES.getlist('file'))
        return response

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['is_edit'] = True
        return context

#####################################
# Deeplearning 뷰
#####################################
class DeeplearningForm(forms.Form):
    run = forms.BooleanField(required=False, initial=True, widget=forms.HiddenInput)

class Deeplearning(FormView):
    template_name = 'bbs/deeplearning.html'
    form_class = DeeplearningForm
    success_url = reverse_lazy('bbs:deeplearning')

    def form_valid(self, form):
        logger.debug("Deeplearning 실행 시작")
        try:
            result = engine()
        except Exception as e:
            logger.error(f"engine() 실행 오류: {e}")
            result = f"Error: {e}"
        logger.debug("Deeplearning 실행 완료")
        return render(self.request, self.template_name, {'form': form, 'result': result})

#####################################
# 예측 페이지 (월별, 구별, 가격대, 보증금)
# 이 부분은 변경 사항이 없습니다.
#####################################
class MonthlyForecastPage(TemplateView):
    template_name = 'bbs/monthly_forecast.html'

    def post(self, request, *args, **kwargs):
        year = request.POST.get("year")
        month = request.POST.get("month")
        result = {
            "labels": ["1월", "2월", "3월", "4월", "5월"],
            "datasets": [{"label": "예측 가격","data": [1000, 1030, 1050, 1070, 1100],"borderColor": "rgb(75, 192, 192)","tension": 0.1}]
        }
        predicted_price = result["datasets"][0]["data"][-1]
        return render(request, self.template_name, {"result": result, "predicted_price": predicted_price, "year": year, "month": month})

class DistrictForecastPage(TemplateView):
    template_name = 'bbs/district_forecast.html'

    def post(self, request, *args, **kwargs):
        district = request.POST.get("district")
        result = {
            "labels": ["강남구", "서초구", "송파구"],
            "datasets": [{"label": "예측 가격","data": [1200, 1100, 1050],"borderColor": "rgb(255, 99, 132)","tension": 0.1}]
        }
        predicted_price = result["datasets"][0]["data"][-1]
        return render(request, self.template_name, {"result": result, "predicted_price": predicted_price, "district": district})

class PriceRangeForecastPage(TemplateView):
    template_name = 'bbs/price_range_forecast.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        result = run_analysis()
        context['result'] = result
        return context

    def post(self, request, *args, **kwargs):
        price_range = request.POST.get("price_range")
        result = {"labels": ["~1억", "1억~3억", "3억~5억", "5억 이상"], "datasets": [{"label": "예측 수요","data": [500, 700, 300, 150],"borderColor": "rgb(54, 162, 235)","tension": 0.1}]}
        predicted_demand = result["datasets"][0]["data"][-1]
        return render(request, self.template_name, {"result": result, "predicted_demand": predicted_demand, "price_range": price_range})

class DepositForecastPage(TemplateView):
    template_name = 'bbs/deposit_forecast.html'

    def post(self, request, *args, **kwargs):
        deposit_type = request.POST.get("deposit_type")
        result = {"labels": ["전세", "월세"], "datasets": [{"label": "예측 가격","data": [15000, 900],"borderColor": "rgb(255, 206, 86)","tension": 0.1}]}
        predicted_price = result["datasets"][0]["data"][-1]
        return render(request, self.template_name, {"result": result, "predicted_price": predicted_price, "deposit_type": deposit_type})

#####################################
# 메인 페이지
#####################################
class MainPage(TemplateView):
    template_name = 'bbs/main_page.html'

#####################################
# 데이터 시각화 페이지 뷰
# 기존의 DataSourcePage와 data_source_view를 통합하고
# 새로운 AJAX 뷰를 추가하여 로직을 분리합니다.
#####################################
class DataVisualizePage(TemplateView):
    template_name = 'bbs/data_visualize.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # 페이지 로드 시 기본 그래프 데이터를 미리 생성하여 템플릿에 전달합니다.
        # 이렇게 하면 초기 로딩 시 AJAX 요청이 한 번 더 발생하는 것을 방지할 수 있습니다.
        plot_type = 'price_code'
        initial_data = self._get_plot_data(plot_type)
        context['initial_plot_json'] = json.dumps(initial_data, cls=plotly.utils.PlotlyJSONEncoder)
        
        return context

    def _get_plot_data(self, plot_type):
        """
        그래프 종류에 따라 Plotly 데이터와 레이아웃을 생성하는 헬퍼 함수
        """
        if plot_type == 'price_code':
            x_data = ['~1억', '1억~3억', '3억~5억', '5억 이상']
            y_data = [400, 750, 600, 250]
            title = '가격대별 전/월세 건수'
            y_axis_title = '건수'
        else:  # 'area_code'
            x_data = ['~60㎡', '60~85㎡', '85~120㎡', '120㎡ 이상']
            y_data = [500, 800, 300, 100]
            title = '전용면적별 전/월세 건수'
            y_axis_title = '건수'

        # Plotly 그래프 생성
        trace = go.Bar(x=x_data, y=y_data, marker_color='rgba(255, 255, 255, 0.5)')
        layout = go.Layout(
            title=title,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis=dict(title='구분', showgrid=False),
            yaxis=dict(title=y_axis_title, showgrid=True, gridcolor='rgba(255,255,255,0.2)'),
        )
        fig = go.Figure(data=[trace], layout=layout)
        return fig

@csrf_exempt
def get_data_visualize_data(request):
    """
    AJAX 요청을 처리하여 그래프 데이터를 JSON으로 반환합니다.
    _get_plot_data 헬퍼 함수를 재사용하여 로직을 통합합니다.
    """
    plot_type = request.GET.get('plot_type', 'price_code')
    
    # DataVisualizePage의 인스턴스를 생성하여 헬퍼 함수를 호출합니다.
    viz_page_instance = DataVisualizePage()
    fig = viz_page_instance._get_plot_data(plot_type)
    
    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return JsonResponse({'plot_json': graph_json})
