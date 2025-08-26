import logging
from django.shortcuts import render
from django.urls import reverse_lazy
from django.views.generic import ListView, DetailView, CreateView, UpdateView, TemplateView, FormView
from django.db import transaction
from django import forms
from django.core.paginator import Paginator
from django.http import JsonResponse

import pandas as pd
import plotly.express as px
import json
import plotly

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
        logger.debug("dispatch 호출됨")
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
# 메인 페이지 / 데이터 시각화
#####################################
class MainPage(TemplateView):
    template_name = 'bbs/main_page.html'

class DataSourcePage(TemplateView):
    template_name = 'bbs/data_source.html'

def data_source_view(request):
    return render(request, 'bbs/data_source.html')

def get_data_source_data(request):
    data = {'price':[150,200,250,300,450,500,600,750,800,950],
            'area':[50,65,70,85,90,100,110,120,135,150],
            'rent_type':['전세','월세','전세','월세','전세','월세','전세','월세','전세','월세']}
    df = pd.DataFrame(data)
    plot_type = request.GET.get('plot_type','price')

    if plot_type=='price':
        fig = px.histogram(df,x='price',color='rent_type',barmode='group',
                           title='가격대별 전/월세 분포',
                           labels={'price':'가격 (단위: 천만원)','rent_type':'유형'})
    elif plot_type=='area':
        fig = px.scatter(df,x='area',y='price',color='rent_type',
                         title='전용면적별 전/월세 가격',
                         labels={'area':'전용면적 (단위: m²)','price':'가격 (단위: 천만원)'})
    else:
        return JsonResponse({'error':'Invalid plot type'},status=400)

    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return JsonResponse({'plot_json': graph_json})
