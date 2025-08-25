import logging
from django.shortcuts import render
from django.urls import reverse_lazy
from django.views.generic import (
    ListView, DetailView, FormView, CreateView, UpdateView, TemplateView
)
from django.db import transaction
from django import forms

from .forms import PostForm
from bbs.dao.bbs_models import Bbs, BbsFile
from bbs.db.bbs_mysql import get_bbs_with_rownum, get_total_bbs_count
from bbs.biz.real_estate_price_forecast import engine

logger = logging.getLogger(__name__)

#####################################
# 파일 저장 헬퍼 함수
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
        page = context['page_obj']
        current_page = page.number
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
# 게시판 내용 신규등록
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
# 게시판 내용 수정
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
# 게시판 상세조회
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
# Deeplearning (FormView 버전)
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
            result = engine()  # ✅ 실제 함수 확인 필요
        except Exception as e:
            logger.error(f"engine() 실행 오류: {e}")
            result = f"Error: {e}"
        logger.debug("Deeplearning 실행 완료")
        return render(self.request, self.template_name, {'form': form, 'result': result})

#####################################
# 예측 페이지 (모두 POST-safe 버전)
#####################################
class MonthlyForecastPage(TemplateView):
    template_name = 'bbs/monthly_forecast.html'

    def post(self, request, *args, **kwargs):
        year = request.POST.get("year")
        month = request.POST.get("month")

        result = {
            "labels": ["1월", "2월", "3월", "4월", "5월"],
            "datasets": [{
                "label": "예측 가격",
                "data": [1000, 1030, 1050, 1070, 1100],
                "borderColor": "rgb(75, 192, 192)",
                "tension": 0.1
            }]
        }
        predicted_price = result["datasets"][0]["data"][-1]

        return render(request, self.template_name, {
            "result": result,
            "predicted_price": predicted_price,
            "year": year,
            "month": month,
        })


class DistrictForecastPage(TemplateView):
    template_name = 'bbs/district_forecast.html'

    def post(self, request, *args, **kwargs):
        district = request.POST.get("district")

        result = {
            "labels": ["강남구", "서초구", "송파구"],
            "datasets": [{
                "label": "예측 가격",
                "data": [1200, 1100, 1050],
                "borderColor": "rgb(255, 99, 132)",
                "tension": 0.1
            }]
        }
        predicted_price = result["datasets"][0]["data"][-1]

        return render(request, self.template_name, {
            "result": result,
            "predicted_price": predicted_price,
            "district": district,
        })


from bbs.biz.price_range_forecast import resultData

class PriceRangeForecastPage(TemplateView):
    template_name = 'bbs/price_range_forecast.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        try:
            # resultData() 함수를 호출하여 분석 결과를 가져옵니다.
            analysis_result = resultData()
            # context에 결과 딕셔너리 자체를 추가합니다. .items()는 제거합니다.
            context['result'] = analysis_result
        except Exception as e:
            # 오류가 발생하면 오류 메시지를 context에 추가합니다.
            context['result'] = {'ERROR': str(e)}
        return context
    
#####################################
# 가격대별 예측 페이지 뷰 END
#####################################

    def post(self, request, *args, **kwargs):
        price_range = request.POST.get("price_range")

        result = {
            "labels": ["~1억", "1억~3억", "3억~5억", "5억 이상"],
            "datasets": [{
                "label": "예측 수요",
                "data": [500, 700, 300, 150],
                "borderColor": "rgb(54, 162, 235)",
                "tension": 0.1
            }]
        }
        predicted_demand = result["datasets"][0]["data"][-1]

        return render(request, self.template_name, {
            "result": result,
            "predicted_demand": predicted_demand,
            "price_range": price_range,
        })


class DepositForecastPage(TemplateView):
    template_name = 'bbs/deposit_forecast.html'

    def post(self, request, *args, **kwargs):
        deposit_type = request.POST.get("deposit_type")

        result = {
            "labels": ["전세", "월세"],
            "datasets": [{
                "label": "예측 가격",
                "data": [15000, 900],
                "borderColor": "rgb(255, 206, 86)",
                "tension": 0.1
            }]
        }
        predicted_price = result["datasets"][0]["data"][-1]

        return render(request, self.template_name, {
            "result": result,
            "predicted_price": predicted_price,
            "deposit_type": deposit_type,
        })

#####################################
# 메인 페이지
#####################################
class MainPage(TemplateView):
    template_name = 'bbs/main_page.html'
