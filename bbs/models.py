# from bbs.dao.bbs_models import Bbs
# from django.db import models

# from django.db.models import F
from bbs.db.bbs_mysql import get_bbs_with_rownum,get_total_bbs_count
from django.views.generic import ListView
from bbs.dao.bbs_models import Bbs,BbsFile
# from django.core.paginator import Paginator
# from bbs.models import Bbs
class ChildBbs(Bbs):
    model = Bbs