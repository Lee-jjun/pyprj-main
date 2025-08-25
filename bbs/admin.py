from django.contrib import admin
from bbs.dao.bbs_models import Bbs

# @admin.register(Bbs)
class BbsAdmin(admin.ModelAdmin):
    list_display = ('id', 'type', 'title', 'content', 'writer', 'created_at', 'updated_at', 'is_deleted', 'group_id', 'parent_id', 'depth')
admin.site.register(Bbs,BbsAdmin)
