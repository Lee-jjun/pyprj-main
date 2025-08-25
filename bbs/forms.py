from django import forms
from .models import Bbs

class PostForm(forms.ModelForm):
    # file = forms.FileField(
    #     widget=forms.ClearableFileInput(attrs={'multiple': True}),
    #     required=False
    # )
    class Meta:
        model = Bbs
        fields = ['title', 'content', 'writer']  # 사용하는 필드 지정