from django import template

register = template.Library()

@register.filter
def repeat(value, count):
    """문자열을 count만큼 반복"""
    return str(value) * int(count)

"""
 templates에서 호출하는 함수
 사용법 total|desnumber:num
"""
@register.filter
def desnumber(total,num):
    return total - num