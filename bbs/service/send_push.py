from django.http import JsonResponse
from .fcm_service import FCMService
"""
    views.py에서 호출함
"""
def send_test_push(request):
    # DB 또는 임시 토큰
    registration_ids = [
        "앱에서 발급받은 FCM 토큰1",
        "앱에서 발급받은 FCM 토큰2"
    ]
    
    fcm_service = FCMService()
    result = fcm_service.send_push(
        registration_ids=registration_ids,
        title="테스트 푸시",
        body="Django에서 보낸 푸시 메시지입니다.",
        data_message={"key": "value"}
    )
    
    return JsonResponse(result)