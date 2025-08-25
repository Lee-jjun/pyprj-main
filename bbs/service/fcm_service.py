from pyfcm import FCMNotification
from django.conf import settings

class FCMService:
    # def __init__(self):
    #     self.push_service = FCMNotification(api_key=settings.FCM_SERVER_KEY)

    def send_push(self, registration_ids, title, body, data_message=None):
        """
        registration_ids: list of FCM device tokens
        title: 푸시 제목
        body: 푸시 내용
        data_message: 추가 데이터 dict
        """
        result = self.push_service.notify_multiple_devices(
            registration_ids=registration_ids,
            message_title=title,
            message_body=body,
            data_message=data_message
        )
        return result
