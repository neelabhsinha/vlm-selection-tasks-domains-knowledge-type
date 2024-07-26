import os

import requests
from io import BytesIO


class ObjectTagsGenerator:
    def __init__(self, api_key=None, endpoint=None):
        try:
            api_key = os.getenv('AZURE_COMPUTER_VISION_API_KEY')
            endpoint = os.getenv('AZURE_COMPUTER_VISION_ENDPOINT')
        except KeyError:
            print('Azure Computer Vision API key and endpoint not found in environment variables. Please set them.'
                  ' Keys required: AZURE_COMPUTER_VISION_API_KEY, AZURE_COMPUTER_VISION_ENDPOINT')
        self.api_key = api_key
        self.endpoint = endpoint
        self.endpoint_url = endpoint + '/vision/v3.2/tag'

    def get_tags(self, image):
        headers = {
            'Ocp-Apim-Subscription-Key': self.api_key,
            'Content-Type': 'application/octet-stream'
        }
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format=image.format)
        img_byte_arr = img_byte_arr.getvalue()
        response = requests.post(self.endpoint_url, headers=headers, data=img_byte_arr)
        response.raise_for_status()
        tags = response.json()['tags']
        tags = [tag['name'] for tag in tags]
        return tags
