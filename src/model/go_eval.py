import base64
import io
from PIL import Image
import os
import numpy as np
import requests

from openai import OpenAI

from src.utils.results_io_util import process_list_field
from src.model.vlm import resize_image


class GOEval:
    def __init__(self, model_name = 'gpt-4o', mode='referenced', max_tokens=10):
        self.mode = mode
        self.model_name = model_name
        print('Initializing GOEval with model:', model_name)
        api_key = os.getenv('OPENAI_API_KEY')
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        self.client = OpenAI()
        self.referenced_prompt_template = 'Question: {question}\n Reference Answers:{reference}\nCandidate Answer: {candidate}\n\nConsider Reference Answers to be multiple answers provided for the given question in context with the above image. If there are multiple answers, they are separated by semi-colon(;). Based on the image, is the candidate answer a correct answer for the given question? Answer only \'yes\' if the candidate answer is correct or only \'no\' if it is not.'
        self.reference_less_prompt_template = 'Question: {question}\nCandidate Answer: {candidate}\n\nBased on the image, is the candidate answer a correct answer for the given question? Answer only "yes" if the candidate answer is correct or only "no" if it is not.'
        self.max_tokens = max_tokens
        
    def _get_instance_payload(self, question, image, reference, prediction):
        buffered = io.BytesIO()
        image.save(buffered, format='PNG')
        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
        if self.mode == 'referenced':
            prompt = self.referenced_prompt_template.format(question=question, reference=reference, candidate=prediction)
        elif self.mode == 'referenceless':
            prompt = self.reference_less_prompt_template.format(question=question, candidate=prediction)
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            "max_tokens": self.max_tokens
        }
        return payload
    
    def get_response(self, question, image, reference, prediction):
        payload = self._get_instance_payload(question, image, reference, prediction)
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=self._headers, json=payload)
        response = response.json()
        try:
            out = response['choices'][0]['message']['content']
            return 1 if 'yes' in out.lower() else 0
        except Exception as e:
            return np.nan
            
    def evaluate(self, questions, image_paths, references, predictions):
        responses = []
        for prediction, question, reference, image_path in zip(predictions, questions, references, image_paths):
            reference = process_list_field(reference)
            image = Image.open(image_path)
            image = resize_image(image, 224)
            result = self.get_response(question, image, reference, prediction)
            responses.append(result)
        return responses