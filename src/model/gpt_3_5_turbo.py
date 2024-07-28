from openai import OpenAI
import json
from prompts import domain_prompt, knowledge_type_prompt
import os


class GPT35Turbo:
    def __init__(self, model='gpt-3.5-turbo'):
        api_key = os.getenv('OPENAI_API_KEY')
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.instruction = ('You are an AI assistant designed to output a JSON object with a key and list of strings'
                            ' as values. ')

    def get_image_classification(self, question, caption, object_tags, classification_type):
        if classification_type == 'application_domain':
            text = domain_prompt.format(question=question, caption=caption, object_tags=object_tags)
        elif classification_type == 'knowledge_type':
            text = knowledge_type_prompt.format(question=question, caption=caption, object_tags=object_tags)
        messages = [
            {'role': 'system', 'content': self.instruction},
            {'role': 'user', 'content': text}
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            messages=messages
        )
        output = response.choices[0].message.content
        output_json = json.loads(output)
        return output_json
