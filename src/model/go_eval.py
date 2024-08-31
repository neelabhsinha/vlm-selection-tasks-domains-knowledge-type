import base64
import io
from PIL import Image
import os
import pandas as pd
import json

from openai import OpenAI

from const import results_dir
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
        self.referenced_prompt_template = 'Question: {question}\n Reference Answers:{answer}\nCandidate Answer: {candidate}\n\nConsider Reference Answers to be multiple answers provided for the given question in context with the above image. If there are multiple answers, they are separated by semi-colon(;). In such case, consider the reference answer having majority votes as the correct answer. Based on the image, and the reference answers, is the candidate answer a correct answer for the given question? Answer only \'yes\' if the candidate answer is correct or only \'no\' if it is not.'
        self.reference_less_prompt_template = 'Question: {question}\nCandidate Answer: {candidate}\n\nBased on the image, is the candidate answer a correct answer for the given question? Answer only "yes" if the candidate answer is correct or only "no" if it is not.'
        self.max_tokens = max_tokens
        
    def _get_instance_payload(self, question, image, answer, candidate, custom_id):
        buffered = io.BytesIO()
        image.save(buffered, format='PNG')
        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
        if self.mode == 'referenced':
            prompt = self.referenced_prompt_template.format(question=question, answer=answer, candidate=candidate)
        elif self.mode == 'referenceless':
            prompt = self.reference_less_prompt_template.format(question=question, candidate=candidate)
        payload = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
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
        }
        return payload
    
    def _process_csv_to_jsonl(self, results_folder):
        output_jsonl_path = os.path.join(results_dir, results_folder, 'eval_batch.jsonl')
        df = pd.read_csv(os.path.join(results_dir, results_folder, 'predictions.csv'))
        with open(output_jsonl_path, mode='w', encoding='utf-8') as jsonl_file:
            for index, row in df.iterrows():
                custom_id = f"{row['dataset']}_{row['key']}"
                question = row['question']
                image_path = row['image_path']
                image = Image.open(image_path)
                image = resize_image(image, 448)
                answers = process_list_field(row['label'])
                candidate_answer = row['response']
                payload = self._get_instance_payload(question, image, answers, candidate_answer, custom_id)
                jsonl_file.write(json.dumps(payload) + '\n')
                if index >=3:
                    break
        return output_jsonl_path
    
    def submit_batch(self, results_folder):
        jsonl_path = self._process_csv_to_jsonl(results_folder)
        url = "https://api.openai.com/v1/chat/completions"
        batch_input_file = self.client.files.create(
            file=open(jsonl_path, "rb"),
            purpose="batch"
        )
        batch_input_file_id = batch_input_file.id
        batch_object = self.client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
            "description": f"eval_{results_folder}_{self.mode}"
            }
        )
        batch_object = batch_object.to_dict()
        with open(os.path.join(results_dir, results_folder, f'{self.mode}_eval_batch_object.json'), 'w') as f:
            json.dump(batch_object, f)
            os.remove(jsonl_path)
            
    def get_batch_status(self, results_folder):
        batch_object_path = os.path.join(results_dir, results_folder, f'{self.mode}_eval_batch_object.json')
        if not os.path.exists(batch_object_path):
            print(f'No submitted batch found for {results_folder}.')
            return None
        with open(batch_object_path, 'r') as f:
            batch_object = json.load(f)
        batch_id = batch_object['id']
        batch_status = self.client.batches.retrieve(batch_id)
        batch_status = batch_status.to_dict()
        with open(os.path.join(results_dir, results_folder, f'{self.mode}_eval_batch_object.json'), 'w') as f:
            json.dump(batch_status, f)
        print(f'Status of batch {results_folder}:', batch_status['status'])
        return batch_status
    
    def get_batch_results(self, results_folder):
        batch_object = self.get_batch_status(results_folder)
        if batch_object is not None and batch_object['status'] is not None and batch_object['status'] == 'completed':
            output_file_id = batch_object['output_file_id'] if 'output_file_id' in batch_object else None
            error_file_id = batch_object['error_file_id'] if 'error_file_id' in batch_object else None
            if output_file_id is not None:
                output = self.client.files.content(output_file_id)
                output=output.text
                with open(os.path.join(results_dir, results_folder, f'{self.mode}_eval_batch_results.jsonl'), 'w') as f:
                    f.write(output)
                    os.remove(os.path.join(results_dir, results_folder, f'{self.mode}_eval_batch_object.json'))
                    print(f'Results for batch {results_folder} saved to {results_dir}.')
            elif error_file_id is not None:
                error = self.client.files.content(error_file_id)
                error = error.text
                with open(os.path.join(results_dir, results_folder, f'{self.mode}_eval_batch_error.jsonl'), 'w') as f:
                    f.write(error)
                    os.remove(os.path.join(results_dir, results_folder, f'{self.mode}_eval_batch_object.json'))
                    print(f'Error for batch {results_folder} saved to {results_dir}.')
                
    def cancel_batch(self, results_folder):
        batch_object_path = os.path.join(results_dir, results_folder, f'{self.mode}_eval_batch_object.json')
        if not os.path.exists(batch_object_path):
            print(f'No submitted batch found for {results_folder}.')
        with open(batch_object_path, 'r') as f:
            batch_object = json.load(f)
        batch_id = batch_object['id']
        self.client.batches.cancel(batch_id)
        print(f'Batch {results_folder} cancelled.')
        os.remove(batch_object_path)