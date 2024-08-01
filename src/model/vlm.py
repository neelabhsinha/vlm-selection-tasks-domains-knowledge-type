import torch
import os
import base64
import requests
import io

from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, BitsAndBytesConfig, LlavaNextForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer
import google.generativeai as genai

from const import cache_dir

from collections import Counter
from PIL import Image

TORCH_DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[
    0] >= 8 else torch.float16

def print_model_info(model, model_name):
    value_counts = Counter(model.hf_device_map.values())
    total_values = sum(value_counts.values())
    value_percentages = {value: (count / total_values) * 100 for value, count in value_counts.items()}
    print(f'Loaded model {model_name} on these devices:', value_percentages)
    print(f'')
    
def resize_image(image, image_size):
    if image.width > image.height:
        new_width = image_size
        new_height = int((new_width / image.width) * image.height)
    else:
        new_height = image_size
        new_width = int((new_height / image.height) * image.width)
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    return resized_image


class PaliGemma:
    def __init__(self, model_name, do_sample, top_k, top_p, checkpoint):
        self.model_name = checkpoint if checkpoint is not None else f'google/{model_name}'
        self.image_size = int(model_name.split('-')[-1])
        self.nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        if 'paligemma' in self.model_name:
            self.model = PaliGemmaForConditionalGeneration.from_pretrained(self.model_name, cache_dir=cache_dir, device_map='auto', 
                                                                       quantization_config=self.nf4_config, low_cpu_mem_usage=True, torch_dtype=TORCH_DTYPE)
        self.processor = AutoProcessor.from_pretrained(self.model_name, cache_dir=cache_dir)
        self.device = next(self.model.parameters()).device
        self.do_sample = do_sample
        self.top_k = top_k
        self.top_p = top_p
        self.prompt_prefix = 'Only answer the below question. Do not provide any additional information.\n'
        print_model_info(self.model, self.model_name)
        
    def __call__(self, questions, images):
        images = [resize_image(image, self.image_size) for image in images]
        questions = [self.prompt_prefix + question for question in questions]
        inputs = inputs = self.processor(text=questions, images=images, return_tensors="pt", padding=True)
        inputs = inputs.to(device=self.device)
        inputs = {key: value.to(dtype=torch.int32) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=512, do_sample=self.do_sample, top_k=self.top_k, top_p=self.top_p)
            input_len = inputs["input_ids"].shape[-1]
            outputs = outputs[:, input_len:]
        responses = self.processor.batch_decode(outputs, skip_special_tokens=True)
        return responses
    
class LlavaNext:
    def __init__(self, model_name, do_sample, top_k, top_p, checkpoint):
        self.model_name = checkpoint if checkpoint is not None else f'llava-hf/{model_name}'
        self.image_size = 800
        self.nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        if 'mistral' in self.model_name:
            self.prompt_template = '[INST] <image>\n{question} [/INST]'
        elif '34b' in self.model_name:
            self.prompt_template = '<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\n{question}<|im_end|><|im_start|>assistant\n'
        self.model = LlavaNextForConditionalGeneration.from_pretrained(self.model_name, cache_dir=cache_dir, device_map='auto', low_cpu_mem_usage=True, 
                                                                    torch_dtype=TORCH_DTYPE, attn_implementation = 'flash_attention_2')
        self.processor = AutoProcessor.from_pretrained(self.model_name, cache_dir=cache_dir)
        self.device = next(self.model.parameters()).device
        self.do_sample = do_sample
        self.top_k = top_k
        self.top_p = top_p
        self.prompt_prefix = 'Only answer the below question. Do not provide any additional information.\n'
        print_model_info(self.model, self.model_name)
        
    def __call__(self, questions, images):
        prompts = self._get_prompt(questions, images)
        inputs = inputs = self.processor(text=prompts, images=images, return_tensors="pt", padding=True)
        inputs = inputs.to(device=self.device)
        inputs = {key: value.to(dtype=torch.int32) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=512, do_sample=self.do_sample, top_k=self.top_k, top_p=self.top_p)
            input_len = inputs["input_ids"].shape[-1]
            outputs = outputs[:, input_len:]
        responses = self.processor.batch_decode(outputs, skip_special_tokens=True)
        return responses
        
    def _get_prompt(self, questions, images):
        prompts = []
        for question, image in zip(questions, images):
            question = self.prompt_prefix + question
            prompt = self.prompt_template.format(question=question)
            # prompt = self.processor.apply_chat_template(chat_template, add_generation_prompt=True)
            prompts.append(prompt)
        return prompts
    
class Gemini:
    def __init__(self, model_name='gemini-1.5-flash'):
        self.model_name = model_name
        api_key = os.getenv('GOOGLE_API_KEY')
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name=model_name)
        self.prompt_prefix = 'Only answer the below question. Do not provide any additional information.\n'
        
    def __call__(self, questions, images):
        responses = []
        for question, image in zip(questions, images):
            response = self.model.generate_content([self.prompt_prefix + question, image])
            responses.append(response.text)
        return responses
    
class GPT4o:
    def __init__(self, model_name='gpt-4o-mini'):
        self.model_name = model_name
        api_key = os.getenv('OPENAI_API_KEY')
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        self.prompt_prefix = 'Only answer the below question. Do not provide any additional information.\n'

    def _encode_image(self, image):
        buffered = io.BytesIO()
        image.save(buffered, format='PNG')
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def __call__(self, questions, images):
        responses = []
        for question, image in zip(questions, images):
            try:
                base64_image = self._encode_image(image)
            except Exception as e:
                print(f"Error encoding image: {e}")
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": self.prompt_prefix + question
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 300
            }

            response = requests.post("https://api.openai.com/v1/chat/completions", headers=self._headers, json=payload)
            response = response.json()
            responses.append(response['choices'][0]['message']['content'])
        return responses

class CogVLM2:
    def __init__(self, model_name, do_sample, top_k, top_p, checkpoint):
        self.model_name = model_name
        self.model_name = checkpoint if checkpoint is not None else f'THUDM/{model_name}'
        self.image_size = 800
        self.nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map='auto', cache_dir=cache_dir, low_cpu_mem_usage=True, trust_remote_code=True,
                                                                    torch_dtype=TORCH_DTYPE)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=cache_dir, trust_remote_code=True)
        self.device = next(self.model.parameters()).device
        self.prompt_prefix = 'Only answer below the question. Do not provide any additional information.\n'
        self.gen_kwargs = {
            "max_new_tokens": 2048,
            "pad_token_id": 128002,  
        }
        print_model_info(self.model, self.model_name)
        
    def __call__(self, questions, images):
        input_batch = []
        for question, image in zip(questions, images):
            input_sample = self.model.build_conversation_input_ids(self.tokenizer, query=question, history=[], images=[image], template_version='chat')
            input_batch.append(input_sample)
        input_batch = self._collate(input_batch, self.tokenizer)
        input_batch = self._recur_move_to(input_batch, self.device, lambda x: isinstance(x, torch.Tensor))
        input_batch = self._recur_move_to(input_batch, torch.bfloat16, lambda x: isinstance(x, torch.Tensor) and torch.is_floating_point(x))
        with torch.no_grad():
            outputs = self.model.generate(**input_batch, **self.gen_kwargs)
            outputs = outputs[:, input_batch['input_ids'].shape[1]:]
            outputs = self.tokenizer.batch_decode(outputs)
            
    def _collate(self, features, tokenizer) -> dict:
        images = [feature.pop('images', None) for feature in features if 'images' in feature]
        tokenizer.pad_token = tokenizer.eos_token
        max_length = max(len(feature['input_ids']) for feature in features)
        
        def pad_to_max_length(feature, max_length):
            padding_length = max_length - len(feature['input_ids'])
            feature['input_ids'] = torch.cat([feature['input_ids'], torch.full((padding_length,), tokenizer.pad_token_id)])
            feature['token_type_ids'] = torch.cat([feature['token_type_ids'], torch.zeros(padding_length, dtype=torch.long)])
            feature['attention_mask'] = torch.cat([feature['attention_mask'], torch.zeros(padding_length, dtype=torch.long)])
            if feature['labels'] is not None:
                feature['labels'] = torch.cat([feature['labels'], torch.full((padding_length,), tokenizer.pad_token_id)])
            else:
                feature['labels'] = torch.full((max_length,), tokenizer.pad_token_id)
            return feature

        features = [pad_to_max_length(feature, max_length) for feature in features]
        batch = {
            key: torch.stack([feature[key] for feature in features])
            for key in features[0].keys()
        }

        if images:
            batch['images'] = images

        return batch
         
    def _recur_move_to(self, item, tgt, criterion_func):
        if criterion_func(item):
            device_copy = item.to(tgt)
            return device_copy
        elif isinstance(item, list):
            return [self._recur_move_to(v, tgt, criterion_func) for v in item]
        elif isinstance(item, tuple):
            return tuple([self._recur_move_to(v, tgt, criterion_func) for v in item])
        elif isinstance(item, dict):
            return {k: self._recur_move_to(v, tgt, criterion_func) for k, v in item.items()}
        else:
            return item
        
            
            