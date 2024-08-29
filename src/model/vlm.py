import torch
import os
import base64
import requests
import io
import google.generativeai as genai
import numpy as np
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, BitsAndBytesConfig, LlavaNextForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer, AutoModel
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from const import cache_dir

from collections import Counter
from PIL import Image

TORCH_DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[
    0] >= 8 else torch.float16

np.float_ = np.float64
np.complex_ = np.complex128

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
            bnb_4bit_compute_dtype=TORCH_DTYPE
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
            bnb_4bit_compute_dtype=TORCH_DTYPE
        )
        if 'mistral' in self.model_name:
            self.prompt_template = '[INST] <image>\n{question} [/INST]'
        elif '34b' in self.model_name:
            self.prompt_template = '<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\n{question}<|im_end|><|im_start|>assistant\n'
        elif 'vicuna' in self.model_name:
            self.prompt_template = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n{question} ASSISTANT:"
        self.model = LlavaNextForConditionalGeneration.from_pretrained(self.model_name, cache_dir=cache_dir, device_map='auto', low_cpu_mem_usage=True, 
                                                                    torch_dtype=TORCH_DTYPE, attn_implementation = 'flash_attention_2')
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.model_name, cache_dir=cache_dir)
        self.device = next(self.model.parameters()).device
        self.do_sample = do_sample
        self.top_k = top_k
        self.top_p = top_p
        self.prompt_prefix = 'Only answer the below question. Do not provide any additional information.\n'
        print_model_info(self.model, self.model_name)
        
    def __call__(self, questions, images):
        images = [resize_image(image, 448) for image in images]
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
        self.safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }
        
    def __call__(self, questions, images):
        responses = []
        images = [resize_image(image, 448) for image in images]
        for question, image in zip(questions, images):
            try:
                response = self.model.generate_content([self.prompt_prefix + question, image], safety_settings=self.safety_settings)
                responses.append(response.text)
            except Exception as e:
                responses.append(f"Error generating response: {e}")
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
        images = [resize_image(image, 448) for image in images]
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
            try:
                responses.append(response['choices'][0]['message']['content'])
            except Exception as e:
                responses.append(f"Error generating response: {e}")
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
            bnb_4bit_compute_dtype=TORCH_DTYPE
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
        images = [resize_image(image, 448) for image in images]
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
            outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return outputs
            
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
        
class InternVL2:
    def __init__(self, model_name, do_sample, top_k, top_p, checkpoint):
        path = f'OpenGVLab/{model_name}' if checkpoint is None else checkpoint
        self.nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=TORCH_DTYPE
        )
        self.model = AutoModel.from_pretrained(
                        path,
                        cache_dir = cache_dir,
                        torch_dtype=TORCH_DTYPE,
                        load_in_4bit=True,
                        low_cpu_mem_usage=True,
                        use_flash_attn=True,
                        trust_remote_code=True,
                        device_map='auto')
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False, cache_dir=cache_dir)  
        self._gen_kwargs = {
            "do_sample": do_sample,
            "top_k": top_k,
            "top_p": top_p,
            "max_new_tokens": 2048
        }
        self.device = next(self.model.parameters()).device
        print_model_info(self.model, self.model_name)

    def _build_transform(self, input_size):
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform

    def _find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def _dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self._find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def _load_image(self, image, input_size=448, max_num=12):
        if image.mode != 'RGB':
            image = image.convert('RGB')
        transform = self._build_transform(input_size=input_size)
        images = self._dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values).to(self.device).to(TORCH_DTYPE)
        return pixel_values
    
    def __call__(self, questions, images):
        pixel_values_list = []
        num_patches_list = []
        images = [resize_image(image, 448) for image in images]
        for image in images:
            pixel_values = self._load_image(image)
            num_patches_list.append(pixel_values.size(0))
            pixel_values_list.append(pixel_values)
        pixel_values = torch.cat(pixel_values_list, dim=0)
        questions = [f'<image>\n{question}' for question in questions]
        with torch.no_grad():
            responses = self.model.batch_chat(self.tokenizer, pixel_values,
                                            num_patches_list=num_patches_list,
                                            questions=questions,
                                            generation_config=self._gen_kwargs)
        return responses

    
            
