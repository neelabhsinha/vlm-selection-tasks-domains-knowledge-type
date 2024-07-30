import torch

from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, BitsAndBytesConfig, LlavaNextForConditionalGeneration, LlavaNextProcessor

from const import cache_dir

from collections import Counter
from PIL import Image

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
                                                                       quantization_config=self.nf4_config)
        self.processor = AutoProcessor.from_pretrained(self.model_name, cache_dir=cache_dir)
        self.device = next(self.model.parameters()).device
        self.do_sample = do_sample
        self.top_k = top_k
        self.top_p = top_p
        
    def __call__(self, questions, images):
        images = [resize_image(image, self.image_size) for image in images]
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
                                                                    torch_dtype=torch.float16, attn_implementation = 'flash_attention_2')
        self.processor = AutoProcessor.from_pretrained(self.model_name, cache_dir=cache_dir)
        self.device = next(self.model.parameters()).device
        self.do_sample = do_sample
        self.top_k = top_k
        self.top_p = top_p
        
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
            prompt = self.prompt_template.format(question=question)
            # prompt = self.processor.apply_chat_template(chat_template, add_generation_prompt=True)
            prompts.append(prompt)
        return prompts
            