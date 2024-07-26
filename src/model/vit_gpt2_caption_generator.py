from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

from const import cache_dir


class ImageCaptionGenerator:
    def __init__(self, device, model_name="nlpconnect/vit-gpt2-image-captioning"):
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.feature_extractor = ViTImageProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.device = device
        self.model.to(self.device)
        self.gen_kwargs = {"max_length": 20, "num_beams": 4}

    def get_caption(self, images):
        images = [self._convert_image(image) for image in images]
        pixel_values = self.feature_extractor(images=images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)
        output_ids = self.model.generate(pixel_values, **self.gen_kwargs)
        preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]
        return preds

    @staticmethod
    def _convert_image(image):
        if image.mode != "RGB":
            image = image.convert(mode="RGB")
        return image
