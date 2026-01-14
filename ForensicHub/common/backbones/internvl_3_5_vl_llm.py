from ForensicHub.core.base_model import BaseModel
from ForensicHub.registry import register_model

import torch
import torchvision.transforms as T
from PIL import Image
from transformers import AutoTokenizer, AutoModel
from torchvision.transforms.functional import InterpolationMode

class InternVL3_5_VLLLM(BaseModel):
    def __init__(self, backbone='OpenGVLab/InternVL3_5-8B'):
        super(InternVL3_5_VLLLM, self).__init__()

        assert 'InternVL3_5' in backbone, "Backbone must be one of InternVL3_5_VL variants"
        self.model,self.tokenizer = self.init_model(model_path=backbone)
        self.prompt="<image>\nIs the image real or fake?"
    
    def init_model(self, model_path: str, **kwagrs):
        """Initialize the model from a pretrained checkpoint.
        
        Args:
            model_path (str): Path to the pretrained model checkpoint.
        """
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map="auto").eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

        return model, tokenizer

    def forward(self, image, max_new_tokens=1024, **kwargs):
        """Forward pass of the model.
        
        Args:
            image (torch.Tensor): Input image tensor.
            
        Returns:
            Dict[str, Any]: Dictionary containing model outputs.
                Must contain at least:
                    - 'pred': Model prediction
        """
        pixel_values = self._load_image(image, max_num=12).to(torch.bfloat16).cuda()
        generation_config = dict(max_new_tokens=max_new_tokens, do_sample=True)
        response, history = self.model.chat(self.tokenizer, None, self.prompt, generation_config, history=None, return_history=True)
        
        return response

    def _load_image(self, image, input_size=448, max_num=12):
        image = self.to_pil_image(image)
        transform = self._build_transform(input_size=input_size)
        images = self._dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values

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
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

@register_model("InternVL3_5_VL_8B")
class InternVL3_5_VL_8B(InternVL3_5_VLLLM):
    def __init__(self):
        super().__init__(backbone='OpenGVLab/InternVL3.5-VL-8B')
