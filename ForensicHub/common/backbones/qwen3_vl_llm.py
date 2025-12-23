from ForensicHub.core.base_model import BaseModel
from ForensicHub.registry import register_model

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

class Qwen3VLLLM(BaseModel):
    def __init__(self, backbone='Qwen/Qwen3-VL-8B-Instruct'):
        super(Qwen3VLLLM, self).__init__()

        assert 'qwen3_vl' in backbone, "Backbone must be one of Qwen3VL variants"
        self.model,self.processor = self.init_model(model_path=backbone)
    
    def init_model(self, model_path: str, **kwagrs):
        """Initialize the model from a pretrained checkpoint.
        
        Args:
            model_path (str): Path to the pretrained model checkpoint.
        """
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(model_path)
        return model, processor

    def forward(self, image, max_new_tokens=1024, **kwargs):
        """Forward pass of the model.
        
        Args:
            image (torch.Tensor): Input image tensor.
            
        Returns:
            Dict[str, Any]: Dictionary containing model outputs.
                Must contain at least:
                    - 'pred': Model prediction
        """
        messages = self.build_messages(image)
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, **kwargs)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, outputs)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return output_text

@register_model("Qwen3-VL-8B-Instruct")
class Qwen3VL8B(Qwen3VLLLM):
    def __init__(self):
        super().__init__(backbone='Qwen/Qwen3-VL-8B-Instruct')
