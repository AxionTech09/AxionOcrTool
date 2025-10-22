import os
if "LOCAL_RANK" not in os.environ:
    os.environ["LOCAL_RANK"] = "0"

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info
from dots_ocr.utils import dict_promptmode_to_prompt

def inference(image_path, prompt, model, processor):
    # image_path = "demo/demo_image1.jpg"
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path
                },
                {"type": "text", "text": prompt}
            ]
        }
    ]


    # Preparation for inference
    text = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    # Convert all inputs to CPU and float32 for compatibility
    def convert_inputs_to_float32(tensor_dict):
        converted = {}
        for key, value in tensor_dict.items():
            if isinstance(value, torch.Tensor):
                if value.dtype.is_floating_point:
                    converted[key] = value.to(torch.float32).to("cpu")
                else:
                    converted[key] = value.to("cpu")
            else:
                converted[key] = value
        return converted
    
    inputs = convert_inputs_to_float32(inputs)

    # Inference: Generation of the output
    with torch.no_grad():  # Save memory for CPU operation
        generated_ids = model.generate(**inputs, max_new_tokens=24000)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)



if __name__ == "__main__":
    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    model_path = "./weights/DotsOCR"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        attn_implementation=None,          # Disabled for CPU compatibility
        torch_dtype=torch.float32,        # CPU-optimized precision
        device_map="cpu",                 # Force CPU usage for VPS server
        trust_remote_code=True
    )
    
    # Ensure model is in float32 and on CPU for VPS server compatibility
    model = model.float().to("cpu")
    model.eval()
    
    processor = AutoProcessor.from_pretrained(model_path,  trust_remote_code=True)

    image_path = "demo/demo_image1.jpg"
    for prompt_mode, prompt in dict_promptmode_to_prompt.items():
        print(f"prompt: {prompt}")
        inference(image_path, prompt, model, processor)
    