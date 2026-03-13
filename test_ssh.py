import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration , InternVLForConditionalGeneration #,InternVLChatModel

# 1) 模型名（也可以换成本地路径）
model_id = "/home/liuzhilei/python_project/Models/Qwen3-VL-8B-Instruct"

# 2) 加载
processor = AutoProcessor.from_pretrained(model_id)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,   # 没有 bf16 可改成 torch.float16
    device_map="auto",
).eval()

# 3) 准备输入（示例：一张图 + 一个问题）
image = None#Image.open("your_image.jpg").convert("RGB")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "你是谁"},
        ],
    }
]

# 4) 按 Qwen3-VL 的 chat template 组 prompt，然后 processor 打包多模态输入
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

# 5) 生成
with torch.no_grad():
    out_ids = model.generate(**inputs, max_new_tokens=128)

print(processor.batch_decode(out_ids, skip_special_tokens=True)[0])