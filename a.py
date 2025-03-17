import os
import json
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
from utils import check_model_dtypes

# 数据集路径
BASE_PATH = "/home/jiangyanbo/hf_finetune/example_data"
IMAGE_DIR = os.path.join(BASE_PATH, "images")
TRAIN_JSON = os.path.join(BASE_PATH, "celeba_image_train.json")
EVAL_JSON = os.path.join(BASE_PATH, "celeba_image_eval.json") if os.path.exists(os.path.join(BASE_PATH, "celeba_image_eval.json")) else None

# 模型配置
MODEL_ID = "llava-hf/LLaVA-NeXT-Video-7B-hf"
OUTPUT_DIR = "/home/jiangyanbo/hf_finetune/llava_next_celeba_finetuned"

# 创建自定义数据集
class CelebADataset(Dataset):
    def __init__(self, json_file, processor, max_length=1024):
        self.processor = processor
        self.max_length = max_length
        
        # 加载数据
        with open(json_file, 'r') as f:
            self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(IMAGE_DIR, item["image"])
        
        # 加载并预处理图像
        image = Image.open(image_path).convert("RGB")
        image_array = np.array(image)
        clip = np.stack([image_array] * 8)
        
        # 提取对话
        human_msg = item["conversations"][0]["value"].replace("<image>", "")
        assistant_msg = item["conversations"][1]["value"]
        
        # 构建对话历史
        conversation = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": human_msg}]},
            {"role": "assistant", "content": assistant_msg}
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        # 处理文本和图像
        inputs = self.processor(
            text=prompt, 
            # videos=clip,
            images=image,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # 移除批量维度
        # for k, v in inputs.items():
        #     inputs[k] = v.squeeze()
            
        return inputs

# 主函数
def main():
    # 加载处理器和模型
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    
    # 使用 BitsAndBytes 进行量化以节省 GPU 内存
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.float16,
    #     bnb_4bit_use_double_quant=True,
    # )
    
    # model = AutoModelForCausalLM.from_pretrained(
    #     MODEL_ID,
    #     quantization_config=bnb_config,
    #     trust_remote_code=True,
    #     torch_dtype=torch.float16,
    #     device_map="auto"
    # )
    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        MODEL_ID,
        # quantization_config=bnb_config,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    # check_model_dtypes(model)
    
    # 将模型准备为 PEFT（参数高效微调）
    # model = prepare_model_for_kbit_training(model)
    
    # 配置 LoRA
    lora_config = LoraConfig(
        r=16,  # LoRA 的秩
        lora_alpha=32,  # LoRA 的缩放参数
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # 打印可训练参数比例
    
    # 创建数据集
    train_dataset = CelebADataset(TRAIN_JSON, processor)
    eval_dataset = CelebADataset(EVAL_JSON, processor) if EVAL_JSON else None
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=2e-4,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        save_steps=100,
        logging_steps=10,
        save_total_limit=3,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="tensorboard",
        # load_best_model_at_end=True if eval_dataset else False,
    )
    
    # 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=lambda data: {
            "input_ids": torch.cat([item["input_ids"] for item in data], dim=0),
            "attention_mask": torch.cat([item["attention_mask"] for item in data], dim=0),
            "labels": torch.cat([item["input_ids"] for item in data], dim=0),
            "pixel_values": torch.cat([item["pixel_values"] for item in data], dim=0) if "pixel_values" in data[0] else None,
        },
    )
    
    # 开始训练
    trainer.train()
    
    # 保存模型
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    
    print("Training completed and model saved!")

if __name__ == "__main__":
    main()