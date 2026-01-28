#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LaViC Code: Visual Knowledge Self-Distillation
-------------------------------------------------------
This script handles the distillation process for reducing
image token overhead by learning [CLS] embeddings that
capture essential visual features.
"""

import argparse
import json
import math
import os

import pytorch_lightning as pl
import torch
import time
from PIL import Image
from peft import LoraConfig, get_peft_model, TaskType
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, LlavaNextForConditionalGeneration

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ---------------------------------------------------------
# 1) Special tokens (5 sub-image placeholders)
# ---------------------------------------------------------
IMAGE_TOKENS = [
    "<ItemImageEmb1>", "<ItemImageEmb2>", "<ItemImageEmb3>",
    "<ItemImageEmb4>", "<ItemImageEmb5>"
]

# ---------------------------------------------------------
# 2) Prompt Template
# ---------------------------------------------------------
PROMPT_TEMPLATE = (
    "You are a helpful assistant.\n"
    "Given an Amazon product's title and its image, please provide a detailed, visually grounded description of the product "
    "that would help someone decide whether to purchase it. "
    "Focus on the product's appearance, features, and any other visually informative aspects. "
    "Do not mention the product's title in your answer. "
    "This product's title is: {title}\n"
    f"{''.join(IMAGE_TOKENS)}\n\n"
    "Assistant:"
)


# ---------------------------------------------------------
# Dataset for Image-Text Pairs
# ---------------------------------------------------------
class ImageDescriptionDataset(Dataset):
    """
    A dataset that reads title + image paths + descriptions.
    It supports .json or .jsonl sources where each entry
    contains a product title, an image file name, and a text description.
    """
    def __init__(self, data_source, images_dir, is_training=True, default_image_size=(336, 336)):
        super().__init__()
        self.images_dir = images_dir
        self.is_training = is_training
        self.default_image = Image.new('RGB', default_image_size, (255, 255, 255))

        self.data = []
        if data_source.endswith('.json'):
            with open(data_source, 'r', encoding='utf-8') as f:
                data_json = json.load(f)
            for asin, item_data in data_json.items():
                title = item_data.get("title", "No Title")
                image_descs = item_data.get("image_descriptions_llava_cleaned", {})
                for image_name, desc in image_descs.items():
                    image_path = os.path.join(images_dir, image_name)
                    if os.path.exists(image_path):
                        self.data.append({
                            "title": title,
                            "image_path": image_path,
                            "description": desc
                        })
        elif data_source.endswith('.jsonl'):
            with open(data_source, 'r', encoding='utf-8') as f:
                for line in f:
                    entry = json.loads(line.strip())
                    title = entry.get("title", "No Title")
                    image_name = entry.get("image_name", "")
                    desc = entry.get("image_description_llava_cleaned", "")
                    image_path = os.path.join(images_dir, image_name)
                    if os.path.exists(image_path):
                        self.data.append({
                            "title": title,
                            "image_path": image_path,
                            "description": desc
                        })
        else:
            raise ValueError("Data source must be either .json or .jsonl")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item["image_path"]
        if os.path.exists(image_path):
            image = Image.open(image_path).convert("RGB")
        else:
            image = self.default_image

        return {
            "title": item["title"],
            "image": image,
            "description": item["description"]
        }


# ---------------------------------------------------------
# Data Collator
# ---------------------------------------------------------
class DataCollator:
    """
    Prepares batch inputs for the model, including:
      - tokenized prompts
      - tokenized labels
      - vision input as pixel values
      - mask of positions where [CLS] embeddings go
    """
    def __init__(self, processor, tokenizer, max_length, prompt_template):
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template = prompt_template
        self.image_token_ids = [
            self.tokenizer.convert_tokens_to_ids(tk) for tk in IMAGE_TOKENS
        ]

    def __call__(self, batch):
        prompts = []
        target_texts = []
        images = []

        # Build prompt-target pairs
        for item in batch:
            title = item["title"]
            desc = item["description"]
            prompt = self.prompt_template.format(title=title)
            prompts.append(prompt)
            target_texts.append(desc)
            images.append(item["image"])

        # Full text = prompt + target
        full_prompts = [p + t for p, t in zip(prompts, target_texts)]

        # Tokenize prompt only
        tokenized_prompts = self.tokenizer(
            prompts,
            max_length=self.max_length,
            padding='longest',
            truncation=True,
            return_tensors='pt'
        )
        # Tokenize full prompt (prompt+desc)
        tokenized_full_prompts = self.tokenizer(
            full_prompts,
            max_length=self.max_length,
            padding='longest',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = tokenized_full_prompts['input_ids']
        attention_mask = tokenized_full_prompts['attention_mask']
        labels = input_ids.clone()

        # Identify prompt lengths to mask out
        prompt_lengths = [len(x) for x in tokenized_prompts['input_ids']]
        for i in range(len(labels)):
            labels[i, :prompt_lengths[i]] = -100

        labels[labels == self.tokenizer.pad_token_id] = -100

        # Mark positions of image tokens
        image_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for b_idx in range(input_ids.size(0)):
            for tk_id in self.image_token_ids:
                positions = (input_ids[b_idx] == tk_id).nonzero(as_tuple=True)
                image_token_mask[b_idx, positions] = True

        # Process images
        images_processed = self.processor.image_processor(images, return_tensors='pt')
        images_tensor = images_processed['pixel_values']  # shape=(B,5,3,H,W)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'images': images_tensor,
            'image_token_mask': image_token_mask
        }


# ---------------------------------------------------------
# LightningModule for Vision Distillation
# ---------------------------------------------------------
class PretrainVisionModel(pl.LightningModule):
    """
    Trains LoRA modules on the vision tower + projector to distill
    sub-image embeddings into a [CLS]-style compact representation.
    """
    def __init__(self, model, processor, tokenizer, args):
        super().__init__()
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        self.args = args

        self.data_collator = DataCollator(
            processor,
            tokenizer,
            max_length=args.max_length,
            prompt_template=PROMPT_TEMPLATE
        )
        self.save_hyperparameters(ignore=['model', 'processor', 'tokenizer'])

        # Running sums for validation
        self.val_loss_sum = 0.0
        self.val_token_count = 0

        # Performance counters (optional)
        self.total_forward_time = 0.0
        self.total_forward_samples = 0

    def forward(self, input_ids, attention_mask, images, image_token_mask, labels=None):
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        image_token_mask = image_token_mask.to(device)
        if labels is not None:
            labels = labels.to(device)
    
        # Language token embeddings (LM hidden = 4096)
        inputs_embeds = self.model.language_model.get_input_embeddings()(input_ids)
    
        if images is not None:
            images = images.to(device, dtype=torch.float16)
            B, num_views, C, H, W = images.shape
    
            # (B*V, C, H, W)
            images_reshaped = images.view(B * num_views, C, H, W)
    
            # Vision tower forward
            vision_outputs = self.model.vision_tower(
                images_reshaped,
                return_dict=False
            )
            last_hidden_state = vision_outputs[0]          # (B*V, seq_len, 1024)
    
            # 1. CLS from vision tower (1024)
            cls_states = last_hidden_state[:, 0, :]        # (B*V, 1024)
            cls_states = cls_states.view(B, num_views, -1) # (B, V, 1024)
    
            # 2. Project to LM hidden space (4096) —— 关键一步
            cls_states = self.model.multi_modal_projector(cls_states)  # (B, V, 4096)
    
            # 3. Insert projected CLS into text embeddings
            for b_idx in range(B):
                positions = torch.nonzero(
                    image_token_mask[b_idx],
                    as_tuple=False
                ).squeeze(-1)
                pos_count = min(len(positions), num_views)
                for i in range(pos_count):
                    col = positions[i].item()
                    inputs_embeds[b_idx, col, :] = cls_states[b_idx, i, :]
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs



    def training_step(self, batch, batch_idx):
        inputs = self.data_collator(batch)
        # Optionally measure forward time for performance logging
        if getattr(self.args, 'log_perf', False):
            t0 = time.time()
            outputs = self(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                images=inputs['images'],
                image_token_mask=inputs['image_token_mask'],
                labels=inputs['labels']
            )
            t1 = time.time()
            duration = t1 - t0
            # accumulate time and sample counts
            self.total_forward_time += duration
            self.total_forward_samples += len(batch)
        else:
            outputs = self(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                images=inputs['images'],
                image_token_mask=inputs['image_token_mask'],
                labels=inputs['labels']
            )
        loss = outputs.loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True,
                 batch_size=len(batch))
        return loss

    def on_train_epoch_end(self):
        """Report and save average forward time per sample for the epoch if enabled."""
        if getattr(self.args, 'log_perf', False):
            epoch_num = self.current_epoch + 1
            avg_time = float('inf')
            if self.total_forward_samples > 0:
                avg_time = self.total_forward_time / float(self.total_forward_samples)
            # print and save to file under output_dir
            msg = f"Avg.forward_time_per_sample_epoch_{epoch_num}: {avg_time:.6f}"
            print(f"[PERF] {msg}")
            try:
                with open(os.path.join(self.args.output_dir, f"forward_time_epoch_{epoch_num}.txt"), 'w') as f:
                    f.write(msg + "\n")
            except Exception:
                pass
            # reset counters for next epoch
            self.total_forward_time = 0.0
            self.total_forward_samples = 0

    def validation_step(self, batch, batch_idx):
        inputs = self.data_collator(batch)
        outputs = self(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            images=inputs['images'],
            image_token_mask=inputs['image_token_mask'],
            labels=inputs['labels']
        )
        val_loss = outputs.loss
        num_tokens = (inputs['labels'] != -100).sum().item()

        self.val_loss_sum += val_loss.item() * num_tokens
        self.val_token_count += num_tokens
        return val_loss

    def on_validation_epoch_end(self):
        if self.val_token_count > 0:
            avg_val_loss = self.val_loss_sum / self.val_token_count
        else:
            avg_val_loss = float('inf')
        ppl = math.exp(avg_val_loss) if avg_val_loss < 20 else float('inf')

        self.log('val_loss', avg_val_loss, prog_bar=True)
        self.log('val_perplexity', ppl, prog_bar=True)

        # Save metrics and LoRA adapter for each epoch
        epoch_num = self.current_epoch + 1
        with open(os.path.join(self.args.output_dir, f"val_metrics_epoch_{epoch_num}.txt"), "w") as f:
            f.write(f"Val Loss: {avg_val_loss}\nVal PPL: {ppl}\n")
        
        # Save LoRA adapter for this epoch
        if self.args.save_every_epoch:
            lora_save_dir = os.path.join(self.args.output_dir, f"lora_adapter_epoch_{epoch_num}")
            self.model.save_pretrained(lora_save_dir)
            print(f"[INFO] Saved LoRA adapter to {lora_save_dir}")

        # Reset counters
        self.val_loss_sum = 0.0
        self.val_token_count = 0

    def configure_optimizers(self):
        # Only optimize parameters that require grad (LoRA on vision tower + projector)
        params_to_optimize = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=self.args.lr,
            weight_decay=self.args.weight_decay
        )
        return optimizer


def print_trainable_parameters(model):
    """
    Utility to display the total and trainable parameters.
    """
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        all_params += num_params
        if param.requires_grad:
            trainable_params += num_params

    pct = 100 * trainable_params / all_params if all_params > 0 else 0
    msg = f"Trainable params: {trainable_params} | All params: {all_params} | Trainable %: {pct:.2f}%"
    print(msg)
    return trainable_params, all_params, pct


def manual_validation(pl_module, val_loader):
    """
    Optional: A manual validation loop for checking perplexity
    outside the Lightning trainer context.
    """
    pl_module.eval()
    device = next(pl_module.model.parameters()).device
    total_loss_sum = 0.0
    total_token_count = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Manual Validation"):
            inputs = pl_module.data_collator(batch)
            outputs = pl_module(
                input_ids=inputs['input_ids'].to(device),
                attention_mask=inputs['attention_mask'].to(device),
                images=inputs['images'],
                image_token_mask=inputs['image_token_mask'].to(device),
                labels=inputs['labels'].to(device)
            )
            val_loss = outputs.loss.item()
            num_tokens = (inputs['labels'] != -100).sum().item()
            total_loss_sum += val_loss * num_tokens
            total_token_count += num_tokens

    avg_val_loss = float('inf')
    ppl = float('inf')
    if total_token_count > 0:
        avg_val_loss = total_loss_sum / total_token_count
        ppl = math.exp(avg_val_loss) if avg_val_loss < 20 else float('inf')

    print(f"[Manual Validation] Loss={avg_val_loss:.4f}, PPL={ppl:.4f}")
    pl_module.train()
    return avg_val_loss, ppl


def find_vision_linear_layer_names(vision_model, prefix="vision_tower"):
    """
    Recursively finds all nn.Linear layers within the specified
    vision model. Returns a list of layer names (with optional prefix).
    """
    import torch
    linear_names = []
    for name, module in vision_model.named_modules():
        if isinstance(module, torch.nn.Linear):
            full_name = f"{prefix}.{name}" if prefix else name
            linear_names.append(full_name)
    return linear_names


def main():
    parser = argparse.ArgumentParser(description="Distill Vision Embeddings with LoRA")
    parser.add_argument("--model_name", type=str, default="llava-hf/llava-v1.6-mistral-7b-hf", help="Pretrained model path")
    parser.add_argument("--train_data", type=str, default="../data/item2meta_train.json", help="Path to training data")
    parser.add_argument("--val_data", type=str, default="../data/item2meta_valid.jsonl", help="Path to validation data")
    parser.add_argument("--train_images_dir", type=str, default="../data/train_images", help="Directory with training images")
    parser.add_argument("--val_images_dir", type=str, default="../data/valid_images", help="Directory with validation images")
    parser.add_argument("--output_dir", type=str, default="./out_distilled", help="Output directory for checkpoints")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_perf", action="store_true", help="Measure and log avg forward time per sample during training.")
    parser.add_argument("--validate_before_training", action="store_true", help="Optionally validate before training starts.")
    parser.add_argument("--save_every_epoch", action="store_true", help="Save LoRA adapter after every epoch.")
    args = parser.parse_args()

    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # 1) Load base model
    base_model = LlavaNextForConditionalGeneration.from_pretrained(
        args.model_name,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    base_model.to(device)

    #processor = AutoProcessor.from_pretrained(args.model_name)
    processor = AutoProcessor.from_pretrained(
        args.model_name,
        use_fast=True
    )

    tokenizer = processor.tokenizer

    # 2) Add special tokens
    special_tokens_dict = {'additional_special_tokens': IMAGE_TOKENS}
    """
    tokenizer.add_special_tokens(special_tokens_dict)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = tokenizer.eos_token
    base_model.resize_token_embeddings(len(tokenizer))
    """
    tokenizer.add_special_tokens(
        {"additional_special_tokens": IMAGE_TOKENS}
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model.resize_token_embeddings(len(tokenizer))


    # 3) Identify vision layers for LoRA
    vision_linear_names = find_vision_linear_layer_names(base_model.vision_tower, prefix="vision_tower")
    projector_linear_names = find_vision_linear_layer_names(base_model.multi_modal_projector, prefix="multi_modal_projector")
    target_modules = vision_linear_names + projector_linear_names

    print("[INFO] Vision tower linear layers to apply LoRA:")
    for ln in target_modules:
        print("  ", ln)

    # 4) Construct LoRA config and wrap the base model
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=target_modules
    )
    lora_model = get_peft_model(base_model, lora_config)
    lora_model.to(device)

    # Print trainable/total parameters so logs capture this info
    t_params, a_params, pct = print_trainable_parameters(lora_model)
    # Also save to a text file for easy parsing
    try:
        params_file = os.path.join(args.output_dir, "params_summary.txt")
        with open(params_file, 'w') as pf:
            pf.write(f"trainable_params: {t_params}\n")
            pf.write(f"all_params: {a_params}\n")
            pf.write(f"trainable_percent: {pct:.4f}\n")
    except Exception:
        pass

    # 5) Lightning module
    pl_model = PretrainVisionModel(lora_model, processor, tokenizer, args).to(device)

    # 6) Datasets & DataLoaders
    train_dataset = ImageDescriptionDataset(args.train_data, args.train_images_dir, is_training=True)
    val_dataset = ImageDescriptionDataset(args.val_data, args.val_images_dir, is_training=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda x: x
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda x: x
    )

    # Optional initial validation
    if args.validate_before_training:
        manual_validation(pl_model, val_loader)

    # 7) Set up Trainer
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename='pretrain_epoch{epoch}-val_loss{val_loss:.4f}',
        save_top_k=1,
        monitor='val_loss',
        mode='min',
    )
    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        accelerator="gpu",
        devices=[1],
        callbacks=[checkpoint_callback],
        precision="16",
        gradient_clip_val=1.0,
        log_every_n_steps=10,
    )

    print("\n[INFO] Starting training for vision distillation.")
    trainer.fit(pl_model, train_loader, val_loader)

    # 8) Retrieve best checkpoint
    best_ckpt_path = checkpoint_callback.best_model_path
    print(f"[INFO] Best checkpoint path: {best_ckpt_path}")

    torch.serialization.add_safe_globals([argparse.Namespace])

    # 9) Load best model
    best_model = PretrainVisionModel.load_from_checkpoint(
        checkpoint_path=best_ckpt_path,
        model=lora_model,
        processor=processor,
        tokenizer=tokenizer,
        args=args
    ).to(device)

    # 10) Save only the LoRA adapter (best model)
    best_model.model.save_pretrained(os.path.join(args.output_dir, "lora_adapter_best"))
    print("[INFO] Best LoRA adapter saved to: " + os.path.join(args.output_dir, "lora_adapter_best"))
    
    # Print summary of saved adapters
    print("\n[INFO] Training completed. Saved LoRA adapters summary:")
    print(f"  - Best adapter: {os.path.join(args.output_dir, 'lora_adapter_best')}")
    if args.save_every_epoch:
        print(f"  - Per-epoch adapters: {os.path.join(args.output_dir, 'lora_adapter_epoch_*')}")


if __name__ == "__main__":
    main()