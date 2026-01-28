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
import torch.nn as nn
import torch.nn.functional as F
import time
from PIL import Image
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, LlavaNextForConditionalGeneration

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ---------------------------------------------------------
# FlyLoRA Implementation
# ---------------------------------------------------------
class FlyLoRALinear(nn.Module):
    """
    FlyLoRA Linear Layer
    Implements the FlyLoRA method with implicit routing via fixed sparse random projection.
    """
    def __init__(self, in_features, out_features, r=32, k=8, sparsity_ratio=None, alpha=None, bias_lr=1e-3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r  # total rank
        self.k = k  # activated ranks
        self.alpha = alpha or (2.0 * r)  # scaling factor, default 2r as in LoRA
        self.bias_lr = bias_lr  # learning rate for bias update
        self.sparsity_ratio = sparsity_ratio or (k / r)  # sparsity ratio ρ = p/n

        # Fixed sparse random projection A ∈ R^{r×n}
        # Each row has exactly p non-zero entries sampled from N(0, 1/r^2)
        A = torch.zeros(r, in_features)
        p = max(1, int(in_features * self.sparsity_ratio))  # number of non-zero entries per row
        
        for i in range(r):
            # Randomly select p indices for non-zero entries
            indices = torch.randperm(in_features)[:p]
            # Initialize selected entries with normal distribution
            A[i, indices] = torch.randn(p) * (1.0 / r)
        
        self.register_buffer("A", A)  # frozen during training

        # Trainable up-projection B ∈ R^{m×r}
        self.B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.zeros_(self.B)

        # Expert-wise bias term for load balancing d ∈ R^r
        self.d = nn.Parameter(torch.zeros(r), requires_grad=False)

    def forward(self, x):
        """
        Forward pass of FlyLoRA:
        1. Project input through frozen sparse A: y = A x
        2. Add expert bias for routing: y' = y + d
        3. Select top-k experts based on |y'|
        4. Compute output using only activated experts in B
        """
        # Project input through frozen sparse A
        y = F.linear(x, self.A)  # (batch_size, r)
        
        # Add expert bias for routing
        y_biased = y + self.d  # (batch_size, r)
        
        # Select top-k experts based on magnitude
        _, selected_experts = torch.topk(y_biased.abs(), self.k, dim=-1)  # (batch_size, k)
        
        # Create mask for activated experts
        mask = torch.zeros_like(y_biased)  # (batch_size, r)
        mask.scatter_(-1, selected_experts, 1.0)  # set top-k positions to 1
        
        # Update assignment counts for load balancing
        if self.training:
            ci = torch.bincount(selected_experts.flatten(), minlength=self.r).float()
            delta_bias = (ci.mean() - ci).sign()
            self.d.data = self.d.data + self.bias_lr * delta_bias
            
        # Compute output using only activated experts
        activated_y = y * mask
        output = F.linear(activated_y, self.B) * (self.alpha / self.r)
        
        return output

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

        # Debug: print first few validation batches with sample info
        if batch_idx < 3:
            # Get the batch items to see what data is being used
            batch_items = batch  # batch contains the original items before collation
            sample_desc = batch_items[0]["description"][:80] if batch_items else "N/A"
            print(f"\n[VAL DEBUG] Batch {batch_idx}: loss={val_loss.item():.6f}, tokens={num_tokens}, sample_desc='{sample_desc}...'")

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
        
        # Debug: print validation summary
        epoch_num = self.current_epoch + 1
        print(f"\n[VAL SUMMARY] Epoch {epoch_num}: avg_loss={avg_val_loss:.6f}, ppl={ppl:.4f}, total_tokens={self.val_token_count}")

        # Save metrics and LoRA adapter for each epoch
        with open(os.path.join(self.args.output_dir, f"val_metrics_epoch_{epoch_num}.txt"), "w") as f:
            f.write(f"Val Loss: {avg_val_loss}\nVal PPL: {ppl}\n")
        
        # Save only FlyLoRA parameters for this epoch
        if self.args.save_every_epoch:
            lora_save_dir = os.path.join(self.args.output_dir, f"lora_adapter_epoch_{epoch_num}")
            os.makedirs(lora_save_dir, exist_ok=True)
            
            # Extract and save only trainable parameters (which are all FlyLoRA params)
            flylora_state_dict = {}
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    flylora_state_dict[name] = param.data.clone()
            
            torch.save(flylora_state_dict, os.path.join(lora_save_dir, "pytorch_model.bin"))
            print(f"[INFO] Saved FlyLoRA adapter ({len(flylora_state_dict)} params) to {lora_save_dir}")

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


def replace_linear_with_flylora(model, target_modules, r=8, alpha=32, k=None, sparsity_ratio=None):
    """
    Replace specified Linear layers in the model with FlyLoRA layers.
    
    Args:
        model: The base model
        target_modules: List of module names to replace (e.g., "vision_tower.layer1.linear")
        r: Rank of FlyLoRA
        alpha: Scaling factor
        k: Number of activated experts (default: r // 4)
        sparsity_ratio: Sparsity ratio (default: k / r)
    
    Returns:
        The model with replaced layers
    """
    if k is None:
        k = max(1, r // 4)
    if sparsity_ratio is None:
        sparsity_ratio = k / r
    
    replaced_modules = []
    for target_name in target_modules:
        parts = target_name.split('.')
        # Navigate to parent module
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        
        # Get the original linear layer
        linear_layer = getattr(parent, parts[-1])
        
        # Create FlyLoRA replacement
        flylora_layer = FlyLoRALinear(
            in_features=linear_layer.in_features,
            out_features=linear_layer.out_features,
            r=r,
            k=k,
            sparsity_ratio=sparsity_ratio,
            alpha=alpha
        )
        
        # Copy bias if exists
        if linear_layer.bias is not None:
            flylora_layer.bias = nn.Parameter(linear_layer.bias.data.clone())
        
        # Replace the layer
        setattr(parent, parts[-1], flylora_layer)
        replaced_modules.append(target_name)
        print(f"[INFO] Replaced {target_name} with FlyLoRA (r={r}, k={k}, alpha={alpha})")
    
    return model, replaced_modules



def main():
    parser = argparse.ArgumentParser(description="Distill Vision Embeddings with LoRA")
    parser.add_argument("--model_name", type=str, default="llava-hf/llava-v1.6-mistral-7b-hf", help="Pretrained model path")
    parser.add_argument("--train_data", type=str, default="../data/item2meta_train.json", help="Path to training data")
    parser.add_argument("--val_data", type=str, default="../data/item2meta_valid.jsonl", help="Path to validation data")
    parser.add_argument("--train_images_dir", type=str, default="../data/train_images", help="Directory with training images")
    parser.add_argument("--val_images_dir", type=str, default="../data/valid_images", help="Directory with validation images")
    parser.add_argument("--output_dir", type=str, default="./fly_out_distilled", help="Output directory for checkpoints")
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    # Freeze all parameters first
    for param in base_model.parameters():
        param.requires_grad = False

    # 3) Identify vision layers for FlyLoRA
    vision_linear_names = find_vision_linear_layer_names(base_model.vision_tower, prefix="vision_tower")
    projector_linear_names = find_vision_linear_layer_names(base_model.multi_modal_projector, prefix="multi_modal_projector")
    target_modules = vision_linear_names + projector_linear_names

    print(f"[INFO] Found {len(vision_linear_names)} Linear layers in vision_tower")
    print(f"[INFO] Found {len(projector_linear_names)} Linear layers in multi_modal_projector")
    print(f"[INFO] Total {len(target_modules)} Linear layers to apply FlyLoRA:")
    for ln in target_modules:
        print("  ", ln)

    # 4) Replace Linear layers with FlyLoRA layers
    lora_model, replaced_modules = replace_linear_with_flylora(base_model, target_modules, r=8, alpha=32, k=2)
    lora_model.to(device)
    
    print(f"[DEBUG] Replaced modules list (first 5):")
    for rm in replaced_modules[:5]:
        print(f"  {rm}")

    # Unfreeze only FlyLoRA parameters (B, d, and bias)
    # Note: parameter names have "model." prefix, but replaced_modules don't
    unfrozen_count = 0
    flylora_param_count = 0
    print(f"\n[DEBUG] Unfreezing FlyLoRA parameters:")
    for name, param in lora_model.named_parameters():
        # Check if this parameter belongs to a replaced FlyLoRA module
        for replaced_module in replaced_modules:
            # Add "model." prefix to match the parameter names from named_parameters()
            prefixed_module = "model." + replaced_module
            if name.startswith(prefixed_module):
                param.requires_grad = True
                flylora_param_count += 1
                if flylora_param_count <= 20:
                    print(f"  ✓ {name} -> requires_grad=True")
                break
    
    print(f"[DEBUG] Total FlyLoRA parameters kept trainable: {flylora_param_count}")
    
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

    # Debug: Check if train and val datasets are the same
    print(f"\n[DEBUG] Dataset Information:")
    print(f"  Train data file: {args.train_data}")
    print(f"  Val data file: {args.val_data}")
    print(f"  Same file? {args.train_data == args.val_data}")
    print(f"  Train dataset size: {len(train_dataset)}")
    print(f"  Val dataset size: {len(val_dataset)}")
    
    # Check first 3 samples
    print(f"\n[DEBUG] First 3 samples from train_dataset:")
    for i in range(min(3, len(train_dataset))):
        item = train_dataset.data[i]
        print(f"  Train[{i}]: title='{item['title'][:50]}...', image={item['image_path'].split('/')[-1]}")
    
    print(f"\n[DEBUG] First 3 samples from val_dataset:")
    for i in range(min(3, len(val_dataset))):
        item = val_dataset.data[i]
        print(f"  Val[{i}]: title='{item['title'][:50]}...', image={item['image_path'].split('/')[-1]}")

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
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=[0],
        callbacks=[checkpoint_callback],
        precision="16-mixed",
        gradient_clip_val=1.0,
        log_every_n_steps=10,
    )

    print("\n[INFO] Starting training for vision distillation.")
    trainer.fit(pl_model, train_loader, val_loader)

    # 8) Retrieve best checkpoint
    best_ckpt_path = checkpoint_callback.best_model_path
    print(f"[INFO] Best checkpoint path: {best_ckpt_path}")

    # 9) Load best model
    torch.serialization.add_safe_globals([argparse.Namespace])
    
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