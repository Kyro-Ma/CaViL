#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LaViC Code: Recommendation Prompt Tuning
----------------------------------------------------------------------
This script fine-tunes a pretrained LLaVA model on a recommendation task,
applying LoRA to the language model side. We always assume images
are used in the prompt (5 sub-image approach).

use: 
python prompt_tuning.py  --category Movies_and_TV  --candidate_type candidates_st  --finetune_output_dir ./out_finetuned  --max_length 2048  --batch_size 1  --lr 5e-5  --weight_decay 1e-5  --num_epochs 1
change the num
"""

import argparse
import csv
import json
import time
import os
import random
import re

import pytorch_lightning as pl
import torch
from PIL import Image
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from pytorch_lightning import seed_everything
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, LlavaNextForConditionalGeneration

os.environ["TOKENIZERS_PARALLELISM"] = "false"


##############################################################
# LLaVA v1.6: 5 image tokens
##############################################################
IMAGE_TOKENS = [
    "<ItemImageEmb1>", "<ItemImageEmb2>", "<ItemImageEmb3>",
    "<ItemImageEmb4>", "<ItemImageEmb5>"
]


##############################################################
# Utility Functions
##############################################################
# Rough character budget per token for truncating long conversations.
CONV_CHARS_PER_TOKEN = 4
def load_item_meta(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f]

def evaluate_recall_at_k(recommended_ids, gt_items, k=1):
    hits = 0
    total = 0
    for rec_id, gts in zip(recommended_ids, gt_items):
        for gt in gts:
            if rec_id is not None and rec_id == gt:
                hits += 1
            total += 1
    recall = hits / total if total > 0 else 0.0
    return recall

def check_validity(file_path, model_key):
    """
    Checks whether the recommended item is in the candidate list.
    """
    candidates_key = f"candidates_{model_key}"
    recommended_key = f"recommended_{model_key}"

    total = 0
    valid = 0
    invalid_entries = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            recommended = data.get(recommended_key, None)
            candidates = data.get(candidates_key, [])
            total += 1
            if recommended in candidates:
                valid += 1
            else:
                invalid_entries.append({
                    "line_number": idx,
                    "recommended_id": recommended,
                    "candidates": candidates
                })
    validity = valid / total if total > 0 else 0.0
    return validity, invalid_entries, total

def calculate_image_missing_proportion(data, item_meta, image_dir, candidate_type):
    total_candidates = 0
    missing_images = 0
    for entry in data:
        candidates = entry.get(candidate_type, [])
        for cid in candidates:
            total_candidates += 1
            image_path = os.path.join(image_dir, f"{cid}_0.jpg")
            if not os.path.exists(image_path):
                missing_images += 1

    proportion = (missing_images / total_candidates * 100) if total_candidates > 0 else 0
    print(f"Total candidates: {total_candidates}")
    print(f"Missing images: {missing_images}")
    print(f"Proportion of candidates without images: {proportion:.2f}%")

def prepare_candidate_info(candidates, item_meta, image_dir, default_image):
    candidate_info = []
    for cid in candidates:
        title = item_meta.get(cid, {}).get('title', 'No Title')
        image_path = os.path.join(image_dir, f"{cid}_0.jpg")
        if os.path.exists(image_path):
            image = Image.open(image_path).convert('RGB')
        else:
            image = default_image
        candidate_info.append({
            'id': cid,
            'title': title,
            'image': image
        })
    return candidate_info

def build_prompt(conversation_text, candidates_info, max_conv_chars=None):
    """
    Always uses images in the prompt.
    """
    if max_conv_chars is not None and len(conversation_text) > max_conv_chars:
        head_len = max_conv_chars // 2
        tail_len = max_conv_chars - head_len
        conversation_text = (
            conversation_text[:head_len]
            + "\n...[TRUNCATED]...\n"
            + conversation_text[-tail_len:]
        )
    prompt = (
        "You are an AI assistant specialized in providing personalized product recommendations based on user conversations. "
        "You are given a conversation between a user seeking recommendation (denoted by <submission>) and other users providing comments (denoted by <comment>). "
        "You are also given a set of candidate products with their IDs, titles and images formatted as \"ID: title\" followed by an image. "
        "Among the candidates, recommend the most relevant product to the seeker. "
        "Only reply with its ID, and don't say anything else.\n\n"
        f"Conversation:\n{conversation_text}\n\n"
        "Candidates:\n"
    )
    for candidate in candidates_info:
        cid = candidate['id']
        title = candidate['title']
        prompt += f"{cid}: {title}\n"
        prompt += "".join(IMAGE_TOKENS) + "\n"

    prompt += "\nAssistant:"
    return prompt


##############################################################
# Dataset for Recommendation
##############################################################
class RecommendationDataset(Dataset):
    """
    Builds training or evaluation samples from conversation data,
    always using images in the prompt.
    """
    def __init__(self, data, item_meta, image_dir, candidate_type, default_image, is_training=True, max_conv_chars=None):
        self.data = data
        self.item_meta = item_meta
        self.image_dir = image_dir
        self.candidate_type = candidate_type
        self.default_image = default_image
        self.is_training = is_training
        self.max_conv_chars = max_conv_chars

        if self.is_training:
            self.index_mapping = [
                (entry_idx, gt_idx)
                for entry_idx, entry in enumerate(self.data)
                for gt_idx in range(len(entry.get('gt_items', [])))
            ]
        else:
            self.index_mapping = [(entry_idx, None) for entry_idx in range(len(self.data))]

    def __len__(self):
        return len(self.index_mapping)

    def __getitem__(self, idx):
        entry_idx, gt_idx = self.index_mapping[idx]
        entry = self.data[entry_idx]
        conversation_text = entry.get('context', '')
        gt_items = entry.get('gt_items', [])
        candidates = entry.get(self.candidate_type, [])

        if self.is_training:
            gt_item = gt_items[gt_idx]
            # Ensure exactly one ground truth in the candidate list
            if gt_item not in candidates and len(candidates) > 0:
                replace_idx = random.randint(0, len(candidates) - 1)
                candidates[replace_idx] = gt_item
            target_text = gt_item
        else:
            target_text = ""

        candidate_info = prepare_candidate_info(candidates, self.item_meta, self.image_dir, self.default_image)
        prompt = build_prompt(conversation_text, candidate_info, max_conv_chars=self.max_conv_chars)
        images = [c['image'] for c in candidate_info]

        return {
            'prompt': prompt,
            'images': images,
            'target_text': target_text,
            'gt_items': gt_items,
            'entry_idx': entry_idx
        }


##############################################################
# Data Collator for LLaVA v1.6
##############################################################
class DataCollatorForLLaVA:
    """
    Prepares batch inputs for LLaVA, with multi-subimage approach (5 per candidate).
    """
    def __init__(self, processor, tokenizer, max_length):
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_token_ids = [
            self.tokenizer.convert_tokens_to_ids(tk) for tk in IMAGE_TOKENS
        ]

    def __call__(self, batch):
        prompts = [item['prompt'] for item in batch]
        target_texts = [item['target_text'] for item in batch]
        images_per_sample = [item['images'] for item in batch]

        # Tokenize prompts and targets separately to reserve space for targets.
        tokenized_prompts = self.tokenizer(
            prompts,
            add_special_tokens=True,
            padding=False,
            truncation=False,
        )
        tokenized_targets = self.tokenizer(
            target_texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )

        input_ids_list = []
        prompt_lengths = []
        target_lengths = []
        for prompt_ids, target_ids in zip(tokenized_prompts['input_ids'], tokenized_targets['input_ids']):
            prompt_ids = list(prompt_ids)
            target_ids = list(target_ids)

            target_lengths.append(len(target_ids))
            if len(target_ids) >= self.max_length:
                target_ids = target_ids[: self.max_length - 1]

            reserve = self.max_length - len(target_ids)
            bos_id = self.tokenizer.bos_token_id
            if bos_id is not None and len(prompt_ids) > 0 and prompt_ids[0] == bos_id:
                if reserve <= 1:
                    prompt_ids = [bos_id]
                else:
                    prompt_ids = [bos_id] + prompt_ids[-(reserve - 1):]
            else:
                prompt_ids = prompt_ids[-reserve:]

            input_ids = prompt_ids + target_ids
            input_ids_list.append(input_ids)
            prompt_lengths.append(len(prompt_ids))

        max_len = max(len(ids) for ids in input_ids_list)
        pad_id = self.tokenizer.pad_token_id
        input_ids = torch.full((len(input_ids_list), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(input_ids_list), max_len), dtype=torch.long)
        labels = torch.full((len(input_ids_list), max_len), -100, dtype=torch.long)

        for i, ids in enumerate(input_ids_list):
            seq_len = len(ids)
            input_ids[i, :seq_len] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, :seq_len] = 1
            labels[i, prompt_lengths[i]:seq_len] = torch.tensor(ids[prompt_lengths[i]:], dtype=torch.long)

        # Mark image token positions
        image_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for b_idx in range(input_ids.size(0)):
            for tid in self.image_token_ids:
                positions = (input_ids[b_idx] == tid).nonzero(as_tuple=False).squeeze(-1)
                image_token_mask[b_idx, positions] = True

        # Flatten images
        all_images = []
        for imgs in images_per_sample:
            all_images.extend(imgs)

        if all_images:
            images_processed = self.processor.image_processor(all_images, return_tensors='pt')
            images_tensor = images_processed['pixel_values']
        else:
            images_tensor = None

        images_per_sample_lengths = []
        for imgs in images_per_sample:
            subcount = 5 * len(imgs)  # 5 subimages per candidate
            images_per_sample_lengths.append(subcount)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'images': images_tensor,
            'image_token_mask': image_token_mask,
            'images_per_sample_lengths': images_per_sample_lengths,
            'prompt_token_lengths': prompt_lengths,
            'target_token_lengths': target_lengths,
            'input_token_lengths': [len(ids) for ids in input_ids_list],
            'candidate_counts': [len(item.get('images', [])) for item in batch],
        }


##############################################################
# LLaVAModel => forward => always uses images
##############################################################
class LLaVAModel(pl.LightningModule):
    """
    Model for recommendation prompt tuning:
    - Applies LoRA to the language model side.
    - Always uses images in the prompt.
    """
    def __init__(self, model, processor, tokenizer, args):
        super().__init__()
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        self.args = args
        self.save_hyperparameters(ignore=['model', 'processor', 'tokenizer'])

        self.data_collator = DataCollatorForLLaVA(processor, tokenizer, max_length=args.max_length)
        self.test_results = []
        self.total_forward_time = 0.0
        self.total_forward_samples = 0
        self.debug_print_limit = 5
        self._reset_debug_counters()

    def _reset_debug_counters(self):
        self.debug_train_printed = 0
        self.debug_val_printed = 0
        self.train_zero_token_batches = 0
        self.val_zero_token_batches = 0
        self.train_nan_batches = 0
        self.val_nan_batches = 0
        self.train_label_tokens = 0
        self.val_label_tokens = 0
        self.train_batches = 0
        self.val_batches = 0

    def forward(self, input_ids, attention_mask, images, image_token_mask, images_per_sample_lengths, labels=None):
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
            B = input_ids.size(0)
            num_views = 5  # fixed to 5 sub-images
            C, H, W = images.shape[2], images.shape[3], images.shape[4]
    
            # Reshape images to (B*V, C, H, W) for vision tower
            images_reshaped = images.view(-1, C, H, W)
    
            # Vision tower forward
            vision_outputs = self.model.vision_tower(
                images_reshaped,
                return_dict=False
            )
            last_hidden_state = vision_outputs[0]          # (B*V, seq_len, 1024)
    
            # 1. CLS from vision tower (1024)
            cls_states = last_hidden_state[:, 0, :]        # (Total_Patches, 1024)
            total_patches = cls_states.shape[0]
            expected_images = B * num_views
            if total_patches > expected_images:
                patches_per_image = total_patches // expected_images
                if total_patches % expected_images == 0 and patches_per_image > 1:
                    cls_states = cls_states.view(expected_images, patches_per_image, -1).mean(dim=1)
            cls_states = cls_states.reshape(B, num_views, -1) # (B, V, 1024)
    
            # 2. Project to LM hidden space (4096) —— key step
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

    def _generate_recommendations(self, inputs):
        device = next(self.model.parameters()).device
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        image_token_mask = inputs['image_token_mask'].to(device)
        images = inputs['images']

        inputs_embeds = self.model.language_model.get_input_embeddings()(input_ids)

        if images is not None:
            images = images.to(device, dtype=torch.float16)
            B = input_ids.size(0)
            num_views = 5
            C, H, W = images.shape[2], images.shape[3], images.shape[4]

            images_reshaped = images.view(-1, C, H, W)
            vision_outputs = self.model.vision_tower(images_reshaped, return_dict=False)
            cls_states = vision_outputs[0][:, 0, :]
            total_patches = cls_states.shape[0]
            expected_images = B * num_views
            if total_patches > expected_images:
                patches_per_image = total_patches // expected_images
                if total_patches % expected_images == 0 and patches_per_image > 1:
                    cls_states = cls_states.view(expected_images, patches_per_image, -1).mean(dim=1)
            cls_states = cls_states.reshape(B, num_views, -1)

            cls_states = self.model.multi_modal_projector(cls_states)

            for b_idx in range(B):
                image_positions = torch.nonzero(image_token_mask[b_idx], as_tuple=False).squeeze(-1)
                if image_positions.numel() == 0:
                    continue

                pos_count = min(len(image_positions), num_views)
                for i in range(pos_count):
                    col = image_positions[i].item()
                    inputs_embeds[b_idx, col, :] = cls_states[b_idx, i, :]

        generated_ids = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=10,
            num_beams=1,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        recommended_ids = []
        for txt in generated_texts:
            match = re.findall(r'\bB[A-Z0-9]{9}\b', txt.strip())
            recommended_ids.append(match[0][:10] if match else None)

        return recommended_ids, generated_texts

    def training_step(self, batch, batch_idx):
        inputs = self.data_collator(batch)
        if getattr(self.args, 'log_perf', False):
            t0 = time.time()
            outputs = self(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                images=inputs['images'],
                image_token_mask=inputs['image_token_mask'],
                images_per_sample_lengths=inputs['images_per_sample_lengths'],
                labels=inputs['labels'],
            )
            t1 = time.time()
            self.total_forward_time += (t1 - t0)
            self.total_forward_samples += len(batch)
        else:
            outputs = self(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                images=inputs['images'],
                image_token_mask=inputs['image_token_mask'],
                images_per_sample_lengths=inputs['images_per_sample_lengths'],
                labels=inputs['labels'],
            )
        loss = outputs.loss
        num_tokens = (inputs['labels'] != -100).sum().item()
        self.train_label_tokens += num_tokens
        self.train_batches += 1
        if num_tokens == 0:
            self.train_zero_token_batches += 1
        if torch.isnan(loss):
            self.train_nan_batches += 1
        if (num_tokens == 0 or torch.isnan(loss)) and self.debug_train_printed < self.debug_print_limit:
            entry_idxs = [b.get('entry_idx') for b in batch]
            prompt_char_lens = [len(b.get('prompt', '')) for b in batch]
            target_char_lens = [len(b.get('target_text', '')) for b in batch]
            prompt_tok_lens = inputs.get('prompt_token_lengths', [])
            target_tok_lens = inputs.get('target_token_lengths', [])
            input_tok_lens = inputs.get('input_token_lengths', [])
            candidate_counts = inputs.get('candidate_counts', [])
            candidate_token_counts = inputs['image_token_mask'].sum(dim=1).tolist()
            print(
                f"[DEBUG][TRAIN] batch={batch_idx} loss={loss.item()} tokens={num_tokens} "
                f"entry_idx={entry_idxs} prompt_chars={prompt_char_lens} target_chars={target_char_lens} "
                f"prompt_toks={prompt_tok_lens} target_toks={target_tok_lens} input_toks={input_tok_lens} "
                f"candidates={candidate_counts} image_tokens={candidate_token_counts}"
            )
            self.debug_train_printed += 1
        elif batch_idx <= 5:
            prompt_tok_lens = inputs.get('prompt_token_lengths', [])
            target_tok_lens = inputs.get('target_token_lengths', [])
            input_tok_lens = inputs.get('input_token_lengths', [])
            candidate_counts = inputs.get('candidate_counts', [])
            candidate_token_counts = inputs['image_token_mask'].sum(dim=1).tolist()
            prompt_previews = [b.get('prompt', '') for b in batch]
            print(
                f"[DEBUG][TRAIN] batch={batch_idx} prompt_toks={prompt_tok_lens} "
                f"target_toks={target_tok_lens} input_toks={input_tok_lens} "
                f"candidates={candidate_counts} image_tokens={candidate_token_counts}"
            )
            for idx, preview in enumerate(prompt_previews):
                print(f"[DEBUG][TRAIN] batch={batch_idx} prompt_preview[{idx}]={preview}")
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(batch))
        return loss

    def on_train_epoch_start(self):
        self._reset_debug_counters()

    def on_train_epoch_end(self):
        if getattr(self.args, 'log_perf', False):
            epoch_num = self.current_epoch + 1
            avg_time = float('inf')
            if self.total_forward_samples > 0:
                avg_time = self.total_forward_time / float(self.total_forward_samples)
            msg = f"Avg.forward_time_per_sample_epoch_{epoch_num}: {avg_time:.6f}"
            print(f"[PERF] {msg}")
            try:
                out_path = os.path.join(self.args.finetune_output_dir, f"forward_time_epoch_{epoch_num}.txt")
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(msg + "\n")
            except Exception:
                pass
            self.total_forward_time = 0.0
            self.total_forward_samples = 0
        avg_tokens = 0.0
        if self.train_batches > 0:
            avg_tokens = self.train_label_tokens / float(self.train_batches)
        print(
            f"[DEBUG][TRAIN] epoch={self.current_epoch + 1} "
            f"batches={self.train_batches} avg_label_tokens={avg_tokens:.2f} "
            f"zero_token_batches={self.train_zero_token_batches} nan_batches={self.train_nan_batches}"
        )

    def validation_step(self, batch, batch_idx):
        inputs = self.data_collator(batch)
        num_tokens = (inputs['labels'] != -100).sum().item()
        self.val_label_tokens += num_tokens
        self.val_batches += 1
        if num_tokens == 0:
            self.val_zero_token_batches += 1
        val_loss = None
        if num_tokens > 0:
            outputs = self(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                images=inputs['images'],
                image_token_mask=inputs['image_token_mask'],
                images_per_sample_lengths=inputs['images_per_sample_lengths'],
                labels=inputs['labels'],
            )
            val_loss = outputs.loss
            if torch.isnan(val_loss):
                self.val_nan_batches += 1
        if (num_tokens == 0 or (val_loss is not None and torch.isnan(val_loss))) and \
                self.debug_val_printed < self.debug_print_limit:
            entry_idxs = [b.get('entry_idx') for b in batch]
            prompt_char_lens = [len(b.get('prompt', '')) for b in batch]
            target_char_lens = [len(b.get('target_text', '')) for b in batch]
            prompt_tok_lens = inputs.get('prompt_token_lengths', [])
            target_tok_lens = inputs.get('target_token_lengths', [])
            input_tok_lens = inputs.get('input_token_lengths', [])
            candidate_counts = inputs.get('candidate_counts', [])
            candidate_token_counts = inputs['image_token_mask'].sum(dim=1).tolist()
            print(
                f"[DEBUG][VAL] batch={batch_idx} loss={val_loss.item() if val_loss is not None else 'None'} "
                f"tokens={num_tokens} entry_idx={entry_idxs} "
                f"prompt_chars={prompt_char_lens} target_chars={target_char_lens} "
                f"prompt_toks={prompt_tok_lens} target_toks={target_tok_lens} input_toks={input_tok_lens} "
                f"candidates={candidate_counts} image_tokens={candidate_token_counts}"
            )
            self.debug_val_printed += 1
        if val_loss is not None:
            self.log('val_loss', val_loss, on_epoch=True, prog_bar=True, batch_size=len(batch))

        recommended_ids, _ = self._generate_recommendations(inputs)
        gt_items_list = [b['gt_items'] for b in batch]
        recall = evaluate_recall_at_k(recommended_ids, gt_items_list, k=1)
        self.log('val_recall', recall, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(batch))
        return {'val_loss': val_loss, 'val_recall': recall}

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            inputs = self.data_collator(batch)
            recommended_ids, generated_texts = self._generate_recommendations(inputs)

            # gather ground truth
            gt_items_list = [b['gt_items'] for b in batch]
            entry_idxs = [b['entry_idx'] for b in batch]
            for i in range(len(batch)):
                self.test_results.append({
                    'entry_idx': entry_idxs[i],
                    'recommended_id': recommended_ids[i],
                    'response': generated_texts[i]
                })

            recall = evaluate_recall_at_k(recommended_ids, gt_items_list, k=1)
            self.log('test_recall', recall, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(batch))
            return {'test_recall': recall}

    def on_validation_epoch_end(self):
        """Save LoRA adapter for each epoch if enabled."""
        avg_tokens = 0.0
        if self.val_batches > 0:
            avg_tokens = self.val_label_tokens / float(self.val_batches)
        print(
            f"[DEBUG][VAL] epoch={self.current_epoch + 1} "
            f"batches={self.val_batches} avg_label_tokens={avg_tokens:.2f} "
            f"zero_token_batches={self.val_zero_token_batches} nan_batches={self.val_nan_batches}"
        )
        if self.args.save_every_epoch:
            epoch_num = self.current_epoch + 1
            lora_save_dir = os.path.join(self.args.finetune_output_dir, f"lora_adapter_epoch_{epoch_num}")
            self.model.save_pretrained(lora_save_dir)
            print(f"[INFO] Saved LoRA adapter to {lora_save_dir}")

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params,
            lr=self.args.lr,
            weight_decay=self.args.weight_decay
        )
        return optimizer


##############################################################
# Re-introduce LoRA on the Language Model
##############################################################
def find_llm_linear_layer_names(llm_module, prefix="language_model"):
    """
    Recursively find all nn.Linear layers in the language model part.
    """
    import torch.nn as nn
    linear_names = []
    for name, module in llm_module.named_modules():
        if isinstance(module, nn.Linear):
            full_name = f"{prefix}.{name}" if prefix else name
            linear_names.append(full_name)
    return linear_names


def main():
    parser = argparse.ArgumentParser(description="Recommendation Prompt Tuning with pretrained LLaVA v1.6 + LM LoRA.")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Path to the vision LoRA adapter directory. "
                             "If None, will be auto-constructed as ./out_distilled/{category}/vision_lora_adapter_best")
    parser.add_argument("--base_model_name", type=str, default="llava-hf/llava-v1.6-mistral-7b-hf",
                        help="Base pretrained model name/path for loading model/processor.")
    parser.add_argument("--candidate_type", type=str, default="candidates_st",
                        help="JSON key for candidate items.")
    parser.add_argument("--finetune_output_dir", type=str, default="./new_out_finetuned",
                        help="Directory to save finetuning results.")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_perf", action="store_true",
                        help="Measure and log avg forward time per sample during training.")

    # Required dataset info
    parser.add_argument("--item_meta_path", type=str, default=None,
                        help="Path to item metadata JSON file. If None, auto-constructed from category.")
    parser.add_argument("--category", type=str, required=True,
                        help="Category name (e.g., 'Baby_Products', 'Sports_and_Outdoors')")
    parser.add_argument("--save_every_epoch", action="store_true", help="Save LoRA adapter after every epoch.")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 0) Auto-construct model_dir if not provided
    # Pattern: ./out_distilled/{category}/vision_lora_adapter_best
    if args.model_dir is None:
        args.model_dir = f"./out_distilled/{args.category}/vision_lora_adapter_best"
    
    print(f"[INFO] Using model from: {args.model_dir}")

    # 1) Load base model and processor
    base_model = LlavaNextForConditionalGeneration.from_pretrained(
        args.base_model_name,
        torch_dtype=torch.float16
    ).to(device)

    processor = AutoProcessor.from_pretrained(args.base_model_name)
    processor.tokenizer.padding_side = "right"
    tokenizer = processor.tokenizer

    # 2) Add 5 special tokens
    special_tokens_dict = {'additional_special_tokens': IMAGE_TOKENS}
    tokenizer.add_special_tokens(special_tokens_dict)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = tokenizer.eos_token
    base_model.resize_token_embeddings(len(tokenizer))

    # 3) Load vision LoRA adapter if provided, then merge into base model
    if args.model_dir is not None and os.path.isdir(args.model_dir):
        base_model = PeftModel.from_pretrained(base_model, args.model_dir)
        base_model = base_model.merge_and_unload()
        base_model.to(device)

    # 4) LoRA to Language Model (re-introduce)
    lm_linear_names = find_llm_linear_layer_names(base_model.language_model, prefix="language_model")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=lm_linear_names
    )
    model = get_peft_model(base_model, lora_config)
    model.to(device)

    # 5) Load metadata & data
    # Auto-construct item metadata path if not provided
    # Pattern: ../data/item2meta_train_{category}.json
    if args.item_meta_path is None:
        args.item_meta_path = f"../data/item2meta_train_{args.category}.with_desc.json"
    
    # Image directory pattern: ../data/train_images/{category}
    image_dir = f"../data/train_images/{args.category}"
    
    # Data paths follow the pattern: ../data/new_{category}/train.jsonl
    train_data_path = f"../data/{args.category}/train.jsonl"
    val_data_path = f"../data/{args.category}/valid.jsonl"
    test_data_path = f"../data/{args.category}/test.jsonl"
    
    print(f"[INFO] Loading data from:")
    print(f"  Train: {train_data_path}")
    print(f"  Val:   {val_data_path}")
    print(f"  Test:  {test_data_path}")
    print(f"  Item Meta: {args.item_meta_path}")
    print(f"  Image Dir: {image_dir}")
    
    item_meta = load_item_meta(args.item_meta_path)
    train_data = load_jsonl(train_data_path)
    val_data = load_jsonl(val_data_path)
    test_data = load_jsonl(test_data_path)

    print("Calculating missing images (train):")
    calculate_image_missing_proportion(train_data, item_meta, image_dir, args.candidate_type)
    print("Calculating missing images (valid):")
    calculate_image_missing_proportion(val_data, item_meta, image_dir, args.candidate_type)
    print("Calculating missing images (test):")
    calculate_image_missing_proportion(test_data, item_meta, image_dir, args.candidate_type)

    default_image = Image.new('RGB', (336, 336), color=(255, 255, 255))

    max_conv_chars = args.max_length * CONV_CHARS_PER_TOKEN
    train_dataset = RecommendationDataset(
        data=train_data,
        item_meta=item_meta,
        image_dir=image_dir,
        candidate_type=args.candidate_type,
        default_image=default_image,
        is_training=True,
        max_conv_chars=max_conv_chars
    )
    val_dataset = RecommendationDataset(
        data=val_data,
        item_meta=item_meta,
        image_dir=image_dir,
        candidate_type=args.candidate_type,
        default_image=default_image,
        is_training=False,
        max_conv_chars=max_conv_chars
    )
    test_dataset = RecommendationDataset(
        data=test_data,
        item_meta=item_meta,
        image_dir=image_dir,
        candidate_type=args.candidate_type,
        default_image=default_image,
        is_training=False,
        max_conv_chars=max_conv_chars
    )

    def collate_fn(batch):
        return batch

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    llava_model = LLaVAModel(model, processor, tokenizer, args)
    os.makedirs(args.finetune_output_dir, exist_ok=True)

    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        accelerator="gpu",
        devices=[0],
        callbacks=[],
        precision='16',
        gradient_clip_val=1.0,
        log_every_n_steps=10,
    )

    # Train
    print("[INFO] Starting training for recommendation tuning (LM LoRA).")
    trainer.fit(llava_model, train_loader, val_loader)

    # Test
    print("[INFO] Testing on test data.")
    test_results = trainer.test(llava_model, dataloaders=test_loader)
    print(f"Test Results: {test_results}")

    test_results_list = llava_model.test_results
    results_by_idx = {res['entry_idx']: res for res in test_results_list}

    recommended_ids = []
    ground_truths = []
    for idx, entry in enumerate(test_data):
        result = results_by_idx.get(idx)
        if result:
            recommended_ids.append(result['recommended_id'])
            ground_truths.append(entry.get('gt_items', []))
        else:
            recommended_ids.append(None)
            ground_truths.append(entry.get('gt_items', []))

    recall = evaluate_recall_at_k(recommended_ids, ground_truths, k=1)
    print(f"[Test] Recall@1: {recall:.4f}")

    # Prepare recommended/response fields
    model_key = "st" if args.candidate_type == 'candidates_st' else "gpt_large"
    if model_key == "st":
        recommended_field = 'recommended_st'
        response_field = 'response_st'
    else:
        recommended_field = f"recommended_{model_key}"
        response_field = f"response_{model_key}"

    # Save final test outputs
    for idx, entry in enumerate(test_data):
        res = results_by_idx.get(idx)
        if res:
            entry[recommended_field] = res['recommended_id']
            entry[response_field] = res['response']

    out_file_name = f'test_results_{args.candidate_type}.jsonl'
    output_file_path = os.path.join(args.finetune_output_dir, out_file_name)
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for entry in test_data:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')
    print(f"Test details saved to {output_file_path}")

    validity, invalid_entries, total_count = check_validity(output_file_path, model_key)
    print(f"Validity@1: {validity:.4f}, invalid entries: {len(invalid_entries)} / {total_count}")

    # Save final LoRA adapter
    trained_lora_path = os.path.join(args.finetune_output_dir, 'trained_lora_adapter')
    model.save_pretrained(trained_lora_path)
    print(f"LoRA adapter saved to {trained_lora_path}")

    # Summaries
    summary_file = os.path.join(args.finetune_output_dir, "results_summary.csv")
    csv_exists = os.path.exists(summary_file)
    with open(summary_file, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if not csv_exists:
            writer.writerow(["model_dir", "candidate_type", "lr", "weight_decay",
                             "num_epochs", "recall@1", "validity@1", "output_file"])
        writer.writerow([
            args.model_dir,
            args.candidate_type,
            args.lr,
            args.weight_decay,
            args.num_epochs,
            recall,
            validity,
            out_file_name
        ])
    print(f"Results summary updated: {summary_file}")
    print("Done finetuning (LM LoRA) and testing with images.")


if __name__ == "__main__":
    main()
