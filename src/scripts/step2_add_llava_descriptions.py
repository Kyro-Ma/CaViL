import argparse, io, json, os, re, requests
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from PIL import Image

import torch
#from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import AutoProcessor, LlavaNextForConditionalGeneration

# ---- Five custom placeholder tokens. We will overwrite their embeddings with vision features. ----
IMAGE_TOKENS = [
    "<ItemImageEmb1>", "<ItemImageEmb2>", "<ItemImageEmb3>",
    "<ItemImageEmb4>", "<ItemImageEmb5>"
]


def pick_device(name: str) -> str:
    n = (name or "auto").lower()
    if n == "auto":
        if torch.cuda.is_available(): return "cuda:1"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): return "mps"
        return "cpu"
    return n

def to_image(path_or_url: str) -> Image.Image:
    """Load RGB image from local path or URL."""
    if isinstance(path_or_url, str) and path_or_url.startswith("http"):
        r = requests.get(path_or_url, timeout=20)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    return Image.open(path_or_url).convert("RGB")

def ensure_tokens_and_resize(tokenizer, model) -> None:
    """
    Make sure our 5 IMAGE_TOKENS exist in the tokenizer.
    If new tokens are added, resize the model's token embeddings.
    Also ensure pad_token exists (default to eos).
    """
    added = []
    for tk in IMAGE_TOKENS:
        tid = tokenizer.convert_tokens_to_ids(tk)
        if tid is None or tid == tokenizer.unk_token_id:
            added.append(tk)
    if added:
        tokenizer.add_special_tokens({"additional_special_tokens": added})
        model.resize_token_embeddings(len(tokenizer))
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

def load_model(model_name: str, device: str):
    """
    Load LLaVA model + AutoProcessor
    """
    """dtype = torch.float16 if device == "cuda" else torch.float32
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=dtype, low_cpu_mem_usage=True
    )
    processor = AutoProcessor.from_pretrained(model_name)"""

    dtype = torch.float16 if device == "cuda:1" else torch.float32
    # LLaVA-Next model class for llava-hf/llava-v1.6-*
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_name,
        dtype=dtype,
        low_cpu_mem_usage=True,
        # attn_implementation="eager",  # uncomment if flash-attn isn’t available
    )
    
    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)

    if device == "cuda:1":
        model = model.cuda(1)
    # Ensure special tokens are present and embeddings are resized
    ensure_tokens_and_resize(processor.tokenizer, model)
    return model, processor

def _remove_title_from_caption(caption: str, title: str) -> str:
    """Remove exact occurrences of the title (case-insensitive) from the generated caption."""
    if not title:
        return caption
    try:
        pattern = re.escape(title.strip())
        cleaned = re.sub(pattern, "", caption, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+[.,;:]\s*", " ", cleaned)  # tidy punctuation
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned or caption  # if cleaning removed everything, return original
    except Exception:
        return caption

def build_prompt_with_image_tokens(title: str) -> str:
    return (
        "You are a helpful AI assistant. "
        "Given an Amazon product's title and its image, please provide a detailed, visually grounded "
        "description of the product that would help someone decide whether to purchase it. "
        "Focus on the product's appearance, features, and any other visually informative aspects. "
        "Do NOT repeat, paraphrase, or mention the product's title, brand, or any words from it in your answer.\n"
        f"This product's title is: {title}\n"
        + " ".join(IMAGE_TOKENS) 
    )

def prepare_five_views(image: Image.Image, size: int = 336) -> List[Image.Image]:
    """
    Generate 5 'views' for the image. For determinism, we simply replicate
    a center-resized image 5 times. If desired, this can be replaced by
    5 crops/augmentations to mimic the training setup more closely.
    """
    img = image.resize((size, size))
    return [img, img, img, img, img]

def normalize_pixel_values(pixel_values: torch.Tensor, target_batch: int = 1) -> Tuple[torch.Tensor, Optional[int]]:
    """
    Returns:
      pixel_values_normalized: (B, V, C, H, W)
      num_crops: int or None (if multi-crop was present)
    """
    num_crops = None
    if pixel_values.ndim == 4:
        # (V, C, H, W) → (1, V, C, H, W)
        pixel_values = pixel_values.unsqueeze(0)
    elif pixel_values.ndim == 5:
        # Assume (B, V, C, H, W)
        # If B != target_batch (1), try to squeeze or expand as needed
        if pixel_values.shape[0] != target_batch and target_batch == 1:
            # Most captioning flows use batch=1; squeeze if the processor created B>1 unexpectedly.
            pixel_values = pixel_values[:1]
    elif pixel_values.ndim == 6:
        # (B, V, num_crops, C, H, W) → record num_crops, keep shape for now
        num_crops = pixel_values.shape[2]
    else:
        raise ValueError(f"Unexpected pixel_values shape: {tuple(pixel_values.shape)}")
    return pixel_values, num_crops

def extract_projected_features(model, pixel_values: torch.Tensor, num_crops: Optional[int]) -> torch.Tensor:
    """
    Run the vision tower and projector to get visual embeddings corresponding to each view.

    Args:
      pixel_values:
        - If num_crops is None: (B, V, C, H, W)
        - If num_crops is not None: (B, V, num_crops, C, H, W)
      num_crops:
        - None or an integer indicating how many crops per view were present.

    Returns:
      cls_proj: (B, V, hidden) — one projected feature per 'view'.
        If multi-crop was present, average over crops per view (B, V, num_crops, hidden -> B, V, hidden).
    """
    device = next(model.parameters()).device

    if pixel_values.ndim == 6:
        B, V, Nc, C, H, W = pixel_values.shape
        pv = pixel_values.view(B * V * Nc, C, H, W).to(device, dtype=torch.float16)
        with torch.no_grad():
            vision_outputs = model.vision_tower(pv, return_dict=False)
        last_hidden = vision_outputs[0]              # (B*V*Nc, seq_len, hidden)
        cls = last_hidden[:, 0, :].view(B, V, Nc, -1)   # (B, V, Nc, hidden)
        # Average crops → (B, V, hidden)
        cls = cls.mean(dim=2)
        cls_proj = model.multi_modal_projector(cls)     # (B, V, hidden)
        return cls_proj

    elif pixel_values.ndim == 5:
        B, V, C, H, W = pixel_values.shape
        pv = pixel_values.view(B * V, C, H, W).to(device, dtype=torch.float16)
        with torch.no_grad():
            vision_outputs = model.vision_tower(pv, return_dict=False)
        last_hidden = vision_outputs[0]                  # (B*V, seq_len, hidden)
        cls = last_hidden[:, 0, :].view(B, V, -1)       # (B, V, hidden)
        cls_proj = model.multi_modal_projector(cls)      # (B, V, hidden)
        return cls_proj

    else:
        raise ValueError(f"Unexpected pixel_values ndim in extract_projected_features: {pixel_values.ndim}")

def insert_features_into_embeds(
    inputs_embeds: torch.Tensor,
    features: torch.Tensor,
    image_token_mask: torch.Tensor
) -> None:
    """
    Overwrite the embeddings at the positions of IMAGE_TOKENS with the projected
    visual features (one per view). If there are more features than tokens,
    take the first k; if fewer, use as many as available.
    Modifies inputs_embeds in-place.
    """
    B = inputs_embeds.size(0)
    assert features.size(0) == B, "Batch size mismatch between text and image features."
    V = features.size(1)

    for b in range(B):
        positions = torch.nonzero(image_token_mask[b], as_tuple=False).view(-1)
        assert len(positions) >= 1, "No IMAGE_TOKENS found in tokenized prompt."
        k = min(len(positions), V)
        for i in range(k):
            inputs_embeds[b, positions[i].item(), :] = features[b, i, :]

def caption_single_image(model, processor, device: str, image: Image.Image, title: str, max_new_tokens: int = 128) -> str:
    """
    Full robust captioning flow for one image.
      1) Build text that includes 5 placeholder tokens.
      2) Tokenize with the tokenizer only (no chat template).
      3) Prepare 5 'views' of the image and process via image_processor.
      4) Extract CLS features via vision tower, project to LM hidden space.
      5) Overwrite the 5 placeholder token embeddings with those features.
      6) Generate from the outer LLaVA model using inputs_embeds.
    """
    # (1) Build prompt
    tokenizer = processor.tokenizer
    prompt = "USER: " + build_prompt_with_image_tokens(title) + "\nASSISTANT:"

    # (2) Tokenize
    tokenized = tokenizer(
        [prompt],
        max_length=2048,
        padding="longest",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)

    # Mark positions of our image placeholder tokens
    image_token_ids = [tokenizer.convert_tokens_to_ids(tk) for tk in IMAGE_TOKENS]
    image_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for b in range(input_ids.size(0)):
        for tid in image_token_ids:
            pos = (input_ids[b] == tid).nonzero(as_tuple=False).squeeze(-1)
            image_token_mask[b, pos] = True

    # (3) 5 views (replicas for determinism)
    imgs5 = prepare_five_views(image, size=336)
    img_batch = processor.image_processor(imgs5, return_tensors="pt")["pixel_values"]
    # Normalize shapes and record multi-crop if any
    pixel_values, num_crops = normalize_pixel_values(img_batch, target_batch=input_ids.size(0))

    # (4) Extract & project vision features
    features = extract_projected_features(model, pixel_values, num_crops)   # (B, V, hidden)

    # (5) Compute language embeddings and insert visual features at token positions
    inputs_embeds = model.language_model.get_input_embeddings()(input_ids)
    insert_features_into_embeds(inputs_embeds, features, image_token_mask)

    # (6) Generate using the outer LLaVA module
    eos_id = getattr(tokenizer, "eos_token_id", None) or getattr(model.generation_config, "eos_token_id", None)
    pad_id = getattr(tokenizer, "pad_token_id", None) or eos_id
    with torch.no_grad():
        generated = model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,  # e.g., 64
            do_sample=True,                 # <— enable mild sampling
            temperature=0.7,                # small randomness
            top_p=0.9,                      # nucleus sampling
            repetition_penalty=1.15,        # penalize repeats
            no_repeat_ngram_size=3,         # avoid common templates
            num_beams=1,                    # keep simple
            eos_token_id=eos_id,
            pad_token_id=pad_id,
        )
    text = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

    text = _remove_title_from_caption(text, title)
    
    #print("!!!!!!!!!!!!!!DEBUG"+str(text))
    return re.sub(r"\s+", " ", text).strip()
    


def add_train_desc(train_in: str, train_out: str, img_dir: str,
                   model, processor, device: str, max_new_tokens: int,
                   per_asin_cap: int = 10):
    """
    Train file (JSON dict). For each ASIN, try local files {ASIN_0.jpg, ...}.
    If none exist, fall back to URLs in rec['images'].
    Writes: rec['image_descriptions_llava_cleaned'] = { "ASIN_0.jpg": "...", ... }
    """
    data = json.load(open(train_in, "r", encoding="utf-8"))
    base = Path(img_dir)
    updated = 0
    for asin, rec in data.items():
        if rec.get("image_descriptions_llava_cleaned"):
            continue
        cap_map = {}
        k = 0
        while k < per_asin_cap:
            #fp = base / f"{asin}_{k}.jpg"
            cate_name = Path(train_in).stem.replace("new_item2meta_train_", "")
            fp = base / cate_name / f"{asin}_{k}.jpg"
            if not fp.exists(): break
            try:
                title = rec.get("title") or rec.get("item_title") or rec.get("product_title") or ""
                cap_map[f"{asin}_{k}.jpg"] = caption_single_image(model, processor, device, to_image(str(fp)), title, max_new_tokens)
                print(f"[DEBUG][TRAIN] ASIN={asin} | Source=local | Path={fp} | Preview={cap_map[f'{asin}_{k}.jpg'][:100]}")
            except Exception:
                cap_map[f"{asin}_{k}.jpg"] = ""
            k += 1
        # fallback to URLs if needed
        if not cap_map:
            for idx, im in enumerate((rec.get("images") or [])[:per_asin_cap]):
                url = im.get("hi_res") or im.get("large") or im.get("thumb")
                if not url: continue
                try:
                    title = rec.get("title") or rec.get("item_title") or rec.get("product_title") or ""
                    cap_map[f"{asin}_{idx}.jpg"] = caption_single_image(model, processor, device, to_image(url), title, max_new_tokens)
                    print(f"[DEBUG][TRAIN] ASIN={asin} | Source=url | URL={url} | Preview={cap_map[f'{asin}_{k}.jpg'][:100]}")
                except Exception:
                    cap_map[f"{asin}_{idx}.jpg"] = ""
        if cap_map:
            rec["image_descriptions_llava_cleaned"] = cap_map
            updated += 1
            
        #print(f"[DEBUG] Captioning: {fp} -> {cap_map.get(f'{asin}_{k}.jpg','')[:80]}")

    json.dump(data, open(train_out, "w", encoding="utf-8"), ensure_ascii=False, sort_keys=True)
    print(f"[TRAIN] wrote {train_out} | updated {updated} items")

def add_valid_desc(valid_in: str, valid_out: str, img_dir: str,
                   model, processor, device: str, max_new_tokens: int):
    """
    Valid file (JSONL). For each line, use 'image_name' under valid_images_dir
    if present; otherwise try 'image' URL. Adds 'image_description_llava_cleaned'.
    """
    fin = open(valid_in, "r", encoding="utf-8")
    fout = open(valid_out, "w", encoding="utf-8")
    base = Path(img_dir)
    n = 0
    for line in fin:
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("image_description_llava_cleaned"):
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); n += 1; continue
        img = None
        name = rec.get("image_name") or ""
        if name:
            #fp = base / name
            cate_name = Path(valid_in).stem.replace("new_item2meta_valid_", "")
            fp = base / cate_name / name
            if fp.exists():
                try: img = to_image(str(fp))
                except Exception: img = None
        if img is None:
            url = rec.get("image", "")
            if isinstance(url, str) and url.startswith("http"):
                try: img = to_image(url)
                except Exception: img = None
        if img is not None:
            try:
                title = rec.get("title") or rec.get("item_title") or rec.get("product_title") or ""
                rec["image_description_llava_cleaned"] = caption_single_image(model, processor, device, img, title, max_new_tokens)

                if name:
                    src_type = "local"
                    src_link = str(base / cate_name / name)
                else:
                    src_type = "url"
                    src_link = rec.get("image", "")
                print(f"[DEBUG][VALID] Source={src_type} | Link={src_link} | Preview={rec['image_description_llava_cleaned'][:100]}")
                
            except Exception:
                rec["image_description_llava_cleaned"] = ""
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n += 1
    fin.close(); fout.close()
    print(f"[VALID] wrote {valid_out} | processed {n} lines")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="llava-hf/llava-v1.6-mistral-7b-hf")
    ap.add_argument("--device", default="auto", help="auto|cuda|cpu|mps")
    ap.add_argument("--max_new_tokens", type=int, default=128)

    ap.add_argument("--train_in", required=True)
    ap.add_argument("--train_out", required=True)
    ap.add_argument("--train_images_dir", required=True)

    ap.add_argument("--valid_in", required=True)
    ap.add_argument("--valid_out", required=True)
    ap.add_argument("--valid_images_dir", required=True)

    args = ap.parse_args()
    device = pick_device(args.device)
    print(f"Using device: {device}")

    model, processor = load_model(args.model_name, device)

    print("[DEBUG] Quick LLaVA caption sanity test ...")
    test_url = "https://m.media-amazon.com/images/I/61wpNmCUj3L._AC_SL1200_.jpg"
    test_title = "Modern adjustable office chair with ergonomic mesh design"  # example title
    try:
        test_img = to_image(test_url)
        test_caption = caption_single_image(
            model, processor, device, test_img, test_title, max_new_tokens=128
        )
        print(f"[DEBUG] Caption generation succeeded.\n[Title]: {test_title}\n[Image]: {test_url}\n[Caption]: {test_caption}\n")
    except Exception as e:
        print(f"[DEBUG] Caption generation failed: {e}")


    add_train_desc(args.train_in, args.train_out, args.train_images_dir,
                   model, processor, device, args.max_new_tokens)

    # add_valid_desc(args.valid_in, args.valid_out, args.valid_images_dir,
    #                model, processor, device, args.max_new_tokens)

if __name__ == "__main__":
    main()
