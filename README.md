# CaViL: Instance-Level Routing and Gating of Low-Rank Experts for Cross-Domain Vision-Language Adaptation

## Dataset
In this paper，we propose Ex-Reddit-Amazon, which is a Multi-domain dataset for Multi-domain visual-aware tasks.

Link: https://drive.google.com/drive/folders/1iyfxgCLD1dX2AEDOQfoGaJO7LgWgvvQL?usp=sharing

After downloading the datasets, create a directory - data, put these all data files into it.

## Environment setup
```bash
git clone https://github.com/Kyro-Ma/CaViL.git
cd CaViL

pip install -r requirements.txx
```

## Data preparation
Step1 and Step2 are optional, if you want to rerun everything. You can use them. Or it's unnecessary.
```bash
cd src/scripts
Optional: python step1_build_item2meta_and_category.py --<category>
python crawl_images_per_cate.py --<category>
Optional: python step2_add_llava_descriptions.py --<category>
```

## Knowledge Distillation
```bash
cd ..
bash run_distillation_by_category.sh <category>
```

## Prompt tuning
```bash
cd ..
bash run_prompt.sh <category>
```

## Gating
Go through the codes in MoLoRA_Gating_lllm.ipynb.

## Getting results
Go through the codes in MoLoRA_Gating_lllm.ipynb.



