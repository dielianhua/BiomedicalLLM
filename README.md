# BiomedicalLLM

This repository contains the official implementation and resources for the paper **"High-quality data selection-driven instruction tuning for biomedical large language models"**, published in the *Journal of Biomedical Informatics*.

**Authors**: Jieqiong Zheng, Lu Sun, Xinyu He, Ruixia Cao

**Journal**: Journal of Biomedical Informatics

**DOI**: [10.1016/j.jbi.2026.105049](https://doi.org/10.1016/j.jbi.2026.105049)

**Free access until June 12, 2026**: [Read the full article](https://www.sciencedirect.com/science/article/pii/S1532046426000730)

## Abstract

This study presents a novel data selection framework for enhancing the training efficiency of large language models (LLMs) in biomedical natural language processing (NLP) tasks. We focus on critical tasks sourced from the biomedical dataset, encompassing named entity recognition (NER), relation extraction (RE), event extraction (EE), and text classification (TXTCLASS). These tasks encompass a diverse array of challenges in biomedical NLP and correspond to real-world clinical and research needs.

Specifically, our approach introduces the **Data Selection (DS) score**, a metric designed to quantify the extent to which instructions facilitate response generation by comparing model response losses under conditions with and without instructional context. Notably, we employed the DS method to filter high-quality data, and further fine-tuned the base model on the selected dataset; the resulting model was named **BiomedicalLLM**.

##Evaluation code

### 1.start server

##### modelpath：https://www.modelscope.cn/models/dielianhuaxin123/BiomedicalLLM

```
python -m vllm.entrypoints.openai.api_server \
   --model $path/$name \
  --max-model-len 40900 \
```

### 2. predict

```
python predict_final.py --name medinst_"$name" --dir . --model $path/$name --key YOUR_SECRET_KEY  --base_url "http://localhost:8000/v1" >> logs/log_$name
```

### 3. evaluation

```
python evaluation_final_filter.py --name medinst_"$name"  --dir . --original_data_dir ./all_history_filter_all

## Citation

```
If you find this work useful for your research, please cite:

```bibtex
@article{ZHENG2026105049,
title = {High-quality data selection-driven instruction tuning for biomedical large language models},
journal = {Journal of Biomedical Informatics},
volume = {179},
pages = {105049},
year = {2026},
issn = {1532-0464},
doi = {https://doi.org/10.1016/j.jbi.2026.105049},
url = {https://www.sciencedirect.com/science/article/pii/S1532046426000730},
author = {Jieqiong Zheng and Lu Sun and Xinyu He and Ruixia Cao},
keywords = {LLM, Biomedical instruction dataset, Data quality, Data selection, Natural language processing},
}
