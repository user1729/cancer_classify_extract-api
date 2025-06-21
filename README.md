# API for Cancer Extraction and Classification using finetuned BiomedBERT

**Author:** Rahul Nair

**Last Modified:** 22-06-2025

<br><br>

## Overview

The objective here is to classify research paper abstracts (text-classification)  in to two classes (binary classification) cancer vs non-cancer. Also,  extract  specific cases of cancer from these paper abstracts. 
Finetuned model available in hugging face as "user1729/BiomedBERT-cancer-classifier-v1.0" (page uri: https://huggingface.co/user1729/BiomedBERT-cancer-bert-classifier-v1.0/tree/main)

<br><br>

## RUN API:

```
bash deploy.sh
```

<br><br>

## Hugging Face Spaces:

**Spaces:** [cancer_classify_extract-api](https://huggingface.co/spaces/user1729/cancer_classify_extract-api)
**API Acess:** [Running-Access Endpoint](https://user1729-cancer-classify-extract-api.hf.space/docs) 
**Example:**

```
/docs -> /process - Try out -> Replace "string" by - ["This study investigates novel biomarkers for early detection of lung cancer in non-smokers. Patients with breast cancer and melanoma showed improved outcomes."]
Execute
View Results
```


