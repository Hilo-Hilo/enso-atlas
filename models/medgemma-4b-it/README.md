---
license: other
license_name: health-ai-developer-foundations
license_link: https://developers.google.com/health-ai-developer-foundations/terms
library_name: transformers
pipeline_tag: image-text-to-text
extra_gated_heading: Access MedGemma on Hugging Face
extra_gated_prompt: >-
  To access MedGemma on Hugging Face, you're required to review and
  agree to [Health AI Developer Foundation's terms of use](https://developers.google.com/health-ai-developer-foundations/terms).
  To do this, please ensure you're logged in to Hugging Face and click below.
  Requests are processed immediately.
extra_gated_button_content: Acknowledge license
base_model: google/medgemma-4b-pt
tags:
- medical
- radiology
- clinical-reasoning
- dermatology
- pathology
- ophthalmology
- chest-x-ray
---

# MedGemma model card

**Model documentation:** [MedGemma](https://developers.google.com/health-ai-developer-foundations/medgemma)

**Resources:**

*   Model on Google Cloud Model Garden: [MedGemma](https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/medgemma)
*   Model on Hugging Face: [MedGemma](https://huggingface.co/collections/google/medgemma-release-680aade845f90bec6a3f60c4)
*   GitHub repository (supporting code, Colab notebooks, discussions, and
    issues): [MedGemma](https://github.com/google-health/medgemma)
*   Quick start notebook: [GitHub](https://github.com/google-health/medgemma/blob/main/notebooks/quick_start_with_hugging_face.ipynb)
*   Fine-tuning notebook: [GitHub](https://github.com/google-health/medgemma/blob/main/notebooks/fine_tune_with_hugging_face.ipynb)
*   Concept applications built using MedGemma: [Collection](https://huggingface.co/collections/google/medgemma-concept-apps-686ea036adb6d51416b0928a)
*   Support: See [Contact](https://developers.google.com/health-ai-developer-foundations/medgemma/get-started.md#contact)
*   License: The use of MedGemma is governed by the [Health AI Developer
    Foundations terms of
    use](https://developers.google.com/health-ai-developer-foundations/terms).

**Author:** Google

## Model information

This section describes the MedGemma model and how to use it.

### Description

MedGemma is a collection of [Gemma 3](https://ai.google.dev/gemma/docs/core)
variants that are trained for performance on medical text and image
comprehension. Developers can use MedGemma to accelerate building
healthcare-based AI applications. MedGemma currently comes in three variants: a
4B multimodal version and 27B text-only and multimodal versions.

Both MedGemma multimodal versions utilize a
[SigLIP](https://arxiv.org/abs/2303.15343) image encoder that has been
specifically pre-trained on a variety of de-identified medical data, including
chest X-rays, dermatology images, ophthalmology images, and histopathology
slides. Their LLM components are trained on a diverse set of medical data,
including medical text, medical question-answer pairs, FHIR-based electronic
health record data (27B multimodal only), radiology images, histopathology
patches, ophthalmology images, and dermatology images.

MedGemma 4B is available in both pre-trained (suffix: `-pt`) and
instruction-tuned (suffix `-it`) versions. The instruction-tuned version is a
better starting point for most applications. The pre-trained version is
available for those who want to experiment more deeply with the models.

MedGemma 27B multimodal has pre-training on medical image, medical record and
medical record comprehension tasks. MedGemma 27B text-only has been trained
exclusively on medical text. Both models have been optimized for inference-time
computation on medical reasoning. This means it has slightly higher performance
on some text benchmarks than MedGemma 27B multimodal. Users who want to work
with a single model for both medical text, medical record and medical image
tasks are better suited for MedGemma 27B multimodal. Those that only need text
use-cases may be better served with the text-only variant. Both MedGemma 27B
variants are only available in instruction-tuned versions.

MedGemma variants have been evaluated on a range of clinically relevant
benchmarks to illustrate their baseline performance. These evaluations are based
on both open benchmark datasets and curated datasets. Developers can fine-tune
MedGemma variants for improved performance. Consult the [Intended
Use](https://developers.google.com/health-ai-developer-foundations/medgemma/model-card#intended_use)
section below for more details.

MedGemma is optimized for medical applications that involve a text generation
component. For medical image-based applications that do not involve text
generation, such as data-efficient classification, zero-shot classification, or
content-based or semantic image retrieval, the [MedSigLIP image
encoder](https://developers.google.com/health-ai-developer-foundations/medsiglip/model-card)
is recommended. MedSigLIP is based on the same image encoder that powers
MedGemma.

Please consult the [MedGemma Technical Report](https://arxiv.org/abs/2507.05201)
for more details.

### How to use

Below are some example code snippets to help you quickly get started running the
model locally on GPU. If you want to use the model at scale, we recommend that
you create a production version using [Model
Garden](https://cloud.google.com/model-garden).

First, install the Transformers library. Gemma 3 is supported starting from
transformers 4.50.0.

```sh
$ pip install -U transformers
```

**Run model with the `pipeline` API**

```python
from transformers import pipeline
from PIL import Image
import requests
import torch

pipe = pipeline(
    "image-text-to-text",
    model="google/medgemma-4b-it",
    torch_dtype=torch.bfloat16,
    device="cuda",
)

# Image attribution: Stillwaterising, CC0, via Wikimedia Commons
image_url = "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png"
image = Image.open(requests.get(image_url, headers={"User-Agent": "example"}, stream=True).raw)

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are an expert radiologist."}]
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this X-ray"},
            {"type": "image", "image": image}
        ]
    }
]

output = pipe(text=messages, max_new_tokens=200)
print(output[0]["generated_text"][-1]["content"])
```

**Run the model directly**

```python
# pip install accelerate
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import requests
import torch

model_id = "google/medgemma-4b-it"

model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

# Image attribution: Stillwaterising, CC0, via Wikimedia Commons
image_url = "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png"
image = Image.open(requests.get(image_url, headers={"User-Agent": "example"}, stream=True).raw)

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are an expert radiologist."}]
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this X-ray"},
            {"type": "image", "image": image}
        ]
    }
]

inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt"
).to(model.device, dtype=torch.bfloat16)

input_len = inputs["input_ids"].shape[-1]

with torch.inference_mode():
    generation = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    generation = generation[0][input_len:]

decoded = processor.decode(generation, skip_special_tokens=True)
print(decoded)
```

### Examples

See the following Colab notebooks for examples of how to use MedGemma:

*   To give the model a quick try, running it locally with weights from Hugging
    Face, see [Quick start notebook in
    Colab](https://colab.research.google.com/github/google-health/medgemma/blob/main/notebooks/quick_start_with_hugging_face.ipynb).
    Note that you will need to use Colab Enterprise to obtain adequate GPU
    resources to run either 27B model without quantization.

*   For an example of fine-tuning the 4B model, see the [Fine-tuning notebook in
    Colab](https://colab.research.google.com/github/google-health/medgemma/blob/main/notebooks/fine_tune_with_hugging_face.ipynb).
    The 27B models can be fine tuned in a similar manner but will require more
    time and compute resources than the 4B model.

### Model architecture overview

The MedGemma model is built based on [Gemma 3](https://ai.google.dev/gemma/) and
uses the same decoder-only transformer architecture as Gemma 3\. To read more
about the architecture, consult the Gemma 3 [model
card](https://ai.google.dev/gemma/docs/core/model_card_3).

### Technical specifications

*   **Model type**: Decoder-only Transformer architecture, see the [Gemma 3
    Technical
    Report](https://storage.googleapis.com/deepmind-media/gemma/Gemma3Report.pdf)
*   **Input Modalities**: Text, vision
*   **Output Modality:** Text only
*   **Attention mechanism**: Grouped-query attention (GQA)
*   **Context length**: Supports long context, at least 128K tokens
*   **Key publication**: https://arxiv.org/abs/2507.05201
*   **Model created**: July 9, 2025

*   **Model version**: 1.0.1

### Citation

When using this model, please cite: Sellergren et al. "MedGemma Technical
Report." *arXiv preprint arXiv:2507.05201* (2025).

```none
@article{sellergren2025medgemma,
  title={MedGemma Technical Report},
  author={Sellergren, Andrew and Kazemzadeh, Sahar and Jaroensri, Tiam and Kiraly, Atilla and Traverse, Madeleine and Kohlberger, Timo and Xu, Shawn and Jamil, Fayaz and Hughes, Cían and Lau, Charles and others},
  journal={arXiv preprint arXiv:2507.05201},
  year={2025}
}
```

### Inputs and outputs

**Input**:

*   Text string, such as a question or prompt
*   Images, normalized to 896 x 896 resolution and encoded to 256 tokens each
*   Total input length of 128K tokens

**Output**:

*   Generated text in response to the input, such as an answer to a question,
    analysis of image content, or a summary of a document
*   Total output length of 8192 tokens

### Performance and validation

MedGemma was evaluated across a range of different multimodal classification,
report generation, visual question answering, and text-based tasks.

### Key performance metrics

#### Imaging evaluations

The multimodal performance of MedGemma 4B and 27B multimodal was evaluated
across a range of benchmarks, focusing on radiology, dermatology,
histopathology, ophthalmology, and multimodal clinical reasoning.

MedGemma 4B outperforms the base Gemma 3 4B model across all tested multimodal
health benchmarks.

| Task and metric | Gemma 3 4B | MedGemma 4B |
| :---- | :---- | :---- |
| **Medical image classification** |  |  |
| MIMIC CXR\*\* \- macro F1 for top 5 conditions | 81.2 | 88.9 |
| CheXpert CXR \- macro F1 for top 5 conditions | 32.6 | 48.1 |
| CXR14 \- macro F1 for 3 conditions | 32.0 | 50.1 |
| PathMCQA\* (histopathology, internal\*\*)  \- Accuracy | 37.1 | 69.8 |
| US-DermMCQA\* \- Accuracy | 52.5 | 71.8 |
| EyePACS\* (fundus, internal) \- Accuracy | 14.4 | 64.9 |
| **Visual question answering** |  |  |
| SLAKE (radiology) \- Tokenized F1 | 40.2 | 72.3 |
| VQA-RAD\*\*\* (radiology) \- Tokenized F1 | 33.6 | 49.9 |
| **Knowledge and reasoning** |  |  |  |  |
| MedXpertQA (text \+ multimodal questions) \- Accuracy | 16.4 | 18.8 |

*Internal datasets. US-DermMCQA is described in [Liu (2020, Nature
medicine)](https://www.nature.com/articles/s41591-020-0842-3), presented as a
4-way MCQ per example for skin condition classification. PathMCQA is based on
multiple datasets, presented as 3-9 way MCQ per example for identification,
grading, and subtype for breast, cervical, and prostate cancer. EyePACS is a
dataset of fundus images with classification labels based on 5-level diabetic
retinopathy severity (None, Mild, Moderate, Severe, Proliferative). More details
in the [MedGemma Technical Report](https://arxiv.org/abs/2507.05201).

**Based on radiologist adjudicated labels, described in [Yang (2024,
arXiv)](https://arxiv.org/pdf/2405.03162) Section A.1.1.

***Based on "balanced split," described in [Yang (2024,
arXiv)](https://arxiv.org/pdf/2405.03162).

#### Chest X-ray report generation

MedGemma chest X-ray (CXR) report generation performance was evaluated on
[MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.1.0/) using the [RadGraph
F1 metric](https://arxiv.org/abs/2106.14463). We compare the MedGemma
pre-trained checkpoint with our previous best model for CXR report generation,
[PaliGemma 2](https://arxiv.org/abs/2412.03555).

| Metric | MedGemma 4B (pre-trained) | MedGemma 4B (tuned for CXR)| PaliGemma 2 3B (tuned for CXR) | PaliGemma 2 10B (tuned for CXR) |
| :---- | :---- | :---- | :---- | :---- |
| MIMIC CXR \- RadGraph F1 | 29.5 | 30.3 |28.8 | 29.5 |



The instruction-tuned versions of MedGemma 4B and MedGemma 27B achieve lower
scores (21.9 and 21.3, respectively) due to the differences in reporting style
compared to the MIMIC ground truth reports. Further fine-tuning on MIMIC reports
enables users to achieve improved performance, as shown by the improved
performance of the MedGemma 4B model that was tuned for CXR.

#### Text evaluations

MedGemma 4B and text-only MedGemma 27B were evaluated across a range of
text-only benchmarks for medical knowledge and reasoning.

The MedGemma models outperform their respective base Gemma models across all
tested text-only health benchmarks.

| Metric | Gemma 3 4B | MedGemma 4B |
| :---- | :---- | :---- |
| MedQA (4-op) | 50.7 | 64.4 |
| MedMCQA | 45.4 | 55.7 |
| PubMedQA | 68.4 | 73.4 |
| MMLU Med | 67.2 | 70.0 |
| MedXpertQA (text only) | 11.6 | 14.2 |
| AfriMed-QA (25 question test set) | 48.0 | 52.0 |

For all MedGemma 27B results, [test-time
scaling](https://arxiv.org/abs/2501.19393) is used to improve performance.

#### Medical record evaluations

All models were evaluated on a question answer dataset from synthetic FHIR data
to answer questions about patient records. MedGemma 27B multimodal's
FHIR-specific training gives it significant improvement over other MedGemma and
Gemma models.

| Metric | Gemma 3 4B | MedGemma 4B |
| :---- | :---- | :---- |
| EHRQA | 70.9 | 67.6 |


### Ethics and safety evaluation

#### Evaluation approach

Our evaluation methods include structured evaluations and internal red-teaming
testing of relevant content policies. Red-teaming was conducted by a number of
different teams, each with different goals and human evaluation metrics. These
models were evaluated against a number of different categories relevant to
ethics and safety, including:

*   **Child safety**: Evaluation of text-to-text and image-to-text prompts
    covering child safety policies, including child sexual abuse and
    exploitation.
*   **Content safety:** Evaluation of text-to-text and image-to-text prompts
    covering safety policies, including harassment, violence and gore, and hate
    speech.
*   **Representational harms**: Evaluation of text-to-text and image-to-text
    prompts covering safety policies, including bias, stereotyping, and harmful
    associations or inaccuracies.
*   **General medical harms:** Evaluation of text-to-text and image-to-text
    prompts covering safety policies, including information quality and harmful
    associations or inaccuracies.

In addition to development level evaluations, we conduct "assurance evaluations"
which are our "arms-length" internal evaluations for responsibility governance
decision making. They are conducted separately from the model development team,
to inform decision making about release. High-level findings are fed back to the
model team, but prompt sets are held out to prevent overfitting and preserve the
results' ability to inform decision making. Notable assurance evaluation results
are reported to our Responsibility & Safety Council as part of release review.

#### Evaluation results

For all areas of safety testing, we saw safe levels of performance across the
categories of child safety, content safety, and representational harms. All
testing was conducted without safety filters to evaluate the model capabilities
and behaviors. For text-to-text, image-to-text, and audio-to-text, and across
both MedGemma model sizes, the model produced minimal policy violations. A
limitation of our evaluations was that they included primarily English language
prompts.

## Data card

### Dataset overview

#### Training

The base Gemma models are pre-trained on a large corpus of text and code data.
MedGemma 4B utilizes a [SigLIP](https://arxiv.org/abs/2303.15343) image encoder
that has been specifically pre-trained on a variety of de-identified medical
data, including radiology images, histopathology images, ophthalmology images,
and dermatology images. Its LLM component is trained on a diverse set of medical
data, including medical text relevant to radiology images, chest-x rays,
histopathology patches, ophthalmology images and dermatology images.

#### Evaluation

MedGemma models have been evaluated on a comprehensive set of clinically
relevant benchmarks, including over 22 datasets across 5 different tasks and 6
medical image modalities. These include both open benchmark datasets and curated
datasets, with a focus on expert human evaluations for tasks like CXR report
generation and radiology VQA.

### Ethics and safety evaluation

#### Evaluation approach

Our evaluation methods include structured evaluations and internal red-teaming
testing of relevant content policies. Red-teaming was conducted by a number of
different teams, each with different goals and human evaluation metrics. These
models were evaluated against a number of different categories relevant to
ethics and safety, including:

*   **Child safety**: Evaluation of text-to-text and image-to-text prompts
    covering child safety policies, including child sexual abuse and
    exploitation.
*   **Content safety:** Evaluation of text-to-text and image-to-text prompts
    covering safety policies, including harassment, violence and gore, and hate
    speech.
*   **Representational harms**: Evaluation of text-to-text and image-to-text
    prompts covering safety policies, including bias, stereotyping, and harmful
    associations or inaccuracies.
*   **General medical harms:** Evaluation of text-to-text and image-to-text
    prompts covering safety policies, including information quality and harmful
    associations or inaccuracies.

In addition to development level evaluations, we conduct "assurance evaluations"
which are our "arms-length" internal evaluations for responsibility governance
decision making. They are conducted separately from the model development team,
to inform decision making about release. High-level findings are fed back to the
model team, but prompt sets are held out to prevent overfitting and preserve the
results' ability to inform decision making. Notable assurance evaluation results
are reported to our Responsibility & Safety Council as part of release review.

#### Evaluation results

For all areas of safety testing, we saw safe levels of performance across the
categories of child safety, content safety, and representational harms. All
testing was conducted without safety filters to evaluate the model capabilities
and behaviors. For text-to-text, image-to-text, and audio-to-text, and across
both MedGemma model sizes, the model produced minimal policy violations. A
limitation of our evaluations was that they included primarily English language
prompts.

## Data card

### Dataset overview

#### Training

The base Gemma models are pre-trained on a large corpus of text and code data.
MedGemma multimodal variants utilize a
[SigLIP](https://arxiv.org/abs/2303.15343) image encoder that has been
specifically pre-trained on a variety of de-identified medical data, including
radiology images, histopathology images, ophthalmology images, and dermatology
images. Their LLM component is trained on a diverse set of medical data,
including medical text, medical question-answer pairs, FHIR-based electronic
health record data (27B multimodal only), radiology images, histopathology
patches, ophthalmology images, and dermatology images.

#### Evaluation

MedGemma models have been evaluated on a comprehensive set of clinically
relevant benchmarks, including over 22 datasets across 6 different tasks and 4
medical image modalities. These benchmarks include both open and internal
datasets.

#### Source

MedGemma utilizes a combination of public and private datasets.

This model was trained on diverse public datasets including MIMIC-CXR (chest
X-rays and reports), ChestImaGenome: Set of bounding boxes linking image
findings with anatomical regions for MIMIC-CXR (MedGemma 27B multimodal only),
SLAKE (multimodal medical images and questions), PAD-UFES-20 (skin lesion images
and data), SCIN (dermatology images), TCGA (cancer genomics data), CAMELYON
(lymph node histopathology images), PMC-OA (biomedical literature with images),
and Mendeley Digital Knee X-Ray (knee X-rays).

Additionally, multiple diverse proprietary datasets were licensed and
incorporated (described next).

### Data Ownership and Documentation

*   [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.1.0/): MIT Laboratory
    for Computational Physiology and Beth Israel Deaconess Medical Center
    (BIDMC).
*   [Slake-VQA](https://www.med-vqa.com/slake/): The Hong Kong Polytechnic
    University (PolyU), with collaborators including West China Hospital of
    Sichuan University and Sichuan Academy of Medical Sciences / Sichuan
    Provincial People's Hospital.
*   [PAD-UFES-20](https://pmc.ncbi.nlm.nih.gov/articles/PMC7479321/): Federal
    University of Espírito Santo (UFES), Brazil, through its Dermatological and
    Surgical Assistance Program (PAD).
*   [SCIN](https://github.com/google-research-datasets/scin): A collaboration
    between Google Health and Stanford Medicine.
*   [TCGA](https://portal.gdc.cancer.gov/) (The Cancer Genome Atlas): A joint
    effort of National Cancer Institute and National Human Genome Research
    Institute. Data from TCGA are available via the Genomic Data Commons (GDC)
*   [CAMELYON](https://camelyon17.grand-challenge.org/Data/): The data was
    collected from Radboud University Medical Center and University Medical
    Center Utrecht in the Netherlands.
*   [PMC-OA (PubMed Central Open Access
    Subset)](https://catalog.data.gov/dataset/pubmed-central-open-access-subset-pmc-oa):
    Maintained by the National Library of Medicine (NLM) and National Center for
    Biotechnology Information (NCBI), which are part of the NIH.
*   [MedQA](https://arxiv.org/pdf/2009.13081): This dataset was created by a
    team of researchers led by Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung
    Weng, Hanyi Fang, and Peter Szolovits
*   [Mendeley Digital Knee
    X-Ray](https://data.mendeley.com/datasets/t9ndx37v5h/1): This dataset is
    from Rani Channamma University, and is hosted on Mendeley Data.
*   [AfriMed-QA](https://afrimedqa.com/): This data was developed and led by
    multiple collaborating organizations and researchers include key
    contributors: Intron Health, SisonkeBiotik, BioRAMP, Georgia Institute of
    Technology, and MasakhaneNLP.
*   [VQA-RAD](https://www.nature.com/articles/sdata2018251): This dataset was
    created by a research team led by Jason J. Lau, Soumya Gayen, Asma Ben
    Abacha, and Dina Demner-Fushman and their affiliated institutions (the US
    National Library of Medicine and National Institutes of Health)
*   [Chest ImaGenome](https://physionet.org/content/chest-imagenome/1.0.0/): IBM
    Research.
*   [MedExpQA](https://www.sciencedirect.com/science/article/pii/S0933365724001805):
    This dataset was created by researchers at the HiTZ Center (Basque Center
    for Language Technology and Artificial Intelligence).
*   [MedXpertQA](https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA): This
    dataset was developed by researchers at Tsinghua University (Beijing, China)
    and Shanghai Artificial Intelligence Laboratory (Shanghai, China).
*   [HealthSearchQA](https://huggingface.co/datasets/katielink/healthsearchqa):
    This dataset consists of consisting of 3,173 commonly searched consumer
    questions

In addition to the public datasets listed above, MedGemma was also trained on
de-identified, licensed datasets or datasets collected internally at Google from
consented participants.

*   **Radiology dataset 1:** De-identified dataset of different CT studies
    across body parts from a US-based radiology outpatient diagnostic center
    network.
*   **Ophthalmology dataset 1 (EyePACS):** De-identified dataset of fundus
    images from diabetic retinopathy screening.
*   **Dermatology dataset 1:** De-identified dataset of teledermatology skin
    condition images (both clinical and dermatoscopic) from Colombia.
*   **Dermatology dataset 2:** De-identified dataset of skin cancer images (both
    clinical and dermatoscopic) from Australia.
*   **Dermatology dataset 3:** De-identified dataset of non-diseased skin images
    from an internal data collection effort.
*   **Pathology dataset 1:** De-identified dataset of histopathology H\&E whole
    slide images created in collaboration with an academic research hospital and
    biobank in Europe. Comprises de-identified colon, prostate, and lymph nodes.
*   **Pathology dataset 2:** De-identified dataset of lung histopathology H\&E
    and IHC whole slide images created by a commercial biobank in the United
    States.
*   **Pathology dataset 3:** De-identified dataset of prostate and lymph node
    H\&E and IHC histopathology whole slide images created by a contract
    research organization in the United States.
*   **Pathology dataset 4:** De-identified dataset of histopathology whole slide
    images created in collaboration with a large, tertiary teaching hospital in
    the United States. Comprises a diverse set of tissue and stain types,
    predominantly H\&E.
*   **EHR dataset 1:** Question/answer dataset drawn from synthetic FHIR records
    created by [Synthea.](https://synthetichealth.github.io/synthea/) The test
    set includes 19 unique patients with 200 questions per patient divided into
    10 different categories.

### Data citation

*   **MIMIC-CXR:** Johnson, A., Pollard, T., Mark, R., Berkowitz, S., & Horng,
    S. (2024). MIMIC-CXR Database (version 2.1.0). PhysioNet.
    [https://physionet.org/content/mimic-cxr/2.1.0/](https://physionet.org/content/mimic-cxr/2.1.0/)
    *and* Johnson, Alistair E. W., Tom J. Pollard, Seth J. Berkowitz, Nathaniel
    R. Greenbaum, Matthew P. Lungren, Chih-Ying Deng, Roger G. Mark, and Steven
    Horng. 2019\. "MIMIC-CXR, a de-Identified Publicly Available Database of
    Chest Radiographs with Free-Text Reports." *Scientific Data 6* (1): 1–8.

*   **SLAKE:** Liu, Bo, Li-Ming Zhan, Li Xu, Lin Ma, Yan Yang, and Xiao-Ming Wu.
    2021.SLAKE: A Semantically-Labeled Knowledge-Enhanced Dataset for Medical
    Visual Question Answering."
    [http://arxiv.org/abs/2102.09542](http://arxiv.org/abs/2102.09542).

*   **PAD-UEFS-20:** Pacheco, Andre GC, et al. "PAD-UFES-20: A skin lesion
    dataset composed of patient data and clinical images collected from
    smartphones." *Data in brief* 32 (2020): 106221\.

*   **SCIN:** Ward, Abbi, Jimmy Li, Julie Wang, Sriram Lakshminarasimhan, Ashley
    Carrick, Bilson Campana, Jay Hartford, et al. 2024\. "Creating an Empirical
    Dermatology Dataset Through Crowdsourcing With Web Search Advertisements."
    *JAMA Network Open 7* (11): e2446615–e2446615.

*   **TCGA:** The results shown here are in whole or part based upon data
    generated by the TCGA Research Network:
    [https://www.cancer.gov/tcga](https://www.cancer.gov/tcga).

*   **CAMELYON16:** Ehteshami Bejnordi, Babak, Mitko Veta, Paul Johannes van
    Diest, Bram van Ginneken, Nico Karssemeijer, Geert Litjens, Jeroen A. W. M.
    van der Laak, et al. 2017\. "Diagnostic Assessment of Deep Learning
    Algorithms for Detection of Lymph Node Metastases in Women With Breast
    Cancer." *JAMA 318* (22): 2199–2210.

*   **Mendeley Digital Knee X-Ray:** Gornale, Shivanand; Patravali, Pooja
    (2020), "Digital Knee X-ray Images", Mendeley Data, V1, doi:
    10.17632/t9ndx37v5h.1

*   **VQA-RAD:** Lau, Jason J., Soumya Gayen, Asma Ben Abacha, and Dina
    Demner-Fushman. 2018\. "A Dataset of Clinically Generated Visual Questions
    and Answers about Radiology Images." *Scientific Data 5* (1): 1–10.

*   **Chest ImaGenome:** Wu, J., Agu, N., Lourentzou, I., Sharma, A., Paguio,
    J., Yao, J. S., Dee, E. C., Mitchell, W., Kashyap, S., Giovannini, A., Celi,
    L. A., Syeda-Mahmood, T., & Moradi, M. (2021). Chest ImaGenome Dataset
    (version 1.0.0). PhysioNet. RRID:SCR\_007345.
    [https://doi.org/10.13026/wv01-y230](https://doi.org/10.13026/wv01-y230)

*   **MedQA:** Jin, Di, Eileen Pan, Nassim Oufattole, Wei-Hung Weng, Hanyi Fang,
    and Peter Szolovits. 2020\. "What Disease Does This Patient Have? A
    Large-Scale Open Domain Question Answering Dataset from Medical Exams."
    [http://arxiv.org/abs/2009.13081](http://arxiv.org/abs/2009.13081).

*   **AfrimedQA:** Olatunji, Tobi, Charles Nimo, Abraham Owodunni, Tassallah
    Abdullahi, Emmanuel Ayodele, Mardhiyah Sanni, Chinemelu Aka, et al. 2024\.
    "AfriMed-QA: A Pan-African, Multi-Specialty, Medical Question-Answering
    Benchmark Dataset."
    [http://arxiv.org/abs/2411.15640](http://arxiv.org/abs/2411.15640).

*   **MedExpQA:** Alonso, I., Oronoz, M., & Agerri, R. (2024). MedExpQA:
    Multilingual Benchmarking of Large Language Models for Medical Question
    Answering. *arXiv preprint arXiv:2404.05590*. Retrieved from
    [https://arxiv.org/abs/2404.05590](https://arxiv.org/abs/2404.05590)

*   **MedXpertQA:** Zuo, Yuxin, Shang Qu, Yifei Li, Zhangren Chen, Xuekai Zhu,
    Ermo Hua, Kaiyan Zhang, Ning Ding, and Bowen Zhou. 2025\. "MedXpertQA:
    Benchmarking Expert-Level Medical Reasoning and Understanding."
    [http://arxiv.org/abs/2501.18362](http://arxiv.org/abs/2501.18362).

### De-identification/anonymization:

Google and its partners utilize datasets that have been rigorously anonymized or
de-identified to ensure the protection of individual research participants and
patient privacy.

## Implementation information

Details about the model internals.

### Software

Training was done using [JAX](https://github.com/jax-ml/jax).

JAX allows researchers to take advantage of the latest generation of hardware,
including TPUs, for faster and more efficient training of large models.

## Use and limitations

### Intended use

MedGemma is an open multimodal generative AI model intended to be used as a
starting point that enables more efficient development of downstream healthcare
applications involving medical text and images. MedGemma is intended for
developers in the life sciences and healthcare space. Developers are responsible
for training, adapting and making meaningful changes to MedGemma to accomplish
their specific intended use. MedGemma models can be fine-tuned by developers
using their own proprietary data for their specific tasks or solutions.

MedGemma is based on Gemma 3 and has been further trained on medical images and
text. MedGemma enables further development in any medical context (image and
textual), however the model was pre-trained using chest X-ray, pathology,
dermatology, and fundus images. Examples of tasks within MedGemma's training
include visual question answering pertaining to medical images, such as
radiographs, or providing answers to textual medical questions. Full details of
all the tasks MedGemma has been evaluated can be found in the [MedGemma
Technical Report](https://arxiv.org/abs/2507.05201).

### Benefits

*   Provides strong baseline medical image and text comprehension for models of
    its size.
*   This strong performance makes it efficient to adapt for downstream
    healthcare-based use cases, compared to models of similar size without
    medical data pre-training.
*   This adaptation may involve prompt engineering, grounding, agentic
    orchestration or fine-tuning depending on the use case, baseline validation
    requirements, and desired performance characteristics.

### Limitations

MedGemma is not intended to be used without appropriate validation, adaptation
and/or making meaningful modification by developers for their specific use case.
The outputs generated by MedGemma are not intended to directly inform clinical
diagnosis, patient management decisions, treatment recommendations, or any other
direct clinical practice applications. Performance benchmarks highlight baseline
capabilities on relevant benchmarks, but even for image and text domains that
constitute a substantial portion of training data, inaccurate model output is
possible. All outputs from MedGemma should be considered preliminary and require
independent verification, clinical correlation, and further investigation
through established research and development methodologies.

MedGemma's multimodal capabilities have been primarily evaluated on single-image
tasks. MedGemma has not been evaluated in use cases that involve comprehension
of multiple images.

MedGemma has not been evaluated or optimized for multi-turn applications.

MedGemma's training may make it more sensitive to the specific prompt used than
Gemma 3\.

When adapting MedGemma developer should consider the following:

*   **Bias in validation data:** As with any research, developers should ensure
    that any downstream application is validated to understand performance using
    data that is appropriately representative of the intended use setting for
    the specific application (e.g., age, sex, gender, condition, imaging device,
    etc).
*   **Data contamination concerns**: When evaluating the generalization
    capabilities of a large model like MedGemma in a medical context, there is a
    risk of data contamination, where the model might have inadvertently seen
    related medical information during its pre-training, potentially
    overestimating its true ability to generalize to novel medical concepts.
    Developers should validate MedGemma on datasets not publicly available or
    otherwise made available to non-institutional researchers to mitigate this
    risk.


### Release notes

*   May 20, 2025: Initial Release
*   July 9, 2025 Bug Fix: Fixed the subtle degradation in the multimodal
    performance. The issue was due to a missing end-of-image token in the model
    vocabulary, impacting combined text-and-image tasks. This fix reinstates and
    correctly maps that token, ensuring text-only tasks remain unaffected while
    restoring multimodal performance.