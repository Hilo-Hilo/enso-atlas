---
license: other
license_name: health-ai-developer-foundations
license_link: https://developers.google.com/health-ai-developer-foundations/terms
language:
  - en
tags:
  - medical
  - pathology
  - digital-pathology
  - medical-embeddings
  - image-classification
  - image-feature-extraction
extra_gated_heading: Access Path Foundation on Hugging Face
extra_gated_prompt: >-
  To access Path Foundation on Hugging Face, you're required to review and
  agree to [Health AI Developer Foundation's terms of use](https://developers.google.com/health-ai-developer-foundations/terms).
  To do this, please ensure you're logged in to Hugging Face and click below.
  Requests are processed immediately.
extra_gated_button_content: Acknowledge license
---

# Path Foundation model card

**Model documentation**:
[Path Foundation](https://developers.google.com/health-ai-developer-foundations/path-foundation)

**Resources**:

*   Model on Google Cloud Model Garden:
    [Path Foundation](https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/path-foundation)
*   Model on Hugging Face:
    [google/path-foundation](https://huggingface.co/google/path-foundation)
*   GitHub repository (supporting code, Colab notebooks, discussions, and
    issues): [path-foundation](https://github.com/google-health/path-foundation)
*   Quick start notebook:
    [notebooks/quick_start](https://github.com/google-health/path-foundation/blob/master/notebooks/quick_start_with_hugging_face.ipynb)
*   Support: See
    [Contact](https://developers.google.com/health-ai-developer-foundations/path-foundation/get-started.md#contact).

**Terms of use**:
[Health AI Developer Foundations terms of use](https://developers.google.com/health-ai-developer-foundations/terms)

**Author**: Google

## Model information

This section describes the Path Foundation model and how to use it.

### Description

Path Foundation is a machine learning model for use in histopathology
applications. It produces embeddings that can be used to efficiently train
classifier models for pathology analysis tasks on hematoxylin and eosin (H&E)
patches from whole slide images (WSI) with less data and less compute. Path
Foundation is trained using self-supervised learning in order to create
embeddings from 224 x 224 pixel image patches from histopathology WSIs. The
embeddings returned by the Path Foundation are 384 dimensional vectors of
floating point values that represent a projection of the original image into a
compressed feature space.

You can read more about the research and underlying model in our manuscript,
[Domain-specific optimization and diverse evaluation of self-supervised models
for histopathology](https://arxiv.org/abs/2310.13259).

### How to use

Following are some example code snippets to help you quickly get started running
the model locally. If you want to use the model at scale, we recommend that you
create a production version using
[Model Garden](https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/path-foundation).

```python
from PIL import Image as PILImage
from huggingface_hub import hf_hub_download, from_pretrained_keras
import tensorflow as tf
import numpy as np

# Download a test image from Hugging Face Hub
hf_hub_download(repo_id="google/path-foundation", filename='Test.png', local_dir='.')

# Open the image, crop it to match expected input size.
img = PILImage.open("Test.png").crop((0, 0, 224, 224)).convert('RGB')

# Convert the image to a Tensor and scale to [0, 1] (in case needed)
tensor = tf.cast(tf.expand_dims(np.array(img), axis=0), tf.float32) / 255.0

# Load the model directly from Hugging Face Hub
loaded_model = from_pretrained_keras("google/path-foundation")

# Call inference
infer = loaded_model.signatures["serving_default"]
embeddings = infer(tf.constant(tensor))

# Extract the embedding vector
embedding_vector = embeddings['output_0'].numpy().flatten()
```

### Examples

See the following Colab notebooks for examples of how to use Path Foundation:

*   To give the model a quick try, running it locally with weights from Hugging
    Face, see
    [Quick start notebook in Colab](https://colab.research.google.com/github/google-health/path-foundation/blob/master/notebooks/quick_start_with_hugging_face.ipynb).

*   For an example of how to use the model to train a linear classifier using
    data from Google Cloud
    [DICOM Store](https://cloud.google.com/healthcare-api/docs/how-tos/dicom),
    see
    [DICOM linear classifier notebook in Colab](https://colab.research.google.com/github/google-health/path-foundation/blob/master/notebooks/train_data_efficient_classifier_dicom.ipynb).

*   For an example of how to use the model to train a linear classifier using
    data from Google Cloud Storage (GCS), see
    [GCS endpoint linear classifier notebook in Colab](https://colab.research.google.com/github/google-health/path-foundation/blob/master/notebooks/train_data_efficient_classifier_gcs.ipynb).

### Model architecture overview

Path Foundation uses the ViT-S architecture and was trained using
[Masked Siamese Networks](https://arxiv.org/pdf/2204.07141) across
magnifications with domain-specific tuning and optimization. The resulting
feature representations provided by the model offer robust input for downstream
tasks in histopathology. Additional information can be found in the preprint
[Domain-specific optimization and diverse evaluation of self-supervised models
for histopathology](https://arxiv.org/pdf/2310.13259).

### Technical specifications

*   Model type: ViT-S architecture
*   Manuscript: [Domain-specific optimization and diverse evaluation of
    self-supervised models for histopathology](https://arxiv.org/abs/2310.13259)
*   Model created: 2023-12-19
*   Model version: Version: 1.0.0

### Performance and validation

Linear probe evaluation was conducted across a diverse set of 11 benchmark tasks
involving 17 unique tissue types and spanning different optimal magnifications
and task types. See the [manuscript](https://arxiv.org/abs/2310.13259) for more
details, including additional results for slide-level tasks (e.g., tissue type
classification and molecular findings) and fine tuning with data titration.

### Key performance metrics

*   **93%** - A Linear Probing AUC for a suite of histopathology classification
    tasks. 95% CI: \[92.9 - 93.8\]

### Inputs and outputs

*   **Input**: Image patch of 224 x 224 pixels from H&E Whole Slide Images
    (WSIs).

    Path Foundation is closely integrated with
    [EZ-WSI](https://github.com/GoogleCloudPlatform/EZ-WSI-DICOMweb/tree/main),
    a library for digital pathology that lets you process WSIs to patches and
    send them to the model.

*   **Output**: Embedding vector of floating point values (Dimensions: 384).

## Dataset details

### Training dataset

Training data consisted of hematoxylin and eosin stained (H&E) WSIs from The
Cancer Genome Atlas (TCGA), accessed at
[https://portal.gdc.cancer.gov](https://portal.gdc.cancer.gov). Training was
performed using 60 million patches across three magnifications (~2 µm/pixel, ~1
µm/pixel, ~0.5 µm/pixel) and across the 32 solid tumor TCGA studies
(representing different cancer types and with training data including both tumor
and diverse, non-tumor patches).

### Labeling

Model was trained using self-supervised learning, meaning no supervised labels
were used. Labels used to measure model performance on downstream tasks were
provided either through pathologist annotation or slide-level metadata.

*Additional information about data and labels used for downstream tasks can be
found in the following references:*

-   [Benjordi, B. et al. Diagnostic Assessment of Deep Learning Algorithms for
    Detection of Lymph Node Metastases in Women With Breast Cancer. JAMA
    (2017).](https://jamanetwork.com/journals/jama/fullarticle/2665774)
-   [Jaroensri, R. et al. Deep learning models for histologic grading of breast
    cancer and association with disease prognosis. npj Breast Cancer 8, 1–12
    (2022).](https://www.nature.com/articles/s41523-022-00478-y)
-   [Liu, Y. et al. Artificial Intelligence-Based Breast Cancer Nodal Metastasis
    Detection: Insights Into the Black Box for Pathologists. Arch. Pathol. Lab.
    Med. 143, (2019).](https://pubmed.ncbi.nlm.nih.gov/30295070/)
-   [Lai, J. et al. Domain-specific optimization and diverse evaluation of
    self-supervised models for histopathology. arXiv
    (2023).](https://arxiv.org/abs/2310.13259)
-   [Nagpal, K. et al. Development and Validation of a Deep Learning Algorithm
    for Gleason Grading of Prostate Cancer From Biopsy Specimens. JAMA Oncol 6,
    1372–1380
    (2020).](https://jamanetwork.com/journals/jamaoncology/fullarticle/2768225)
-   [Nagpal, K. et al. Development and validation of a deep learning algorithm
    for improving Gleason scoring of prostate cancer. npj Digital Medicine 2,
    1–10 (2019).](https://www.nature.com/articles/s41746-019-0112-2)
-   [Sadhwani, A. et al. Comparative analysis of machine learning approaches to
    classify tumor mutation burden in lung adenocarcinoma using histopathology
    images. Sci. Rep. 11, 1–11
    (2021).](https://www.nature.com/articles/s41598-021-95747-4)
-   [Wulczyn, E. et al. Interpretable survival prediction for colorectal cancer
    using deep learning. NPJ Digital Medicine 4,
    (2021).](https://www.nature.com/articles/s41746-021-00427-2)
-   [Weng, WH. et al. Multimodal Multitask Representation Learning for Pathology
    Biobank Metadata Prediction. arXiv
    (2019).](https://arxiv.org/abs/1909.07846)

## License

The use of Path Foundations is governed by the
[Health AI Developer Foundations terms of use](https://developers.google.com/health-ai-developer-foundations/terms).

## Data citation

The results of Path Foundation are in whole or in part based upon data generated
by the [TCGA Research Network](https://www.cancer.gov/tcga).

## Implementation information

This section provides details about the model internals.

### Software

Training was done using [JAX](https://github.com/jax-ml/jax). JAX allows
researchers to take advantage of the latest generation of hardware, including
TPUs, for faster and more efficient training of large models.

## Use and limitations

### Intended use

*   Path Foundation can reduce the training data, compute, and technical
    expertise necessary to develop task-specific models for H&E pathology
    slides.

*   Embeddings from the model can be used for a variety of user-defined
    downstream tasks including, but not limited to: cancer detection,
    classification, and grading; metadata prediction (stain, tissue type,
    specimen type, etc.); quality assessment (e.g., imaging artifacts); and
    similar image search.

*   The embeddings can also be used to explore the feature space of
    histopathology images for biomarker development associated with prognostic
    and predictive tasks.

### Benefits

*   Path Foundation Embeddings can be used for efficient training of AI
    development for H&E histopathology image analysis with significantly less
    data and compute than traditional methods.

*   By leveraging the large set of pre-trained images Path Foundation is trained
    on, users need less data but can also build more generalizable models than
    training on more limited datasets.

*   Provides a rich, compressed representation of histopathology image patches.

*   Helps users build AI classifiers for a variety of different applications
    with less data and with less compute.

### Limitations

Below are the number of known factors that may degrade model performance or
decrease confidence in the model results:

*   The model has only been validated on a limited number of the many potential
    downstream tasks involving H&E histopathology.

*   This model version was trained and validated only on H&E images from a
    limited set of scanners and countries.

*   Model output may not generalize well to data from other image types, patient
    populations, or scanner manufacturers not used in training.

*   Task-specific validation remains an important aspect of downstream model
    development by the end user.

*   Training and validation was performed on patches corresponding to 5x, 10x,
    and 20x magnification (~2 µm/pixel, ~1 µm/pixel, and ~0.5 µm/pixel,
    respectively). Using input patches corresponding to magnifications other
    than these has not been evaluated.

*   The model is only used to generate embeddings of user-provided data. It does
    not generate any predictions or diagnosis on its own.

*   As with any research, developers should ensure that any downstream
    application is validated to understand performance using data that is
    appropriately representative of the intended use setting for the specific
    application (e.g., age, sex, gender, condition, scanner, etc.).