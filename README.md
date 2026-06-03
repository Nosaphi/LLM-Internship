# PDF Anonymization Pipeline

Automated pipeline for extracting, preprocessing, and anonymizing Italian administrative PDF documents. Handles both digital and scanned PDFs, applies NER-based anonymization, and outputs a structured dataset.

---

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Virtual Environment Setup](#virtual-environment-setup)
- [External Dependencies](#external-dependencies)
- [GPU Acceleration](#gpu-acceleration)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Output](#output)
- [T5 Classification Notebook (Google Colab)](#t5-classification-notebook-google-colab)

---

## Overview

The pipeline performs the following steps for each PDF:

1. Detects whether the PDF is digital or scanned
2. Extracts text directly (digital) or via OCR (scanned)
3. Cleans and normalizes the extracted text
4. Anonymizes sensitive data (emails, phone numbers, names, IDs, addresses) using regex and a spaCy NER model
5. Exports the anonymized dataset as `.jsonl` and optionally `.csv`, along with a `mapping.json` pseudonymization dictionary

---

## Requirements

- **Python** 3.9+
- **Tesseract OCR** (system-level install, not a Python package)
- **Poppler** (required by `pdf2image` to convert PDF pages to images)

### Install Tesseract

**Ubuntu/Debian:**
```bash
sudo apt install tesseract-ocr tesseract-ocr-ita
```

---

### Install Poppler

**Ubuntu/Debian:**
```bash
sudo apt install poppler-utils
```

---

## Virtual Environment Setup

It is strongly recommended to use a virtual environment to avoid dependency conflicts.

```bash
# Create the virtual environment
python -m venv venv

# Activate it
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Install all Python dependencies
pip install pypdf pymupdf pdf2image pytesseract spacy opencv-python numpy pandas unicodedata2 ftfy
```

### Download the spaCy Italian model

```bash
python -m spacy download it_core_news_lg
```

---

## External Dependencies

| Dependency | Role | Install |
|---|---|---|
| `tesseract-ocr` | OCR engine for scanned PDFs | System package |
| `tesseract-ocr-ita` | Italian language pack for Tesseract | System package |
| `poppler-utils` | PDF-to-image conversion backend | System package |
| `it_core_news_lg` | spaCy Italian NER model | `python -m spacy download` |

---

## GPU Acceleration

### OpenCV (CPU only)

The OCR preprocessing step uses OpenCV, which runs on CPU by default. For most use cases this is fast enough since the bottleneck is Tesseract, not OpenCV.

### Tesseract (CPU / GPU via LSTM)

Tesseract uses an LSTM-based OCR engine (`--oem 1`). It runs on CPU only by default.  
GPU acceleration for Tesseract is not officially supported and is generally not recommended.

### spaCy NER

spaCy uses CPU by default. If you have a CUDA-compatible GPU and want to accelerate NER inference:

```bash
# Install the GPU-compatible spaCy version
pip install spacy[cuda12x]   # for CUDA 12.x
pip install spacy[cuda11x]   # for CUDA 11.x

# Then download the model as usual
python -m spacy download it_core_news_lg
```

> For large batch processing (thousands of documents), GPU inference for spaCy can offer a significant speedup. For small batches, the CPU version is sufficient.

---

## Project Structure

```
.
├── main.py                  # Entry point
├── classes.json             # Label-to-encoding mapping
├── README.md
└── /media/protocolli/       # Input directory for JSON protocol files
```

---


## Usage

```bash
# Activate your virtual environment first
source venv/bin/activate

# Run the pipeline
python main.py
```

The script will process all PDFs referenced in the JSON files found in the `dir` directory and print the file count as it progresses.

---

## Output

| File | Description |
|---|---|
| `data.jsonl` | Anonymized dataset in JSON Lines format |
| `data.csv` | Same dataset as CSV (optional, set `csv=True` in `output()`) |
| `mapping.json` | Pseudonymization dictionary mapping original values to tokens |

### `data.jsonl` format

```json
{"protocolID": "12345", "text": "...", "label": "Example", "encoding": 0.0.0, "date": "2024-01-15T00:00:00.000Z"}
```

### `mapping.json` format

```json
{
  "PERSON": { "Mario Rossi": "PERSON1"},
  "EMAIL":  { "m.rossi@comune.it": "EMAIL1" },
  ...
}
```

> The mapping is consistent within a single run: the same name always maps to the same token. It resets between runs.

---

## T5 Classification Notebook (Google Colab)

The notebook `PipelineT5.ipynb` fine-tunes a `t5-small` model to classify anonymized documents into top-level encoding categories. It is designed to run on **Google Colab** with GPU acceleration.

### Prerequisites

Before running the notebook, you need to upload the `data.jsonl` file produced by the anonymization pipeline (see [Output](#output) above).

#### Upload `data.jsonl` to Colab

The notebook reads the dataset from the following hardcoded path:

```python
with open('/content/data.jsonl', 'r', encoding='utf-8') as f:
```

You **must** place `data.jsonl` at `/content/data.jsonl` in the Colab runtime or change the path in the code. To do it on Colab, you can click on the folder icon on the left menu then on the import file icon. If you did not change anything, it should be in the content file in Colab.

### Enable GPU Runtime

The notebook automatically detects and uses a GPU if available (`fp16=torch.cuda.is_available()`). To enable GPU in Colab:

1. Go to **Runtime → Change runtime type**
2. Set **Hardware accelerator** to **GPU T4** (or better)
3. Click **Save**

Training on CPU is possible but significantly slower (40 epochs on a large dataset can take hours).

---

### Install Dependencies

The first cell installs all required Python packages:

```python
!pip install transformers datasets evaluate accelerate sentencepiece scikit-learn -q
```

No additional system-level packages are required.

---

### Output Files

After training, the notebook saves the following files to the Colab runtime's local filesystem:

| Path | Description |
|---|---|
| `./t5_encoding_classifier/` | Intermediate checkpoints saved each epoch |
| `./t5_encoding_classifier_final/` | Best model weights and tokenizer |
| `./t5_encoding_classifier_final/metadata.json` | Training metadata (categories, accuracy, sizes) |
| `./logs/` | Training logs |

> **Important:** Files saved to `/content/` are lost when the Colab session ends. Download the final model before closing your session:
>
> ```python
> from google.colab import files
> import shutil
>
> shutil.make_archive('t5_model', 'zip', './t5_encoding_classifier_final')
> files.download('t5_model.zip')
> ```

---

### Training Configuration

| Parameter | Value |
|---|---|
| Base model | `t5-small` |
| Max input length | 256 tokens |
| Max output length | 8 tokens |
| Epochs | 40 (with early stopping, patience=5) |
| Batch size | 4 (train & eval) |
| Learning rate | 5e-4 (cosine scheduler) |
| Evaluation metric | Exact-match accuracy |
| Beam search | 4 beams |

The input prompt prefix used during training and inference is:

```
classify encoding: <document text>
```

---

### Inference After Training

Once the model is trained, you can classify any new document with:

```python
result = predict_category("your anonymized document text here")
print(result['predicted_category'])
```

To reload the saved model in a new session:

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

model = T5ForConditionalGeneration.from_pretrained('./t5_encoding_classifier_final')
tokenizer = T5Tokenizer.from_pretrained('./t5_encoding_classifier_final')
```
