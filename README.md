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
