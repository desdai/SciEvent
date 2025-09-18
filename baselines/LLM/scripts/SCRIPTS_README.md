
# SciEvent LLM Scripts: Complete Guide

This README provides comprehensive guidance on running the SciEvent Large Language Model (LLM) pipeline for scientific event extraction tasks.

The pipeline consists of two main tasks:
1. **Event Segmentation** - Breaking abstracts into semantic segments
2. **Event Extraction** - Extracting argument structures from events

Each task requires running a script to generate raw output, followed by a post-processing script to extract structured results.

---

## Installation

Install the required dependencies:

```bash
# Install dependencies using pip
pip install -r requirements.txt
```

---

## Complete Workflow

### 1. Event Segmentation

**Step 1a: Generate Raw Segmentation Output**

Choose one approach:

- **Open-source model:**
```bash
python Event_segmentation.py \
  --domains ACL BIOINFO CSCW DH JMIR \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --prompt-template Zero-Shot_Event_Segmentation \
  --output-base-dir ./baselines/LLM/output/Event_segmentation \
  --input-dir ./SciEvent_data/raw/domain_specific_unannotated \
  --clean-cache
```

Model choices are: ``"Qwen/Qwen2.5-7B-Instruct"``, ``"meta-llama/Meta-Llama-3.1-8B-Instruct"``, ``"deepseek-ai/DeepSeek-R1-Distill-Llama-8B"``

- **OpenAI model:**
```bash
python Event_segmentation_openai.py \
  --domains ACL BIOINFO CSCW DH JMIR \
  --model "gpt-4.1" \
  --prompt-template Zero-Shot_Event_Segmentation \
  --output-base-dir ./baselines/LLM/output/Event_Segmentation \
  --input-dir ./SciEvent_data/raw/domain_specific_unannotated \
  --api-key "YOUR_API_KEY" \
  --max-concurrent 5
```

**Step 1b: Process Raw Segmentation Output**

After generating raw output, extract the structured segments:

```bash
python LLM_ES_extraction.py \
  --domains ACL BIOINFO CSCW DH JMIR \
  --model-name Meta-Llama-3.1-8B-Instruct \
  --prompt-template Zero-Shot_Event_Segmentation \
  --base-dir ./baselines/LLM/output/Event_Segmentation \
  --input-dir ./SciEvent_data/raw/domain_specific_unannotated
```


### 2. Event Extraction

**Step 2a: Generate Raw Extraction Output**

Choose one approach:

- **Standard Event Extraction (open-source):**

Model choices are: ``"Qwen/Qwen2.5-7B-Instruct"``, ``"meta-llama/Meta-Llama-3.1-8B-Instruct"``, ``"deepseek-ai/DeepSeek-R1-Distill-Llama-8B"``

```bash
python Event_extraction.py \
  --domains ACL BIOINFO CSCW DH JMIR \
  --model "meta-llama/Meta-Llama-3.1-8B-Instruct"\
  --prompt "Few-shot-2_Event_Extraction" \
  --output-base-dir ./baselines/LLM/output/Event_Extraction \
  --input-dir ./SciEvent_data/raw/domain_specific_unannotated
```


- **Standard Event Extraction (OpenAI):**
```bash
python Event_extraction_openai.py \
  --domains ACL BIOINFO CSCW DH JMIR \
  --model "gpt-4.1" \
  --prompt "Few-shot-2_Event_Extraction" \
  --output-base-dir ./baselines/LLM/output/Event_Extraction \
  --input-dir ./SciEvent_data/raw/domain_specific_unannotated
  --api-key "YOUR_API_KEY" \
  --max-concurrent 5
```

#### Event Type Ablation

Following use same input and output folder as above.

Model choices are: ``"Qwen/Qwen2.5-7B-Instruct"``, ``"meta-llama/Meta-Llama-3.1-8B-Instruct"``, ``"deepseek-ai/DeepSeek-R1-Distill-Llama-8B"`` if ``MODEL_TYPE = huggingface``.

Only ``gpt-4.1``is experimented using ``MODEL_TYPE = openai``. If the API call supports, you can change into any other LLMs.

- **Predicting Event Type and Extraction:**
```bash
  python Pred_Event_Type.py \
    --domains ACL BIOINFO CSCW DH JMIR \
    --model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --model-type huggingface \
    --prompt "Pred_Event_Type" \
    --clean-cache
```

- **Providing True Event Type and Extraction:**
```bash
  python True_Event_type.py \
    --domains ACL BIOINFO CSCW DH JMIR \
    --model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --model-type huggingface \
    --prompt "True_Event_Type" \
    --clean-cache
```

**Step 2b: Process Raw Extraction Output**

All event extraction approaches generate raw text files that must be processed into structured JSON format:

```bash
python LLM_EE_extraction.py \
  --domains ACL BIOINFO CSCW DH JMIR \
  --prompt-template [PROMPT_TEMPLATE] \
  --model-name "meta-llama/Meta-Llama-3.1-8B-Instruct"
```

Where `[PROMPT_TEMPLATE]` should match the prompt template used in Step 2a:
- For Event_extraction.py or Event_extraction_openai.py: Use `"One-Shot_Event_Extraction"` or other extraction templates
- For Pred_Event_Type.py: Use `"Zero-shot_Pred_Event_Type"`
- For True_Event_type.py: Use `"Zero-shot_True_Event_Type"`
Default input folder: ``./SciEvent_data/raw/domain_specific_unannotated``, default base folder: ``./baselines/LLM/output/Event_Extraction``
---

<!-- ## Command-Line Parameters

### Event Segmentation Scripts

**Event_segmentation.py:**
- `--domains`: List of domains (required, e.g., ACL BIOINFO CSCW DH JMIR)
- `--model`: HuggingFace model name (default: "meta-llama/Meta-Llama-3.1-8B-Instruct")
- `--clean-cache`: Force model redownload (optional flag)

**Event_segmentation_openai.py:**
- `--domains`: List of domains (required)
- `--model`: OpenAI model name (default: "gpt-4o")
- `--api-key`: OpenAI API key (required)
- `--max-concurrent`: Maximum concurrent requests (default: 5)

### Event Extraction Scripts

**Event_extraction.py:**
- `--domains`: List of domains (required)
- `--model`: HuggingFace model name (default: "meta-llama/Meta-Llama-3.1-8B-Instruct")
- `--clean-cache`: Force model redownload (optional flag)

**Event_extraction_openai.py:**
- `--domains`: List of domains (required)
- `--model`: OpenAI model name (default: "gpt-4o")
- `--api-key`: OpenAI API key (required)
- `--max-concurrent`: Maximum concurrent requests (default: 5)

**Pred_Event_Type.py / True_Event_type.py:**
- `--domains`: List of domains (required)
- `--model`: Model name (default: "deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
- `--model-type`: "huggingface" or "openai" (default: "huggingface")
- `--prompt`: Prompt template name (default varies by script)
- `--clean-cache`: Force model redownload (HuggingFace only)
- `--openai-api-key`: OpenAI API key (when using OpenAI models)

### Post-processing Scripts

**LLM_ES_extraction.py:**
- `--domains`: List of domains (required)
- `--prompt-template`: Prompt template name (default: "Zero-Shot_Event_Segmentation")
- `--model-name`: Model name (default: "meta-llama/Meta-Llama-3.1-8B-Instruct")
- `--base-dir`: Base output directory (default: "./output/event_segmentation")
- `--input-dir`: Input data directory (default: "./SciEvent_data/raw")

**LLM_EE_extraction.py:**
- `--domains`: List of domains (required)
- `--prompt-template`: Prompt template name (default: "One-Shot_Event_Extraction")
- `--model-name`: Model name (default: "meta-llama/Meta-Llama-3.1-8B-Instruct")
- `--base-dir`: Base output directory (default: "./output/event_extraction")
- `--input-dir`: Input data directory (default: "./SciEvent_data/raw")

--- -->

<!-- ## Directory Structure

The project follows this directory structure:

```
SciEvent/
├── SciEvent_data/
│   ├── raw/domain_specific_unannotated
│   │   ├── ACL/    # NLP abstracts 
│   │   ├── CSCW/   # Social Computing abstracts
│   │   ├── JMIR/   # Medical Informatics abstracts
│   │   ├── BIOINFO/ # Computational Biology abstracts
│   │   └── DH/     # Digital Humanities abstracts
│   ├── event_segmentation/
│   │   └── ground_truth/                # Gold standard for segmentation
│   └── event_extraction/
│       └── ground_truth/                # Gold standard for argument extraction
├── output/
│   ├── event_segmentation/
│   │   └── [MODEL_NAME]/
│   │       └── []/
│   │           └── [DOMAIN]/
│   │               ├── raw_output/      # Raw LLM outputs (from segmentation scripts)
│   │               ├── chunked/         # Processed event segments (from LLM_ES_extraction)
│   │               ├── metrics/         # Performance metrics
│   │               ├── logs/            # Processing logs
│   │               ├── error_files/     # Error files and error reports (from LLM_ES_extraction)
│   │               └── annotated/       # Annotated/cleaned outputs (if any post-processing)
│   └── event_extraction/
│       └── [MODEL_NAME]/
│           └── [PROMPT_TEMPLATE]/
│               └── [DOMAIN]/
│                   ├── raw_output/      # Raw LLM outputs (from extraction scripts)
│                   ├── annotated/       # Processed argument structures (from LLM_EE_extraction)
│                   ├── error/     # Error files and error reports (from LLM_EE_extraction)
│                   ├── metrics/         # Performance metrics
│                   ├── logs/            # Processing logs
│                   └── all_papers_annotated.json  # Aggregated annotation output
``` -->



### 2. Event Extraction

Extract arguments from each segmented abstract using different approaches (all use the same output directory structure):

**Standard Event Extraction:**
```bash
# For open-source models
python Event_extraction.py --domains ACL BIOINFO CSCW DH JMIR --model MODEL_NAME

# For OpenAI API-based models
python Event_extraction_openai.py --domains ACL BIOINFO CSCW DH JMIR --model MODEL_NAME --api-key YOUR_API_KEY
```

**Event Type-Based Extraction:**
```bash
# Predict event type and extract arguments
python Pred_Event_Type.py --domains ACL BIOINFO CSCW DH JMIR --model MODEL_NAME

# Extract arguments using gold event type
python True_Event_type.py --domains ACL BIOINFO CSCW DH JMIR --model MODEL_NAME
```

**Process raw argument extraction outputs into structured format:**
```bash
python LLM_EE_extraction.py --domains ACL BIOINFO CSCW DH JMIR --prompt-template [TEMPLATE] --model-name MODEL_NAME
```



## Prompt Templates

The `Prompts/` folder provides a variety of prompt templates for event segmentation and argument extraction. These can be selected in the config or via command-line arguments in the main scripts.

- **Zero-shot event segmentation:** `Zero-Shot_Event_Segmentation.txt`
- **Zero-shot argument extraction:** `Zero-Shot_Event_Extraction.txt`
- **One-shot argument extraction:** `One-Shot_Event_Extraction.txt`
- **Few-shot argument extraction:**
  - `Few-shot-2_Event_Extraction.txt`
  - `Few-shot-5_Event_Extraction.txt`
- **Event type prediction and extraction:**
  - `Zero-shot_Pred_Event_Type.txt` (predicts event type and extracts arguments)
  - `Zero-shot_True_Event_Type.txt` (uses gold event type for extraction)

**How to use:**
- The prompt template is set in each script via the `PROMPT_TEMPLATE_NAME` variable or a command-line argument.
- For example, in `Event_extraction.py`, set `PROMPT_TEMPLATE_NAME = "Fewshot-2"` to use the 2-shot prompt.
- See each script's config section for details.

**Prompt template summary:**
| File | Use case |
|------|----------|
| Zero-shot_Event_Segmentation.txt | Zero-shot segmentation |
| Zero-shot_Event_Extraction.txt | Zero-shot argument extraction |
| One-shot_Event_Extraction.txt | One-shot argument extraction |
| Few-shot-2_Event_Extraction.txt | 2-shot argument extraction |
| Few-shot-5_Event_Extraction.txt | 5-shot argument extraction |
| Zero-shot_Pred_Event_Type.txt | Predict event type, then extraction |
| Zero-shot_True_Event_Type.txt | Use gold event type for extraction |

## LLM Scripts Overview

The following scripts are available for running the LLM-based pipeline:

- `Event_segmentation.py` / `Event_segmentation_openai.py`: Run event segmentation (open-source or OpenAI models)
- `Event_extraction.py` / `Event_extraction_openai.py`: Run argument extraction (open-source or OpenAI models)
- `LLM_ES_extraction.py`: Post-process raw segmentation outputs into structured format
- `LLM_EE_extraction.py`: Post-process raw argument extraction outputs into structured format
- `Pred_Event_Type.py`: Predict event type and extract arguments (auto event type)
- `True_Event_type.py`: Extract arguments using gold event type

**Usage notes:**
- All scripts support domain selection and model selection via command-line arguments.
- Prompt template can be set in the config or via argument (see script for details).
- For OpenAI models, provide your API key as needed.

**Example commands:**
```bash
# Event segmentation (open-source)
python Event_segmentation.py --domains ACL BIOINFO CSCW DH JMIR --model MODEL_NAME --prompt-template Zero-Shot_Event_Segmentation

# Event segmentation (OpenAI)
python Event_segmentation_openai.py --domains ACL BIOINFO CSCW DH JMIR --model MODEL_NAME --api-key YOUR_API_KEY --prompt-template Zero-Shot_Event_Segmentation

# Argument extraction (open-source)
python Event_extraction.py --domains ACL BIOINFO CSCW DH JMIR --model MODEL_NAME --prompt-template Few-shot-2_Event_Extraction

# Argument extraction (OpenAI)
python Event_extraction_openai.py --domains ACL BIOINFO CSCW DH JMIR --model MODEL_NAME --api-key YOUR_API_KEY --prompt-template Few-shot-2_Event_Extraction

# Post-process segmentation output
python LLM_ES_extraction.py --domains ACL BIOINFO CSCW DH JMIR --prompt-template Zero-Shot_Event_Segmentation --model-name MODEL_NAME

# Post-process argument extraction output
python LLM_EE_extraction.py --domains ACL BIOINFO CSCW DH JMIR --prompt-template Few-shot-2_Event_Extraction --model-name MODEL_NAME
```

## Example Output

### Event Segmentation Output

```json
{
  "paper_code": "paper_123",
  "abstract": "Original abstract text...",
  "[Background]": "Segmented background text...",
  "[Method]": "Segmented method text...",
  "[Results]": "Segmented results text...",
  "[Implications]": "Segmented implications text..."
}
```

### Event Extraction Output

```json
{
  "paper_code": "paper_123",
  "abstract": "Original abstract text...",
  "events": [
    {
      "Text": "Segmented event text...",
      "Action": "introduces",
      "Arguments": {
        "Agent": ["We"],
        "Object": {
          "Primary Object": ["a new approach"],
          "Secondary Object": ["<NONE>"]
        },
        "Context": ["Previous approaches have limitations..."],
        "Purpose": ["To address the challenge of..."],
        "Method": ["by combining neural networks with..."],
        "Results": ["Our approach achieves state-of-the-art performance..."],
        "Analysis": ["This improvement demonstrates..."],
        "Challenge": ["<NONE>"],
        "Ethical": ["<NONE>"],
        "Implications": ["These findings suggest..."],
        "Contradictions": ["<NONE>"]
      }
    }
  ]
}
```


