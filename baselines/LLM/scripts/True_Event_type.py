"""
Raw output extractor that generates LLM outputs for abstract segments and 
saves the complete raw outputs without additional processing.
Now with tqdm progress tracking for better time estimation and enhanced memory management.
Enhanced to include event type information in the prompt and support OpenAI API.
"""

import re
import torch
import json
import os
import time
import logging
import shutil
import gc  # Added for explicit garbage collection
from datetime import datetime, timedelta
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm

# OpenAI imports (optional)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI library not available. Install with: pip install openai")

# Set GPU device explicitly
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0, 1 - change the gpu device based on your requirement

# Configuration
class Config:
    # Model settings
    #MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
    MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    #MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    #MODEL_NAME = "gpt-4o"  # OpenAI model (gpt-4o is the latest, not gpt-4.1)
    
    # Model type: "huggingface" or "openai"
    MODEL_TYPE = "huggingface"  # Change to "openai" for OpenAI models
    
    # OpenAI settings (only used when MODEL_TYPE = "openai")
    OPENAI_API_KEY="your_openai_api_key"  # Set this or use environment variable OPENAI_API_KEY
    OPENAI_MODEL = "gpt-4.1"  # Latest OpenAI model (gpt-4o, gpt-4-turbo, gpt-3.5-turbo)
    
    # model cache directory to store the model so that we dont need download it again and again
    MODEL_CACHE_DIR = "./model_cache"
    
    # Folder structure
    # BASE_DIR = "./SciEvent_data"
    # INPUT_DIR = f"{BASE_DIR}/raw"
    # OUTPUT_BASE_DIR = "./output/event_extraction"
    INPUT_DIR = "./SciEvent_data/raw/domain_specific_unannotated"
    OUTPUT_BASE_DIR = "./baselines/LLM/output/Event_Extraction"

    # Prompt template name
    PROMPT_TEMPLATE_NAME = "Zero-shot_True_Event_Type" # change the prompt template as per requirement 
    
    # Logging
    LOG_LEVEL = logging.INFO
    
    # Processing settings - our default setting is  MAX_NEW_TOKENS = 1200, TEMPERATURE = 0.3 and TOP_P = 0.9
    BATCH_SIZE = 1  # Number of papers to process in one batch
    MAX_NEW_TOKENS = 1200  # Limit new tokens to prevent over-generation
    TEMPERATURE = 0.3  # Set to 0.0 for deterministic output
    TOP_P = 0.9  # Set to 1.0 for deterministic output

# Zero-shot prompt template with event type information - UPDATED VERSION
PROMPT_TEMPLATE = """
You are an expert argument annotator. Given a part of the text and the event type from the scientific abstract (e.g., "Background", "Method", "Results", "Implication"), you need to identify the key trigger for the event (the main verb or action that signals an important research activity) and annotate the abstract with the corresponding argument components related to this trigger. Extractions should capture complete phrases around this key trigger and be organized in a single JSON format, containing only what is explicitly stated in the text without adding any interpretation.

### Event Type Definitions:
- [Background]: Problem, motivation, context, research gap, or objectives.
- [Method]: Techniques, experimental setups, frameworks, datasets.
- [Results]: Main findings, discoveries, statistics, or trends.
- [Implications]: Importance, impact, applications, or future work.

### {event_type} Event Abstract Segment to Analyze: ###
{abstract}

### Argument Components to Extract:

Main Action: What is the SINGLE most representative trigger (verb or verb phrase) in the segment? 

Agent: Who or what is performing this main action? 

Object:
- Primary Object: What is directly receiving or affected by the main action? 
- Secondary Object: What is a secondary entity also receiving the main action?

Context: What provides foundational or situational information of the event?

Purpose: What is the purpose or aim of the event?

Method: What techniques, tools, approaches, or frameworks are used in the event?

Results: What are the outcomes, observations or findings of the event?

Analysis: What are the interpretations or explanations of other arguments?

Challenge: What are the constraints or weaknesses of the event?

Ethical: What are the ethical concerns, justifications or implications of the event?

Implications: What is the broader significance or potential for future applications/research?

Contradictions: What are the disagreements with existing knowledge?

### Extraction Rules:

1. Extract complete phrases, not just single words.
2. Only extract elements that are explicitly present. Mark missing elements as ["<NONE>"].
3. Use the exact text from the abstract.
4. Break down sentences when different parts fit different arguments.
5. NEVER use the same span of text for multiple arguments - each piece of text must be assigned to exactly one argument type. However, multiple text spans can be part of the same argument (e.g., ["text span 1", "text span 2".....] can be used for a single argument type) if different parts of the text contribute to the same argument.
6. If text could fit multiple arguments, prioritize in this order: Results > Purpose > Method > Analysis > Implication > Challenge > Contradiction > Context > Ethical
7. Consider the event type when determining the most appropriate argument assignments.

### Output Format:
{
  "Main Action": "EXACT TEXT or <NONE>",
  "Agent": ["EXACT TEXT or <NONE>"],
   "Object": {
    "Primary Object": ["EXACT TEXT or <NONE>"],
    "Secondary Object": ["EXACT TEXT or <NONE>"]
  },
  "Context": ["EXACT TEXT or <NONE>"],
  "Purpose": ["EXACT TEXT or <NONE>"],
  "Method": ["EXACT TEXT or <NONE>"],
  "Results": ["EXACT TEXT or <NONE>"],
  "Analysis": ["EXACT TEXT or <NONE>"],
  "Challenge": ["EXACT TEXT or <NONE>"],
  "Ethical": ["EXACT TEXT or <NONE>"],
  "Implications": ["EXACT TEXT or <NONE>"],
  "Contradictions": ["EXACT TEXT or <NONE>"]
}

### IMPORTANT INSTRUCTIONS:
- You MUST return ONLY ONE JSON structure.
- NO explanation text, thinking, or commentary before or after the JSON.
- NEVER repeat the JSON structure.
- ALL fields must use arrays with ["<NONE>"] for missing arguments.
- Follow the EXACT format shown in the template.
- ONLY extract arguments that are explicitly present in the text. DO NOT hallucinate or add any information not found in the abstract.
- Use the provided event type to guide your analysis and ensure the extraction is appropriate for that type of event.

### Output (JSON only)

"""

class RawOutputExtractor:
    def __init__(self, config: Config, domain: str):
        """Initialize the raw output extractor with configuration and domain."""
        self.config = config
        self.domain = domain
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Create directory structure
        self._setup_directories()
        
        # Set up logging
        self._setup_logging()
        
        # Initialize OpenAI client if using OpenAI models
        if self.config.MODEL_TYPE == "openai":
            self._setup_openai()
        else:
            # Check if model needs to be redownloaded for HuggingFace models
            if self.should_redownload_model():
                self._clean_model_cache()
            
            # Load model and tokenizer for HuggingFace models
            self._load_model()
        
        # Initialize metrics
        self.metrics = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_processing_time": 0,
            "files_processed": 0,
            "papers_processed": 0,
            "events_processed": 0,
            "raw_outputs_generated": 0
        }
        
        # Key marker for identifying model output section
        self.answer_marker = "Your Answer:"
    
    def _setup_openai(self):
        """Setup OpenAI client for API-based models."""
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library is required for OpenAI models. Install with: pip install openai")
        
        # Get API key from config or environment variable
        api_key = self.config.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or config.OPENAI_API_KEY")
        
        self.openai_client = OpenAI(api_key=api_key)
        self.logger.info(f"OpenAI client initialized for model: {self.config.OPENAI_MODEL}")
        
        # For OpenAI, we don't need tokenizer/model loading
        self.model = None
        self.tokenizer = None
        
    def log_memory_stats(self, stage: str):
        """Log detailed memory statistics."""
        if not torch.cuda.is_available() or self.config.MODEL_TYPE == "openai":
            return
            
        allocated = torch.cuda.memory_allocated() / (1024 * 1024)
        max_allocated = torch.cuda.max_memory_allocated() / (1024 * 1024)
        reserved = torch.cuda.memory_reserved() / (1024 * 1024)
        
        self.logger.info(f"Memory at {stage}: Allocated: {allocated:.2f}MB, "
                       f"Max Allocated: {max_allocated:.2f}MB, "
                       f"Reserved: {reserved:.2f}MB")
    
    def should_redownload_model(self) -> bool:
        """Check if the model cache exists and is valid using the correct HF path format."""
        # Skip for OpenAI models
        if self.config.MODEL_TYPE == "openai":
            return False
            
        # The actual HuggingFace cache structure uses this format:
        model_cache_path = Path(self.config.MODEL_CACHE_DIR) / f"models--{self.config.MODEL_NAME.replace('/', '--')}"
        
        if not model_cache_path.exists():
            self.logger.info(f"Model cache not found at {model_cache_path}, will download model")
            return True
        
        # Check for essential files
        config_file = model_cache_path / "config.json"
        if not config_file.exists():
            self.logger.info(f"Model cache appears incomplete, will redownload")
            return True
        
        return False
    
    def _clean_model_cache(self) -> None:
        """Remove the existing model cache if it exists."""
        model_cache_path = Path(self.config.MODEL_CACHE_DIR) / self.config.MODEL_NAME.replace('/', '--')
        if model_cache_path.exists():
            self.logger.info(f"Removing existing model cache at {model_cache_path}")
            try:
                shutil.rmtree(model_cache_path)
                self.logger.info(f"Successfully removed model cache")
            except Exception as e:
                self.logger.error(f"Failed to remove model cache: {e}")
    
    def _setup_directories(self):
        """Create the directory structure for outputs."""
        # Create base output directory structure
        model_name = self.config.OPENAI_MODEL if self.config.MODEL_TYPE == "openai" else self.config.MODEL_NAME.split('/')[-1]
        self.output_dir = Path(f"{self.config.OUTPUT_BASE_DIR}/{model_name}/{self.config.PROMPT_TEMPLATE_NAME}/{self.domain}")
        
        # Create raw output and logs directories - using same structure as original
        self.raw_output_dir = self.output_dir / "raw_output"
        self.logs_dir = self.output_dir / "logs"
        self.metrics_dir = self.output_dir / "generated_file_metric_logs"
        
        # Create model cache directory if it doesn't exist (only for HF models)
        if self.config.MODEL_TYPE != "openai":
            os.makedirs(self.config.MODEL_CACHE_DIR, exist_ok=True)
        
        # Create output directories
        os.makedirs(self.raw_output_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        print(f"Created directory structure at {self.output_dir}")
    
    def _setup_logging(self):
        """Set up logging configuration."""
        # Create logger
        self.logger = logging.getLogger(f"RawOutputExtractor_{self.domain}")
        self.logger.setLevel(self.config.LOG_LEVEL)
        
        # Clear any existing handlers
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.config.LOG_LEVEL)
        
        # Create file handler
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.logs_dir / f"{self.domain}_{timestamp}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(self.config.LOG_LEVEL)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        
        self.logger.info(f"Logger initialized. Logging to {log_file}")
        self.logger.info(f"Created directory structure at {self.output_dir}")
    
    def _load_model(self):
        """Load the model and tokenizer from HuggingFace, using cache."""
        if self.config.MODEL_TYPE == "openai":
            self.logger.info("Using OpenAI API - no local model loading required")
            return
            
        self.logger.info(f"Loading model {self.config.MODEL_NAME} from HuggingFace (using cache dir: {self.config.MODEL_CACHE_DIR})")
        
        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.MODEL_NAME, 
                cache_dir=self.config.MODEL_CACHE_DIR,
                use_fast=True
            )
            self.logger.info("Tokenizer loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading tokenizer: {e}")
            raise
        
        # Load configuration with rope_scaling fix
        try:
            config = AutoConfig.from_pretrained(
                self.config.MODEL_NAME,
                cache_dir=self.config.MODEL_CACHE_DIR
            )
            
            # Remove rope_scaling to prevent issues
            if hasattr(config, 'rope_scaling'):
                delattr(config, 'rope_scaling')
                self.logger.info("Removed rope_scaling attribute from config")
            
            # Load model with modified config and float16 precision for speed
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.MODEL_NAME, 
                config=config,
                cache_dir=self.config.MODEL_CACHE_DIR,
                torch_dtype=torch.float16,
                device_map="auto",
                ignore_mismatched_sizes=True
            )
            self.logger.info(f"Model loaded successfully. Using device: {self.device}")
            
        except Exception as e:
            self.logger.error(f"Error loading model with modified config: {e}")
            
            # Fallback approach using manual config editing
            try:
                import json
                import tempfile
                from huggingface_hub import hf_hub_download
                
                # Download config file directly
                config_path = hf_hub_download(
                    repo_id=self.config.MODEL_NAME,
                    filename="config.json",
                    cache_dir=self.config.MODEL_CACHE_DIR
                )
                
                # Read and modify config
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                
                # Remove problematic field
                if 'rope_scaling' in config_dict:
                    del config_dict['rope_scaling']
                    self.logger.info("Removed rope_scaling from config_dict")
                
                # Create temporary file with modified config
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_f:
                    json.dump(config_dict, temp_f)
                    temp_config_path = temp_f.name
                
                # Use modified config
                config = AutoConfig.from_pretrained(temp_config_path)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.MODEL_NAME,
                    config=config,
                    cache_dir=self.config.MODEL_CACHE_DIR,
                    torch_dtype=torch.float16,
                    device_map="auto", 
                    ignore_mismatched_sizes=True
                )
                self.logger.info(f"Model loaded successfully with fallback approach.")
                
                # Clean up
                os.unlink(temp_config_path)
                
            except Exception as e2:
                self.logger.error(f"All attempts to load model failed: {e2}")
                raise RuntimeError(f"Could not load model after multiple attempts.")
    
    def _unload_model(self):
        """Unload model and free GPU memory."""
        if self.config.MODEL_TYPE == "openai":
            return  # No model to unload for OpenAI
            
        self.logger.info("Unloading model and freeing GPU memory")
        
        # Delete model and tokenizer references
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            self.logger.info("CUDA cache cleared")
    
    def generate_model_output(self, abstract: str, event_type: str = "Unknown") -> Tuple[str, Dict[str, int]]:
        """
        Generate raw model output with improved stopping logic to prevent over-generation.
        Returns only the LLM generated response (not the prompt) and metrics.
        Now supports both HuggingFace and OpenAI models.
        """
        start_time = time.time()
        
        # Prepare prompt with event type
        prompt = PROMPT_TEMPLATE.replace("{abstract}", abstract).replace("{event_type}", event_type)
        
        if self.config.MODEL_TYPE == "openai":
            return self._generate_openai_output(prompt, start_time)
        else:
            return self._generate_huggingface_output(prompt, start_time)
    
    def _generate_openai_output(self, prompt: str, start_time: float) -> Tuple[str, Dict[str, int]]:
        """Generate output using OpenAI API."""
        try:
            response = self.openai_client.chat.completions.create(
                model=self.config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert argument annotator. Follow the instructions exactly and return only the requested JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.TEMPERATURE,
                max_tokens=self.config.MAX_NEW_TOKENS,
                top_p=self.config.TOP_P
            )
            
            model_output = response.choices[0].message.content
            
            # Get token usage if available
            input_tokens = response.usage.prompt_tokens if response.usage else len(prompt) // 4
            output_tokens = response.usage.completion_tokens if response.usage else len(model_output) // 4
            
        except Exception as e:
            self.logger.error(f"Error during OpenAI generation: {str(e)}")
            model_output = f"Error during OpenAI generation: {str(e)}"
            input_tokens = len(prompt) // 4
            output_tokens = len(model_output) // 4
        
        processing_time = time.time() - start_time
        
        metrics = {
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "processing_time": processing_time
        }
        
        return model_output, metrics
    
    def _generate_huggingface_output(self, prompt: str, start_time: float) -> Tuple[str, Dict[str, int]]:
        """Generate output using HuggingFace model."""
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
            input_token_count = len(inputs.input_ids[0])
            
            # Generate response with improved stopping logic
            try:
                with torch.no_grad():
                    # Use max_new_tokens instead of max_length for better control
                    output = self.model.generate(
                        **inputs,
                        max_new_tokens=self.config.MAX_NEW_TOKENS,
                        temperature=self.config.TEMPERATURE,
                        top_p=self.config.TOP_P,
                        do_sample=True,               
                        num_return_sequences=1,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,  # Explicitly set EOS token
                        early_stopping=True,  # Enable early stopping
                        repetition_penalty=1.2  # Add repetition penalty to discourage repeating the JSON
                    )
                    
                    # Get only the generated tokens (not including the prompt)
                    # Ensure proper detachment with clone
                    generated_tokens = output[0][len(inputs.input_ids[0]):].clone().detach()
                    
                    # Decode only the generated part (excluding the prompt)
                    model_output = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    output_token_count = len(generated_tokens)
                    
                    # Post-process to truncate at reasonable ending point
                    # Find the last closing brace that might signal the end of the JSON
                    last_closing_brace = model_output.rfind("}")
                    
                    # Look for indicators of over-generation after the last closing brace
                    if last_closing_brace > 0:
                        # Truncate at the last closing brace + 1
                        model_output = model_output[:last_closing_brace+1]
                        
                        # Check if we need to fix unclosed JSON
                        open_braces = model_output.count('{')
                        close_braces = model_output.count('}')
                        
                        # If there are more open braces than close braces, add missing close braces
                        if open_braces > close_braces:
                            model_output += '}' * (open_braces - close_braces)
                    
                    # Free memory from generated output - explicit deletion
                    del output
                    del generated_tokens
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()  # Ensure CUDA operations complete
                    
            except RuntimeError as e:
                # Handle CUDA out of memory
                if "CUDA out of memory" in str(e):
                    self.logger.warning("CUDA out of memory during generation - trying with reduced settings")
                    
                    # Clear cache and retry with reduced settings
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    
                    with torch.no_grad():
                        output = self.model.generate(
                            **inputs,
                            max_new_tokens=400, 
                            temperature=0.0,     
                            do_sample=False,    
                            num_beams=1,        
                            repetition_penalty=1.0
                        )
                    
                    # Get only the generated tokens (not including the prompt)
                    generated_tokens = output[0][len(inputs.input_ids[0]):].clone().detach()
                    
                    # Decode only the generated part (excluding the prompt)
                    model_output = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    output_token_count = len(generated_tokens)
                    
                    # Free memory - explicit deletion
                    del output
                    del generated_tokens
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                else:
                    raise
            
            # Clear inputs from CUDA memory - explicit deletion
            del inputs
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        except Exception as e:
            self.logger.error(f"Error during generation: {str(e)}")
            # Create default output for error cases
            model_output = f"Error during generation: {str(e)}"
            output_token_count = len(prompt) // 4  # Rough estimate
            input_token_count = len(prompt) // 4   # Rough estimate
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Metrics 
        metrics = {
            "input_tokens": int(input_token_count) if 'input_token_count' in locals() else 0,
            "output_tokens": int(output_token_count) if 'output_token_count' in locals() else 0,
            "processing_time": processing_time
        }
        
        # Return only the model output with metrics
        return model_output, metrics
    
    def extract_event_type_from_event(self, event: Dict) -> str:
        """
        Extract event type from the event dictionary and map to standardized types.
        
        Based on the actual input file structure, the event type is stored as a key in the event dictionary.
        We map all event types to one of four standardized types:
        - Background
        - Method  
        - Results
        - Implication
        
        Actual event structure from the input file:
        {
            "Background/Introduction": "Exploring summarization models robustness against perturbations",
            "Text": "A robust summarization system should be able to capture...",
            "Main Action": "explore",
            "Arguments": {
                "Agent": [...],
                "Object": {...},
                ...
            }
        }
        
        We extract the event type key and map it to our standardized types.
        """
        # Standard keys that are not event types
        standard_keys = {"Text", "Main Action", "Arguments", "Summary", "text", "main action", "arguments", "summary"}
        
        # Find the event type key
        found_event_type = "Unknown"
        try:
            for key in event.keys():
                if key not in standard_keys:
                    found_event_type = key
                    self.logger.debug(f"Found raw event type: {found_event_type}")
                    break
        except Exception as e:
            self.logger.error(f"Error extracting event type: {e}")
            return "Background"  # Default fallback
        
        # Map to standardized event types
        event_type_lower = found_event_type.lower()
        
        if ("background" in event_type_lower or 
            "introduction" in event_type_lower or
            "intro" in event_type_lower or
            "motivation" in event_type_lower):
            return "Background"
            
        elif ("method" in event_type_lower or 
              "approach" in event_type_lower or
              "technique" in event_type_lower or
              "methodology" in event_type_lower or
              "procedure" in event_type_lower):
            return "Method"
            
        elif ("result" in event_type_lower or 
              "finding" in event_type_lower or
              "outcome" in event_type_lower or
              "experiment" in event_type_lower or
              "evaluation" in event_type_lower):
            return "Results"
            
        elif ("conclusion" in event_type_lower or 
              "implication" in event_type_lower or
              "impact" in event_type_lower or
              "significance" in event_type_lower or
              "future" in event_type_lower or
              "discussion" in event_type_lower):
            return "Implication"
        
        # Default fallback - try to make an educated guess
        else:
            self.logger.warning(f"Unknown event type '{found_event_type}', defaulting to Background")
            return "Background"
    
    def process_file(self, input_file: str) -> Dict:
        """Process a single JSON file containing paper abstracts and return file metrics."""
        file_start_time = time.time()
        file_basename = os.path.basename(input_file)
        
        self.logger.info(f"Processing file: {file_basename}")
        self.log_memory_stats(f"Before processing file {file_basename}")
        
        # Load the JSON file
        try:
            with open(input_file, "r", encoding="utf-8") as f:
                file_content = f.read()
                
            # Handle different JSON formats
            papers = []
            
            # Try parsing as array
            if file_content.strip().startswith('['):
                try:
                    papers = json.loads(file_content)
                    self.logger.info(f"Found {len(papers)} papers in array format")
                except json.JSONDecodeError:
                    self.logger.error(f"Failed to parse array JSON")
                    return {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "processing_time": 0,
                        "papers_count": 0,
                        "events_count": 0,
                        "raw_outputs_generated": 0,
                        "total_time": time.time() - file_start_time
                    }
            else:
                # Try parsing as a single object
                try:
                    paper = json.loads(file_content)
                    papers = [paper]
                    self.logger.info(f"Found 1 paper in single object format")
                except json.JSONDecodeError:
                    # Try parsing as newline-delimited JSON
                    papers = []
                    for line in file_content.strip().split('\n'):
                        if line.strip():
                            try:
                                paper = json.loads(line)
                                papers.append(paper)
                            except json.JSONDecodeError:
                                pass
                    self.logger.info(f"Found {len(papers)} papers in newline-delimited format")
        except Exception as e:
            self.logger.error(f"Error loading file {file_basename}: {e}")
            return {
                "input_tokens": 0,
                "output_tokens": 0,
                "processing_time": 0,
                "papers_count": 0,
                "events_count": 0,
                "raw_outputs_generated": 0,
                "total_time": time.time() - file_start_time
            }
        
        # Initialize file metrics
        file_metrics = {
            "input_tokens": 0,
            "output_tokens": 0,
            "processing_time": 0,
            "papers_count": len(papers),
            "events_count": 0,
            "raw_outputs_generated": 0
        }
        
        # Create a tqdm progress bar for papers in this file
        with tqdm(total=len(papers), desc=f"File: {file_basename}", unit="paper") as pbar:
            # Process each paper
            for i, paper in enumerate(papers):
                try:
                    paper_code = paper.get("paper_code")
                    if not paper_code:
                        self.logger.warning(f"Paper {i} has no paper_code, skipping")
                        pbar.update(1)
                        continue
                        
                    abstract = paper.get("abstract", "")
                    
                    # Include paper progress in the progress bar description
                    pbar.set_description(f"File: {file_basename} | Paper: {paper_code} ({i+1}/{len(papers)})")
                    
                    # Extract events/sections from the original data if available
                    events = paper.get("events", [])
                    
                    if not events:
                        # If no events, process the entire abstract
                        self.logger.info(f"No events found, processing entire abstract")
                        file_metrics["events_count"] += 1
                        self.metrics["events_processed"] += 1
                        
                        try:
                            # Generate complete raw output with Unknown event type
                            raw_output, metrics = self.generate_model_output(abstract, "Unknown")
                            
                            # Save the complete raw output exactly as received
                            raw_txt_file_path = self.raw_output_dir / f"{paper_code}_full.txt"
                            with open(raw_txt_file_path, "w", encoding="utf-8") as raw_txt_file:
                                raw_txt_file.write(raw_output)
                                
                            # Increment raw outputs counter
                            file_metrics["raw_outputs_generated"] += 1
                            
                            # Update metrics
                            file_metrics["input_tokens"] += metrics["input_tokens"]
                            file_metrics["output_tokens"] += metrics["output_tokens"]
                            file_metrics["processing_time"] += metrics["processing_time"]
                        except Exception as e:
                            self.logger.error(f"Error processing abstract for paper {paper_code}: {str(e)}")
                            import traceback
                            self.logger.error(f"Traceback: {traceback.format_exc()}")
                    
                    else:
                        # Process each event separately with nested progress bar
                        file_metrics["events_count"] += len(events)
                        self.metrics["events_processed"] += len(events)
                        
                        # Create a nested tqdm progress bar for events within this paper
                        with tqdm(total=len(events), desc=f"  Events in {paper_code}", unit="event", leave=False) as event_pbar:
                            for event_index, event in enumerate(events):
                                event_text = event.get("Text", "")  # This is the actual text content to analyze
                                
                                # ENHANCED EVENT TYPE EXTRACTION - This is where we get the event type from the input file
                                event_type = self.extract_event_type_from_event(event)
                                
                                # Update event progress bar description with event type
                                event_pbar.set_description(f"  Events in {paper_code} | Type: {event_type} ({event_index+1}/{len(events)})")
                                
                                if not event_text:
                                    self.logger.warning(f"Empty event text for paper {paper_code}, event {event_index}")
                                    event_pbar.update(1)
                                    continue
                                
                                try:
                                    # Generate raw model output with event type and save as-is
                                    raw_output, metrics = self.generate_model_output(event_text, event_type)
                                    
                                    # Keep original file naming convention
                                    raw_txt_file_path = self.raw_output_dir / f"{paper_code}_event_{event_index}.txt"
                                    
                                    with open(raw_txt_file_path, "w", encoding="utf-8") as raw_txt_file:
                                        # Save only the raw model output, exactly as received
                                        raw_txt_file.write(raw_output)
                                        
                                    # Increment raw outputs counter
                                    file_metrics["raw_outputs_generated"] += 1
                                    
                                    # Update metrics
                                    file_metrics["input_tokens"] += metrics["input_tokens"]
                                    file_metrics["output_tokens"] += metrics["output_tokens"]
                                    file_metrics["processing_time"] += metrics["processing_time"]
                                
                                except Exception as e:
                                    self.logger.error(f"Error processing event {event_index} for paper {paper_code}: {str(e)}")
                                    import traceback
                                    self.logger.error(f"Traceback: {traceback.format_exc()}")
                                
                                # Clear CUDA cache after each event to prevent memory buildup (only for HF models)
                                if self.config.MODEL_TYPE != "openai":
                                    torch.cuda.empty_cache()
                                
                                # Update event progress bar
                                event_pbar.update(1)
                    
                    self.metrics["papers_processed"] += 1
                    
                    # Update paper progress bar
                    pbar.update(1)
                    
                    # Calculate and display estimated time remaining for file
                    if i > 0:  # Need at least one paper to estimate
                        avg_paper_time = file_metrics["processing_time"] / (i + 1)
                        remaining_papers = len(papers) - (i + 1)
                        est_remaining_time = avg_paper_time * remaining_papers
                        est_completion_time = datetime.now() + timedelta(seconds=est_remaining_time)
                        
                        # Update progress bar postfix with time estimates
                        pbar.set_postfix({
                            "Avg paper": f"{avg_paper_time:.2f}s", 
                            "ETA": est_completion_time.strftime("%H:%M:%S"),
                            "Remaining": str(timedelta(seconds=int(est_remaining_time)))
                        })
                
                except Exception as e:
                    self.logger.error(f"Error processing paper {paper.get('paper_code', 'unknown')}: {str(e)}")
                    import traceback
                    self.logger.error(f"Traceback: {traceback.format_exc()}")
                    pbar.update(1)
                    continue
                
                # Performs cleanup after each paper to prevent memory accumulation (only for HF models)
                if self.config.MODEL_TYPE != "openai":
                    gc.collect()
                    torch.cuda.empty_cache()
        
        # Save file-specific metrics
        file_metrics["total_time"] = time.time() - file_start_time
        if file_metrics["papers_count"] > 0:
            file_metrics["average_paper_time"] = file_metrics["processing_time"] / file_metrics["papers_count"]
        else:
            file_metrics["average_paper_time"] = 0
        
        metrics_file = self.metrics_dir / f"{os.path.basename(input_file).replace('.json', '')}_metrics.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(file_metrics, f, indent=2)
        
        self.logger.info(f"Completed processing file: {file_basename}")
        self.logger.info(f"File metrics: {file_metrics}")
        
        # Update global metrics
        self.metrics["files_processed"] += 1
        self.metrics["total_input_tokens"] += file_metrics["input_tokens"]
        self.metrics["total_output_tokens"] += file_metrics["output_tokens"]
        self.metrics["total_processing_time"] += file_metrics["processing_time"]
        self.metrics["raw_outputs_generated"] += file_metrics["raw_outputs_generated"]
        
        # Final memory cleanup after file processing (only for HF models)
        if self.config.MODEL_TYPE != "openai":
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Ensure all CUDA operations are completed
        
        self.log_memory_stats(f"After processing file {file_basename}")
        
        return file_metrics

    def process_domain(self) -> None:
        """Process all files in the domain directory with tqdm for tracking overall progress."""
        domain_start_time = time.time()
        
        # Clear GPU cache at the beginning of a new domain (only for HF models)
        if self.config.MODEL_TYPE != "openai":
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.synchronize()
        
        self.log_memory_stats(f"Starting domain {self.domain}")
        
        # Get input directory for the domain
        input_dir = f"{self.config.INPUT_DIR}/{self.domain}"
        
        if not os.path.exists(input_dir):
            self.logger.error(f"Input directory not found: {input_dir}")
            return
        
        # Get all JSON files in the input directory
        json_files = [f for f in os.listdir(input_dir) if f.endswith(".json")]
        
        if not json_files:
            self.logger.warning(f"No JSON files found in {input_dir}")
            return
        
        self.logger.info(f"Found {len(json_files)} JSON files in {input_dir}")
        
        # Initialize file metrics history to calculate overall statistics
        file_metrics_history = []
        
        # Process each file with tqdm for progress tracking
        with tqdm(total=len(json_files), desc=f"Domain: {self.domain}", unit="file") as domain_pbar:
            for file_idx, file in enumerate(json_files):
                input_file = os.path.join(input_dir, file)
                
                # Process file and collect metrics
                file_metrics = self.process_file(input_file)
                file_metrics_history.append(file_metrics)
                
                # Update domain progress bar
                domain_pbar.update(1)
                
                # Calculates and shows overall progress statistics
                if file_idx > 0:
                    # Calculates average processing time per file
                    avg_file_time = sum(m["total_time"] for m in file_metrics_history) / len(file_metrics_history)
                    
                    # Estimates remaining time
                    remaining_files = len(json_files) - (file_idx + 1)
                    est_remaining_time = avg_file_time * remaining_files
                    est_completion_time = datetime.now() + timedelta(seconds=est_remaining_time)
                    
                    # Updates progress bar with overall statistics
                    domain_pbar.set_postfix({
                        "Avg file": f"{avg_file_time:.1f}s",
                        "ETA": est_completion_time.strftime("%H:%M:%S"),
                        "Remaining": str(timedelta(seconds=int(est_remaining_time)))
                    })
                    
                # Cleanup after each file (only for HF models)
                if self.config.MODEL_TYPE != "openai":
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
        
        # Calculate and save overall metrics
        self.metrics["total_time"] = time.time() - domain_start_time
        
        # Avoid division by zero
        if self.metrics["papers_processed"] > 0:
            self.metrics["average_paper_time"] = self.metrics["total_processing_time"] / self.metrics["papers_processed"]
            self.metrics["average_input_tokens"] = self.metrics["total_input_tokens"] / self.metrics["papers_processed"]
            self.metrics["average_output_tokens"] = self.metrics["total_output_tokens"] / self.metrics["papers_processed"]
        else:
            self.metrics["average_paper_time"] = 0
            self.metrics["average_input_tokens"] = 0
            self.metrics["average_output_tokens"] = 0
        
        # Save overall metrics
        overall_metrics_file = self.metrics_dir / "overall_metrics.json"
        with open(overall_metrics_file, "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, indent=2)
        
        self.logger.info(f"Completed processing domain: {self.domain}")
        self.logger.info(f"Overall metrics: {self.metrics}")
        
        # Final cleanup at the end of domain processing (only for HF models)
        if self.config.MODEL_TYPE != "openai":
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Reset max memory allocated to give accurate readings for next domain
            torch.cuda.reset_peak_memory_stats()
        
        self.log_memory_stats(f"Completed domain {self.domain}")
        
        # Print final summary to console
        total_papers = self.metrics["papers_processed"]
        total_events = self.metrics["events_processed"] 
        total_outputs = self.metrics["raw_outputs_generated"]
        total_time = str(timedelta(seconds=int(self.metrics["total_time"])))
        
        print("\n" + "="*80)
        print(f"PROCESSING SUMMARY FOR DOMAIN: {self.domain}")
        print("="*80)
        print(f"Total files processed:    {self.metrics['files_processed']}")
        print(f"Total papers processed:   {total_papers}")
        print(f"Total events processed:   {total_events}")
        print(f"Total outputs generated:  {total_outputs}")
        print(f"Total processing time:    {total_time}")
        if total_papers > 0:
            print(f"Average time per paper:   {self.metrics['average_paper_time']:.2f}s")
        print("="*80)

        # Unload model and free memory before exiting
        self._unload_model()
        
def main():
    """Main function to run the raw output extractor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process abstracts using LLM and save raw outputs")
    parser.add_argument("--domains", type=str, required=True, nargs='+', 
                        help="Domains to process (e.g., ACL BIOINFO CSCW DH JMIR)")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", 
                        help="HuggingFace model name or OpenAI model name")
    parser.add_argument("--model-type", type=str, choices=["huggingface", "openai"], 
                        default="huggingface", help="Type of model to use")
    parser.add_argument("--prompt", type=str, default="Zeroshot-EventType", 
                        help="Name of the prompt template")
    parser.add_argument("--clean-cache", action="store_true", 
                        help="Force redownload the model (HuggingFace only)")
    parser.add_argument("--openai-api-key", type=str, 
                        help="OpenAI API key (can also use OPENAI_API_KEY env variable)")
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    config = Config()
    config.MODEL_NAME = args.model
    config.MODEL_TYPE = args.model_type
    config.PROMPT_TEMPLATE_NAME = args.prompt
    config.CLEAN_CACHE = args.clean_cache
    
    # Set OpenAI model and API key if using OpenAI
    if config.MODEL_TYPE == "openai":
        config.OPENAI_MODEL = args.model
        if args.openai_api_key:
            config.OPENAI_API_KEY = args.openai_api_key
    
    # Print banner
    print("\n" + "="*80)
    print(f"RAW OUTPUT EXTRACTOR WITH EVENT TYPE")
    print("="*80)
    print(f"Domains:           {', '.join(args.domains)}")
    print(f"Model Type:        {config.MODEL_TYPE}")
    if config.MODEL_TYPE == "openai":
        print(f"OpenAI Model:      {config.OPENAI_MODEL}")
    else:
        print(f"HF Model:          {config.MODEL_NAME}")
    print(f"Prompt Template:   {config.PROMPT_TEMPLATE_NAME}")
    if config.MODEL_TYPE == "huggingface":
        print(f"Clean Cache:       {config.CLEAN_CACHE}")
    print("="*80 + "\n")
    
    # Process each domain sequentially
    for i, domain in enumerate(args.domains):
        print(f"\n{'='*80}")
        print(f"PROCESSING DOMAIN {i+1}/{len(args.domains)}: {domain}")
        print(f"{'='*80}\n")
        
        # Initialize and run the raw output extractor for this domain
        extractor = RawOutputExtractor(config, domain)
        extractor.process_domain()
        
        print(f"Completed processing domain: {domain}")
        
        # Force cleanup between domains - explicitly delete extractor
        del extractor
        gc.collect()
        if torch.cuda.is_available() and config.MODEL_TYPE != "openai":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Reset peak memory stats between domains
            torch.cuda.reset_peak_memory_stats()
            
            # Log CUDA memory status after domain completion
            allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            max_allocated = torch.cuda.max_memory_allocated() / (1024 * 1024)
            reserved = torch.cuda.memory_reserved() / (1024 * 1024)
            
            print(f"CUDA Memory after domain {domain}:")
            print(f"  Allocated: {allocated:.2f}MB")
            print(f"  Max Allocated: {max_allocated:.2f}MB")
            print(f"  Reserved: {reserved:.2f}MB")
        
        # Add a longer pause between domains
        if i < len(args.domains) - 1:
            print("Waiting 10 seconds before starting next domain to fully release resources...")
            time.sleep(10)

if __name__ == "__main__":
    # Run the main function
    main()