import torch
import json
import os
import time
import logging
import gc
from datetime import datetime, timedelta
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, StoppingCriteria, StoppingCriteriaList
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm

# Set GPU device explicitly
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 1

# Configuration
class Config:
    # Model settings - add you models
    #MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
    MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    #MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    MODEL_CACHE_DIR = "./model_cache"

    # Folder structure
    # BASE_DIR = "./SciEvent_data/raw"
    # INPUT_DIR = f"{BASE_DIR}/domain_specific_unannotated"
    # OUTPUT_BASE_DIR = "./baselines/LLM/output/Event_Segmentation"
    INPUT_DIR = "./SciEvent_data/raw/domain_specific_unannotated"
    OUTPUT_BASE_DIR = "./baselines/LLM/output/Event_Segmentation"
    
    # Prompt template name
    PROMPT_TEMPLATE_NAME = "Zero-Shot_Event_Segmentation"
    
    # Logging
    LOG_LEVEL = logging.INFO
    
    # Processing settings
    MAX_NEW_TOKENS = 1200
    #TEMPERATURE = 0.3
    #TOP_P = 0.9



# Chunking prompt template
CHUNK_PROMPT = """Extract the following abstract into four sections: Background, Method, Results, and Implications.
### **Rules for Extraction** ###
- **Only extract explicitly present sections. If a section is NOT clearly stated in the abstract, mark it as `<NONE>` and leave it completely empty.**
- **Do not infer, generate, or assume information. Only use exact text spans from the abstract.**
- **Do not break sentences. Assign them to the most relevant section without splitting.**
- **Sections must be connected** in the original abstract. Do not pick non-continuous sentences.
### **Follow These Steps Before Extracting Sections** ###
1. **Clarify** your understanding of the abstract. Read the full text before deciding section boundaries.
2. **Identify preliminary spans** for Background, Method, Results, and Implications based on explicit content.
3. **Critically assess** the extracted spans:
- Ensure each section consists of **continuous** sentences.
- Check that no sentence is assigned to multiple sections.
- If a section does not exist in the abstract, confirm it is marked as `<NONE>`.
4. **Confirm the final extraction** by carefully verifying your choices.
5. **Evaluate your confidence** (0-100%) in the correctness of extracted spans.
### **Final Instruction** ###
**Do not include your thought process in the output. Only print the extracted sections as formatted below.**
Ensure your answer strictly follows this structure:
[Background]: <EXACT TEXT or `<NONE>`>
[Method]: <EXACT TEXT or `<NONE>`>
[Results]: <EXACT TEXT or `<NONE>`>
[Implications]: <EXACT TEXT or `<NONE>`>
#### **Abstract to Extract:**
{abstract}
Your Answer:
""" 





class StopOnSubsequence(StoppingCriteria):
    def __init__(self, tokenizer, stop_string):
        super().__init__()
        self.stop_token_ids = tokenizer.encode(stop_string, add_special_tokens=False)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        return self.stop_token_ids == input_ids[0][-len(self.stop_token_ids):].tolist()


class RawChunkOutputExtractor:
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
        
        # Check if model needs to be redownloaded
        if self.should_redownload_model():
            self._clean_model_cache()
        
        # Load model and tokenizer
        self._load_model()
        
        # Initialize metrics
        self.metrics = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_processing_time": 0,
            "files_processed": 0,
            "papers_processed": 0,
            "raw_outputs_generated": 0
        }
    
    def should_redownload_model(self) -> bool:
        """Check if the model cache exists and is valid."""
        model_cache_path = Path(self.config.MODEL_CACHE_DIR) / self.config.MODEL_NAME.replace('/', '--')
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
        import shutil
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
        self.output_dir = Path(f"{self.config.OUTPUT_BASE_DIR}/{self.config.MODEL_NAME.split('/')[-1]}/{self.config.PROMPT_TEMPLATE_NAME}/{self.domain}")
        
        # Create raw output directories
        self.raw_output_dir = self.output_dir / "raw_output"
        self.logs_dir = self.output_dir / "logs"
        self.metrics_dir = self.output_dir / "metrics"
        
        # Create model cache directory if it doesn't exist
        os.makedirs(self.config.MODEL_CACHE_DIR, exist_ok=True)
        
        # Create output directories
        os.makedirs(self.raw_output_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        print(f"Created directory structure at {self.output_dir}")
    
    def _setup_logging(self):
        """Set up logging configuration."""
        # Create logger
        self.logger = logging.getLogger(f"RawChunkOutputExtractor_{self.domain}")
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
    
    def _load_model(self):
        """Load the model and tokenizer from HuggingFace, using cache."""
        self.logger.info(f"Loading model {self.config.MODEL_NAME} from HuggingFace (using cache dir: {self.config.MODEL_CACHE_DIR})")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.MODEL_NAME, 
                cache_dir=self.config.MODEL_CACHE_DIR,
                use_fast=True
            )
            
            # Load configuration with rope_scaling fix
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
            self.logger.error(f"Error loading model: {e}")
            
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
        self.logger.info("Unloading model and freeing GPU memory")
        
        # Delete model and tokenizer
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.logger.info("CUDA cache cleared")
    
    def generate_model_output(self, abstract: str) -> Tuple[str, Dict[str, int]]:
        """Generate raw model output without additional processing."""
        start_time = time.time()
        
        try:
            # Prepare prompt
            prompt = CHUNK_PROMPT.format(abstract=abstract)
            
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
            input_token_count = len(inputs.input_ids[0])
            
            # Generate response
            try:
                with torch.no_grad():
                    output = self.model.generate(
                        **inputs,
                        max_new_tokens=self.config.MAX_NEW_TOKENS,
                        #temperature=0.3,          # <== Changed
                        #top_p=,                # <== Changed
                        do_sample=False,          # <== Changed
                        num_return_sequences=1,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        repetition_penalty=1.2,
                        stopping_criteria=StoppingCriteriaList([StopOnSubsequence(self.tokenizer, "<|END_OF_CHUNK|>")])
                    )
                    
                    # Get only the generated tokens (not including the prompt)
                    generated_tokens = output[0][len(inputs.input_ids[0]):]
                    
                    # Decode only the generated part
                    model_output = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    output_token_count = len(generated_tokens)
                    
            except RuntimeError as e:
                # Handle CUDA out of memory errors with CPU fallback
                if "CUDA out of memory" in str(e):
                    self.logger.warning("CUDA out of memory, using CPU fallback")
                    
                    # Move to CPU
                    self.model = self.model.to("cpu")
                    inputs = {k: v.to("cpu") for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        output = self.model.generate(
                            **inputs,
                            max_new_tokens=self.config.MAX_NEW_TOKENS,
                            temperature=0.0,
                            top_p=1.0,
                            do_sample=False,
                            num_return_sequences=1,
                            pad_token_id=self.tokenizer.eos_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                            early_stopping=True,
                            repetition_penalty=1.2
                        )
                    
                    # Get only the generated tokens
                    generated_tokens = output[0][len(inputs.input_ids[0]):]
                    model_output = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    output_token_count = len(generated_tokens)
                    
                    try:
                        # Clear some memory before moving back to GPU
                        torch.cuda.empty_cache()
                        self.model = self.model.to(self.device)
                    except Exception as e2:
                        self.logger.warning(f"Couldn't move back to GPU: {e2}")
                else:
                    raise
            
        except Exception as e:
            self.logger.error(f"Error during generation: {str(e)}")
            model_output = f"Error during generation: {str(e)}"
            output_token_count = len(abstract) // 4  # Rough estimate
            input_token_count = len(abstract) // 4   # Rough estimate
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Return metrics
        metrics = {
            "input_tokens": input_token_count if 'input_token_count' in locals() else 0,
            "output_tokens": output_token_count if 'output_token_count' in locals() else 0,
            "processing_time": processing_time
        }
        
        return model_output, metrics

    
    def process_file(self, input_file: str) -> Dict:
        """Process a single JSON file containing paper abstracts."""
        file_start_time = time.time()
        file_basename = os.path.basename(input_file)
        
        self.logger.info(f"Processing file: {file_basename}")
        
        # Load the JSON file
        try:
            with open(input_file, "r", encoding="utf-8") as f:
                file_content = f.read()
            
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
                "raw_outputs_generated": 0,
                "total_time": time.time() - file_start_time
            }
        
        # Initialize file metrics
        file_metrics = {
            "input_tokens": 0,
            "output_tokens": 0,
            "processing_time": 0,
            "papers_count": len(papers),
            "raw_outputs_generated": 0
        }
        
        # Create a tqdm progress bar for papers in this file
        with tqdm(total=len(papers), desc=f"File: {file_basename}", unit="paper") as pbar:
            # Process each paper
            for i, paper in enumerate(papers):
                try:
                    paper_code = paper.get("paper_code")
                    if not paper_code:
                        paper_code = f"paper_{i}"
                        self.logger.warning(f"Paper {i} has no paper_code, assigning: {paper_code}")
                    
                    abstract = paper.get("abstract", "")
                    if not abstract:
                        self.logger.warning(f"Paper {paper_code} has no abstract, skipping")
                        pbar.update(1)
                        continue
                    
                    # Include paper progress in the progress bar description
                    pbar.set_description(f"File: {file_basename} | Paper: {paper_code} ({i+1}/{len(papers)})")
                    
                    # Process the abstract to get chunks
                    raw_output, metrics = self.generate_model_output(abstract)
                    
                    # Save the raw output
                    raw_output_path = self.raw_output_dir / f"{paper_code}_chunks.txt"
                    with open(raw_output_path, "w", encoding="utf-8") as f:
                        f.write(raw_output)
                    
                    # Update file metrics
                    file_metrics["input_tokens"] += metrics["input_tokens"]
                    file_metrics["output_tokens"] += metrics["output_tokens"]
                    file_metrics["processing_time"] += metrics["processing_time"]
                    file_metrics["raw_outputs_generated"] += 1
                    
                    # Update global metrics
                    self.metrics["total_input_tokens"] += metrics["input_tokens"]
                    self.metrics["total_output_tokens"] += metrics["output_tokens"]
                    self.metrics["total_processing_time"] += metrics["processing_time"]
                    self.metrics["raw_outputs_generated"] += 1
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
                    
                    # Periodically clear CUDA cache to prevent memory fragmentation
                    if i > 0 and i % 10 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                except Exception as e:
                    self.logger.error(f"Error processing paper {paper.get('paper_code', f'paper_{i}')}: {str(e)}")
                    import traceback
                    self.logger.error(f"Traceback: {traceback.format_exc()}")
                    pbar.update(1)
                    continue
        
        # Save file-specific metrics
        file_metrics["total_time"] = time.time() - file_start_time
        if file_metrics["papers_count"] > 0:
            file_metrics["average_paper_time"] = file_metrics["processing_time"] / file_metrics["papers_count"]
        else:
            file_metrics["average_paper_time"] = 0
        
        # Save file metrics
        metrics_file = self.metrics_dir / f"{os.path.basename(input_file).replace('.json', '')}_metrics.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(file_metrics, f, indent=2)
        
        self.logger.info(f"Completed processing file: {file_basename}")
        self.logger.info(f"File metrics: {file_metrics}")
        
        # Update global metrics
        self.metrics["files_processed"] += 1
        
        return file_metrics
    
    def process_domain(self) -> None:
        """Process all files in the domain directory with tqdm for tracking overall progress."""
        domain_start_time = time.time()
        
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
                
                # Calculate and show overall progress statistics
                if file_idx > 0:
                    # Calculate average processing time per file
                    avg_file_time = sum(m["total_time"] for m in file_metrics_history) / len(file_metrics_history)
                    
                    # Estimate remaining time
                    remaining_files = len(json_files) - (file_idx + 1)
                    est_remaining_time = avg_file_time * remaining_files
                    est_completion_time = datetime.now() + timedelta(seconds=est_remaining_time)
                    
                    # Update progress bar with overall statistics
                    domain_pbar.set_postfix({
                        "Avg file": f"{avg_file_time:.1f}s",
                        "ETA": est_completion_time.strftime("%H:%M:%S"),
                        "Remaining": str(timedelta(seconds=int(est_remaining_time)))
                    })
                
                # Regularly clear CUDA cache between files
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
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
        
        # Print final summary to console
        total_papers = self.metrics["papers_processed"]
        total_outputs = self.metrics["raw_outputs_generated"]
        total_time = str(timedelta(seconds=int(self.metrics["total_time"])))
        
        print("\n" + "="*80)
        print(f"PROCESSING SUMMARY FOR DOMAIN: {self.domain}")
        print("="*80)
        print(f"Total files processed:    {self.metrics['files_processed']}")
        print(f"Total papers processed:   {total_papers}")
        print(f"Total outputs generated:  {total_outputs}")
        print(f"Total processing time:    {total_time}")
        if total_papers > 0:
            print(f"Average time per paper:   {self.metrics['average_paper_time']:.2f}s")
        print("="*80)
        
        # Unload model and free memory before exiting
        self._unload_model()

def main():
    """Main function to run the raw chunk output extractor for multiple domains."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process abstracts using LLM to chunk them into sections")
    parser.add_argument("--domains", type=str, nargs='+', default=["JMIR", "DH"],
                        help="Domains to process (default: all five domains)")
    parser.add_argument("--model", type=str, default=Config.MODEL_NAME, help="HuggingFace model name")
    parser.add_argument("--prompt-template", type=str, default=Config.PROMPT_TEMPLATE_NAME,
                    help="Name of the prompt template (used in output folder structure)")
    parser.add_argument("--output-base-dir", type=str, default=Config.OUTPUT_BASE_DIR,
                    help="Base directory for all outputs (default: ./baselines/LLM/output/Event_Segmentation)")
    parser.add_argument("--input-dir", type=str, default=Config.INPUT_DIR,
                    help="Base directory for domain-specific input JSON files (default: ./SciEvent_data/raw/domain_specific_unannotated)")
    parser.add_argument("--clean-cache", action="store_true", help="Force redownload the model")
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    config = Config()
    config.MODEL_NAME = args.model
    config.PROMPT_TEMPLATE_NAME = args.prompt_template
    config.OUTPUT_BASE_DIR = args.output_base_dir
    config.INPUT_DIR = args.input_dir
    
    # Print banner
    print("\n" + "="*80)
    print(f"ABSTRACT SECTION CHUNKER - SEQUENTIAL PROCESSING")
    print("="*80)
    print(f"Domains:           {', '.join(args.domains)}")
    print(f"Model:             {config.MODEL_NAME}")
    print(f"Clean Cache:       {args.clean_cache}")
    print("="*80 + "\n")
    
    # Process each domain sequentially
    for i, domain in enumerate(args.domains):
        print(f"\n{'='*80}")
        print(f"PROCESSING DOMAIN {i+1}/{len(args.domains)}: {domain}")
        print(f"{'='*80}\n")
        
        # Force clean GPU between domains
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Free all memory not used by PyTorch
            gc.collect()
        
        # Initialize and run the raw chunk output extractor
        extractor = RawChunkOutputExtractor(config, domain)
        extractor.process_domain()
        
        # Force clean up after domain processing
        del extractor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Short pause between domains to let system stabilize
        if i < len(args.domains) - 1:
            print(f"\nPausing for 10 seconds before starting the next domain to free resources...")
            time.sleep(10)
    
    print(f"\n{'='*80}")
    print(f"ALL DOMAINS PROCESSED SUCCESSFULLY")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()