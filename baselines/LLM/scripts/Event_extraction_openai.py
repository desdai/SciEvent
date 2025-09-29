"""
Raw output extractor that generates LLM outputs for abstract segments using OpenAI ChatGPT API
and saves the complete raw outputs without additional processing.
Now with tqdm progress tracking for better time estimation.
"""

import json
import os
import time
import logging
import requests
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm


# Configuration
class Config:
    # OpenAI API settings
    OPENAI_API_KEY = "your_api_key_here"  # Replace with your actual API key
    OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"  # Current chat completions endpoint
    
    # Model settings
    MODEL_NAME = "gpt-4.1"  # Default model, can be changed.
    
    # Folder structure
    INPUT_DIR = "./SciEvent_data/raw/to_be_annotated"
    OUTPUT_BASE_DIR = "./SciEvent_data/LLM/Event_Extraction"
    
    # Prompt template name
    PROMPT_TEMPLATE_NAME = "Few-shot-2_Event_Extraction" # you can change the name of the prompt template
    
    # Logging
    LOG_LEVEL = logging.INFO
    
    # Processing settings
    BATCH_SIZE = 1 # Number of papers to process in one batch
    MAX_NEW_TOKENS = 800  # Equivalent to max_tokens in OpenAI API
    TEMPERATURE = 0.3  # Set to 0.0 for deterministic output
    TOP_P = 0.9  # Set to 1.0 for deterministic output

class RawOutputExtractor:
    def __init__(self, config: Config, domain: str):
        """Initialize the raw output extractor with configuration and domain."""
        self.config = config
        self.domain = domain
        self.api_key = config.OPENAI_API_KEY
        self.api_url = config.OPENAI_API_URL
        
        # Create directory structure
        self._setup_directories()
        
        # Set up logging
        self._setup_logging()
        
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
    
    def _setup_directories(self):
        """Create the directory structure for outputs."""
        # Create base output directory structure
        self.output_dir = Path(f"{self.config.OUTPUT_BASE_DIR}/{self.config.MODEL_NAME}/{self.config.PROMPT_TEMPLATE_NAME}/{self.domain}")
        
        # Create raw output and logs directories - using same structure as original
        self.raw_output_dir = self.output_dir / "raw_output"
        self.logs_dir = self.output_dir / "logs"
        self.metrics_dir = self.output_dir / "generated_file_metric_logs"
        
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
    
    def generate_model_output(self, abstract: str) -> Tuple[str, Dict[str, int]]:
        """
        Generate raw model output using the OpenAI API.
        Returns only the LLM generated response and metrics.
        """
        start_time = time.time()
        
        # Estimate input tokens count (rough approximation)
        input_token_count = len(abstract.split()) + 300  # base prompt + abstract words
        
        try:
            # Prepare the messages format for chat completions API
            messages = [
                {"role": "system", "content": "You are an expert argument annotator that extracts information in JSON format."},
                {"role": "user", "content": PROMPT_TEMPLATE.replace("{abstract}", abstract)}
            ]
            
            # Create request payload for chat completions
            payload = {
                "model": self.config.MODEL_NAME,
                "messages": messages,
                "max_tokens": self.config.MAX_NEW_TOKENS,
                "temperature": self.config.TEMPERATURE,
                "top_p": self.config.TOP_P,
                "n": 1,  # Number of completions to generate
                "response_format": {"type": "json_object"}  # Ensure JSON response
            }
            
            # Set up headers with API key
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # Make API request with retry mechanism
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    self.logger.info(f"Sending request to OpenAI API...")
                    response = requests.post(
                        self.api_url, 
                        json=payload, 
                        headers=headers,
                        timeout=60  # Set a timeout of 60 seconds
                    )
                    
                    if response.status_code == 200:
                        # Successful response
                        response_data = response.json()
                        
                        # Extract the content from the response
                        model_output = response_data["choices"][0]["message"]["content"]
                        
                        # Get token counts if provided by the API
                        output_token_count = response_data.get("usage", {}).get("completion_tokens", len(model_output.split()))
                        total_tokens = response_data.get("usage", {}).get("total_tokens", input_token_count + output_token_count)
                        
                        break  # Exit retry loop on success
                    
                    elif response.status_code == 429:
                        # Rate limit exceeded, wait and retry
                        retry_count += 1
                        wait_time = (2 ** retry_count) * 10  # Exponential backoff
                        self.logger.warning(f"Rate limit exceeded. Waiting {wait_time} seconds before retry {retry_count}/{max_retries}...")
                        time.sleep(wait_time)
                    
                    else:
                        # Other errors
                        error_message = f"API Error: {response.status_code} - {response.text}"
                        self.logger.error(error_message)
                        model_output = f"Error: {error_message}"
                        break
                
                except requests.exceptions.RequestException as e:
                    # Handle network errors
                    retry_count += 1
                    wait_time = (2 ** retry_count) * 5  # Exponential backoff
                    self.logger.warning(f"Network error: {str(e)}. Waiting {wait_time} seconds before retry {retry_count}/{max_retries}...")
                    time.sleep(wait_time)
                    
                    if retry_count >= max_retries:
                        model_output = f"Error after {max_retries} retries: {str(e)}"
            
            # If we exhausted all retries without a successful API call
            if retry_count >= max_retries and not isinstance(model_output, str):
                model_output = f"Error: Failed after {max_retries} retries"
                output_token_count = 10  # Placeholder
            
        except Exception as e:
            self.logger.error(f"Error during generation: {str(e)}")
            # Create default output for error cases
            model_output = f"Error during generation: {str(e)}"
            output_token_count = len(abstract) // 4  # Rough estimate
            
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Metrics
        metrics = {
            "input_tokens": input_token_count,
            "output_tokens": output_token_count if 'output_token_count' in locals() else len(model_output.split()),
            "processing_time": processing_time
        }
        
        # Return only the model output with metrics
        return model_output, metrics
    
    def process_file(self, input_file: str) -> Dict:
        """Process a single JSON file containing paper abstracts and return file metrics."""
        file_start_time = time.time()
        file_basename = os.path.basename(input_file)
        
        self.logger.info(f"Processing file: {file_basename}")
        
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
                            # Generate complete raw output
                            raw_output, metrics = self.generate_model_output(abstract)
                            
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
                                event_text = event.get("Text", "")
                                
                                # Find event type
                                event_type = None
                                try:
                                    for key in event.keys():
                                        if key not in ["Text", "Action", "Arguments", "Summary"]:
                                            event_type = key
                                            break
                                except Exception as e:
                                    self.logger.error(f"Error finding event type: {e}")
                                    event_type = None
                                
                                # Update event progress bar description with event type
                                if event_type:
                                    event_pbar.set_description(f"  Events in {paper_code} | Type: {event_type} ({event_index+1}/{len(events)})")
                                else:
                                    event_pbar.set_description(f"  Events in {paper_code} | Event: {event_index+1}/{len(events)}")
                                
                                if not event_text:
                                    self.logger.warning(f"Empty event text for paper {paper_code}, event {event_index}")
                                    event_pbar.update(1)
                                    continue
                                
                                try:
                                    # Generate raw model output and save as-is
                                    raw_output, metrics = self.generate_model_output(event_text)
                                    
                                    # Save the complete raw output exactly as received
                                    raw_txt_file_path = self.raw_output_dir / f"{paper_code}_event_{event_index}.txt"
                                    with open(raw_txt_file_path, "w", encoding="utf-8") as raw_txt_file:
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
        
def main():
    """Main function to run the raw output extractor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process abstracts using OpenAI ChatGPT API and save raw outputs")
    parser.add_argument("--domain", type=str, required=True, help="Domain to process (e.g., ACL, JMIR, CSCW)")
    parser.add_argument("--model", type=str, default=Config.MODEL_NAME, help="OpenAI model name (e.g., gpt-3.5-turbo, gpt-4)")
    parser.add_argument("--prompt", type=str, default=Config.PROMPT_TEMPLATE_NAME, help="Name of the prompt template")
    parser.add_argument("--output-base-dir", type=str, default=Config.OUTPUT_BASE_DIR,
                    help="Base directory for all outputs (default: ./baselines/LLM/output/Event_Extraction)")
    parser.add_argument("--input-dir", type=str, default=Config.INPUT_DIR,
                    help="Base directory for domain-specific input JSON files (default: ./SciEvent_data/raw/to_be_annotated)")
    parser.add_argument("--api-key", type=str, help="OpenAI API key (overrides config)")
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    config = Config()
    config.MODEL_NAME = args.model
    config.PROMPT_TEMPLATE_NAME = args.prompt
    config.OUTPUT_BASE_DIR = args.output_base_dir
    config.INPUT_DIR = args.input_dir

    # Override API key if provided
    if args.api_key:
        config.OPENAI_API_KEY = args.api_key
    
    # Print banner
    print("\n" + "="*80)
    print(f"RAW OUTPUT EXTRACTOR (OpenAI ChatGPT API Version)")
    print("="*80)
    print(f"Domain:            {args.domain}")
    print(f"Model:             {config.MODEL_NAME}")
    print(f"Prompt Template:   {config.PROMPT_TEMPLATE_NAME}")
    print(f"API URL:           {config.OPENAI_API_URL}")
    print("="*80 + "\n")
    
    # --- NEW: load prompt file into the global PROMPT_TEMPLATE ---
    PROMPT_DIR = "baselines/LLM/prompts"  # fixed folder

    def load_prompt_text(prompt_name: str, prompt_dir: str = PROMPT_DIR) -> str:
        """Return the prompt text from prompt_dir + prompt_name[.txt|.md]."""
        from pathlib import Path
        base = Path(prompt_dir)
        name = prompt_name

        # if user passed an explicit extension, use it
        if Path(name).suffix:
            p = base / name
            if p.exists():
                return p.read_text(encoding="utf-8")
            raise FileNotFoundError(f"Prompt file not found: {p}")

        # otherwise try common extensions
        for cand in (base / f"{name}.txt", base / f"{name}.md"):
            if cand.exists():
                return cand.read_text(encoding="utf-8")

        raise FileNotFoundError(
            f"Prompt template '{name}' not found in {base} "
            f"(tried .txt and .md)."
        )

    global PROMPT_TEMPLATE
    PROMPT_TEMPLATE = load_prompt_text(config.PROMPT_TEMPLATE_NAME)

    # (Optional) sanity check
    if "{abstract}" not in PROMPT_TEMPLATE:
        print("[WARN] Prompt template missing '{abstract}' placeholder.")
    # --- END NEW ---

    # Check if API key is set
    if not config.OPENAI_API_KEY or config.OPENAI_API_KEY == "your_api_key_here":
        print("ERROR: OpenAI API key not set. Please set it in the config or provide it with --api-key.")
        return
    
    # Initialize and run the raw output extractor
    extractor = RawOutputExtractor(config, args.domain)
    extractor.process_domain()

if __name__ == "__main__":
    # Run the main function
    main()