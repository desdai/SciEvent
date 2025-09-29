import json
import os
import time
import logging
import gc
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import openai
from openai import OpenAI
import asyncio
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import argparse

# Configuration
class Config:
    # API settings
    API_KEY = "Your api key"  # Replace with your actual API key
    MODEL_NAME = "gpt-4.1"  # ChatGPT 4.1 model name
    
    # Folder structure
    INPUT_DIR = "./SciEvent_data/raw/to_be_annotated"
    OUTPUT_BASE_DIR = "./SciEvent_data/LLM/Event_Segmentation"

    # Prompt template name
    PROMPT_TEMPLATE_NAME = "Zero-Shot_Event_Segmentation"
    
    # Logging
    LOG_LEVEL = logging.INFO
    
    # Processing settings
    MAX_TOKENS = 1200
    TEMPERATURE = 0.3
    TOP_P = 0.9
    
    # Rate limiting
    MAX_CONCURRENT_REQUESTS = 10  # Maximum number of concurrent API calls


# Chunking prompt template
CHUNK_PROMPT = """You are a strict extraction assistant. Never explain, never repeat, only extract in the required format.

### Abstract: ###
{abstract}

### Extraction Rules: ###
- Copy full, continuous sentences from the abstract. No changes, summaries, or guessing allowed.
- Each sentence must belong to only one section.
- Sections must use continuous text spans. No skipping around.
- If no content fits a section, output exactly <NONE>.
- No explanations, no extra text, no format changes.

### Section Definitions: ###
- [Background]: Problem, motivation, context, research gap, or objectives.
- [Method]: Techniques, experimental setups, frameworks, datasets.
- [Results]: Main findings, discoveries, statistics, or trends.
- [Implications]: Importance, impact, applications, or future work.

### Exact Output Format: ###
[Background]: <EXACT TEXT or <NONE>>

[Method]: <EXACT TEXT or <NONE>>

[Results]: <EXACT TEXT or <NONE>>

[Implications]: <EXACT TEXT or <NONE>>

<|END_OF_CHUNK|>
"""


class OpenAIChunkOutputExtractor:
    def __init__(self, config: Config, domain: str):
        """Initialize the output extractor with configuration and domain."""
        self.config = config
        self.domain = domain
        
        # Create directory structure
        self._setup_directories()
        
        # Set up logging
        self._setup_logging()
        
        # Set up OpenAI client
        self._setup_openai_client()
        
        # Initialize metrics
        self.metrics = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_processing_time": 0,
            "files_processed": 0,
            "papers_processed": 0,
            "raw_outputs_generated": 0,
            "api_cost": 0.0  # Track API costs
        }
        
        # Semaphore for controlling concurrent API calls
        self.semaphore = asyncio.Semaphore(self.config.MAX_CONCURRENT_REQUESTS)
    
    def _setup_directories(self):
        """Create the directory structure for outputs."""
        # Create base output directory structure
        self.output_dir = Path(f"{self.config.OUTPUT_BASE_DIR}/{self.config.MODEL_NAME}/{self.config.PROMPT_TEMPLATE_NAME}/{self.domain}")
        
        # Create raw output directories
        self.raw_output_dir = self.output_dir / "raw_output"
        self.logs_dir = self.output_dir / "logs"
        self.metrics_dir = self.output_dir / "metrics"
        
        # Create output directories
        os.makedirs(self.raw_output_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        print(f"Created directory structure at {self.output_dir}")
    
    def _setup_logging(self):
        """Set up logging configuration."""
        # Create logger
        self.logger = logging.getLogger(f"OpenAIChunkOutputExtractor_{self.domain}")
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
    
    def _setup_openai_client(self):
        """Set up the OpenAI API client."""
        # Set API key for the client
        self.client = OpenAI(api_key=self.config.API_KEY)
        self.logger.info(f"OpenAI client initialized with model: {self.config.MODEL_NAME}")
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type((openai.APIError, openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError))
    )
    async def _call_openai_api(self, abstract: str) -> Tuple[str, Dict[str, Any]]:
        """Call the OpenAI API with retry logic."""
        async with self.semaphore:
            try:
                start_time = time.time()
                
                # Prepare messages
                messages = [
                    {"role": "system", "content": "You are a helpful assistant that extracts sections from academic abstracts."},
                    {"role": "user", "content": CHUNK_PROMPT.format(abstract=abstract)}
                ]
                
                # Call the API with the new client interface
                response = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=self.config.MODEL_NAME,
                    messages=messages,
                    max_tokens=self.config.MAX_TOKENS,
                    temperature=self.config.TEMPERATURE,
                    top_p=self.config.TOP_P,
                    n=1,
                    stream=False
                )
                
                # Extract response content
                model_output = response.choices[0].message.content
                
                # Calculate processing time
                processing_time = time.time() - start_time
                
                # Calculate tokens and cost
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                
                # Calculate cost (adjust rates as needed for the actual model)
                input_cost = input_tokens * (0.03 / 1000)  # $0.03 per 1000 tokens for input
                output_cost = output_tokens * (0.06 / 1000)  # $0.06 per 1000 tokens for output
                total_cost = input_cost + output_cost
                
                # Return metrics
                metrics = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "processing_time": processing_time,
                    "cost": total_cost
                }
                
                return model_output, metrics
                
            except Exception as e:
                self.logger.error(f"Error calling OpenAI API: {str(e)}")
                raise
    
    async def generate_model_output(self, abstract: str) -> Tuple[str, Dict[str, int]]:
        """Generate raw model output from OpenAI API."""
        try:
            # Call the OpenAI API
            model_output, metrics = await self._call_openai_api(abstract)
            
            # Add END_OF_CHUNK marker if it's not present
            if "<|END_OF_CHUNK|>" not in model_output:
                model_output += "\n<|END_OF_CHUNK|>"
            
            return model_output, metrics
            
        except Exception as e:
            self.logger.error(f"Error during generation: {str(e)}")
            model_output = f"Error during generation: {str(e)}"
            
            # Rough estimate for error case
            metrics = {
                "input_tokens": len(abstract) // 4,
                "output_tokens": 0,
                "processing_time": 0,
                "cost": 0
            }
            
            return model_output, metrics

    async def process_paper(self, paper, paper_code, i, total_papers):
        """Process a single paper abstract."""
        try:
            abstract = paper.get("abstract", "")
            if not abstract:
                self.logger.warning(f"Paper {paper_code} has no abstract, skipping")
                return None
            
            # Process the abstract to get chunks
            raw_output, metrics = await self.generate_model_output(abstract)
            
            # Save the raw output
            raw_output_path = self.raw_output_dir / f"{paper_code}_chunks.txt"
            with open(raw_output_path, "w", encoding="utf-8") as f:
                f.write(raw_output)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error processing paper {paper_code}: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    async def process_file(self, input_file: str) -> Dict:
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
                        "total_time": time.time() - file_start_time,
                        "api_cost": 0.0
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
                "total_time": time.time() - file_start_time,
                "api_cost": 0.0
            }
        
        # Initialize file metrics
        file_metrics = {
            "input_tokens": 0,
            "output_tokens": 0,
            "processing_time": 0,
            "papers_count": len(papers),
            "raw_outputs_generated": 0,
            "api_cost": 0.0
        }
        
        # Create a progress bar for this file
        progress_bar = tqdm(total=len(papers), desc=f"File: {file_basename}", unit="paper")
        
        # Process papers concurrently with rate limiting through semaphore
        tasks = []
        for i, paper in enumerate(papers):
            paper_code = paper.get("paper_code")
            if not paper_code:
                paper_code = f"paper_{i}"
                self.logger.warning(f"Paper {i} has no paper_code, assigning: {paper_code}")
            
            task = asyncio.create_task(self.process_paper(paper, paper_code, i, len(papers)))
            tasks.append((task, paper_code, i))
        
        # Wait for all tasks to complete
        for task, paper_code, i in tasks:
            metrics = await task
            
            # Update progress bar
            progress_bar.update(1)
            progress_bar.set_description(f"File: {file_basename} | Paper: {paper_code} ({i+1}/{len(papers)})")
            
            if metrics:
                # Update file metrics
                file_metrics["input_tokens"] += metrics["input_tokens"]
                file_metrics["output_tokens"] += metrics["output_tokens"]
                file_metrics["processing_time"] += metrics["processing_time"]
                file_metrics["raw_outputs_generated"] += 1
                file_metrics["api_cost"] += metrics["cost"]
                
                # Update global metrics
                self.metrics["total_input_tokens"] += metrics["input_tokens"]
                self.metrics["total_output_tokens"] += metrics["output_tokens"]
                self.metrics["total_processing_time"] += metrics["processing_time"]
                self.metrics["raw_outputs_generated"] += 1
                self.metrics["papers_processed"] += 1
                self.metrics["api_cost"] += metrics["cost"]
                
                # Calculate and display estimated time remaining for file
                if i > 0:  # Need at least one paper to estimate
                    avg_paper_time = file_metrics["processing_time"] / (i + 1)
                    remaining_papers = len(papers) - (i + 1)
                    est_remaining_time = avg_paper_time * remaining_papers
                    est_completion_time = datetime.now() + timedelta(seconds=est_remaining_time)
                    
                    # Update progress bar postfix with time estimates
                    progress_bar.set_postfix({
                        "Avg paper": f"{avg_paper_time:.2f}s", 
                        "ETA": est_completion_time.strftime("%H:%M:%S"),
                        "Cost": f"${file_metrics['api_cost']:.2f}"
                    })
        
        # Close progress bar
        progress_bar.close()
        
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
    
    async def process_domain(self) -> None:
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
        domain_progress_bar = tqdm(total=len(json_files), desc=f"Domain: {self.domain}", unit="file")
        
        for file_idx, file in enumerate(json_files):
            input_file = os.path.join(input_dir, file)
            
            # Process file and collect metrics
            file_metrics = await self.process_file(input_file)
            file_metrics_history.append(file_metrics)
            
            # Update domain progress bar
            domain_progress_bar.update(1)
            
            # Calculate and show overall progress statistics
            if file_idx > 0:
                # Calculate average processing time per file
                avg_file_time = sum(m["total_time"] for m in file_metrics_history) / len(file_metrics_history)
                
                # Estimate remaining time
                remaining_files = len(json_files) - (file_idx + 1)
                est_remaining_time = avg_file_time * remaining_files
                est_completion_time = datetime.now() + timedelta(seconds=est_remaining_time)
                
                # Calculate total API cost so far
                total_api_cost = sum(m["api_cost"] for m in file_metrics_history)
                
                # Update progress bar with overall statistics
                domain_progress_bar.set_postfix({
                    "Avg file": f"{avg_file_time:.1f}s",
                    "ETA": est_completion_time.strftime("%H:%M:%S"),
                    "Cost": f"${total_api_cost:.2f}"
                })
        
        # Close domain progress bar
        domain_progress_bar.close()
        
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
        print(f"Total API cost:           ${self.metrics['api_cost']:.2f}")
        if total_papers > 0:
            print(f"Average time per paper:   {self.metrics['average_paper_time']:.2f}s")
            print(f"Average cost per paper:   ${self.metrics['api_cost'] / total_papers:.4f}")
        print("="*80)


async def process_domains(config, domains):
    """Process multiple domains sequentially."""
    for i, domain in enumerate(domains):
        print(f"\n{'='*80}")
        print(f"PROCESSING DOMAIN {i+1}/{len(domains)}: {domain}")
        print(f"{'='*80}\n")
        
        # Initialize and run the output extractor
        extractor = OpenAIChunkOutputExtractor(config, domain)
        await extractor.process_domain()
        
        # Force clean up after domain processing
        del extractor
        gc.collect()
        
        # Short pause between domains to let system stabilize
        if i < len(domains) - 1:
            print(f"\nPausing for 5 seconds before starting the next domain...")
            await asyncio.sleep(5)
    
    print(f"\n{'='*80}")
    print(f"ALL DOMAINS PROCESSED SUCCESSFULLY")
    print(f"{'='*80}")


def main():
    """Main function to run the OpenAI extractor for multiple domains."""
    parser = argparse.ArgumentParser(description="Process abstracts using ChatGPT to chunk them into sections")
    parser.add_argument("--domains", type=str, nargs='+', default=["JMIR", "DH"],
                        help="Domains to process (default: JMIR and DH)")
    parser.add_argument("--api-key", type=str, help="OpenAI API key")
    parser.add_argument("--prompt-template", type=str, default=Config.PROMPT_TEMPLATE_NAME,
                    help="Name of the prompt template (used in output folder structure)")
    parser.add_argument("--output-base-dir", type=str, default=Config.OUTPUT_BASE_DIR,
                    help="Base directory for all outputs (default: ./baselines/LLM/output/Event_Segmentation)")
    parser.add_argument("--input-dir", type=str, default=Config.INPUT_DIR,
                    help="Base directory for domain-specific input JSON files (default: SciEvent_data/to_be_annotated)")
    parser.add_argument("--model", type=str, default=Config.MODEL_NAME, 
                        help="OpenAI model name (default: gpt-4-1106-preview)")
    parser.add_argument("--max-concurrent", type=int, default=10,
                        help="Maximum number of concurrent API calls (default: 10)")
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    config = Config()

    config.PROMPT_TEMPLATE_NAME = args.prompt_template
    config.OUTPUT_BASE_DIR = args.output_base_dir
    config.INPUT_DIR = args.input_dir

    if args.api_key:
        config.API_KEY = args.api_key
    elif os.environ.get("OPENAI_API_KEY"):
        config.API_KEY = os.environ.get("OPENAI_API_KEY")
    
    # Set the model name properly
    if args.model:
        config.MODEL_NAME = args.model
    elif config.MODEL_NAME == "":  # If default is empty, set a fallback
        config.MODEL_NAME = "gpt-4-1106-preview"  # Fallback model
        
    config.MAX_CONCURRENT_REQUESTS = args.max_concurrent
    
    # Print banner
    print("\n" + "="*80)
    print(f"ABSTRACT SECTION CHUNKER - OPENAI API")
    print("="*80)
    print(f"Domains:             {', '.join(args.domains)}")
    print(f"Model:               {config.MODEL_NAME}")
    print(f"Max Concurrent:      {config.MAX_CONCURRENT_REQUESTS}")
    print(f"API Key:             {'Provided' if config.API_KEY else 'NOT PROVIDED'}")
    print("="*80 + "\n")
    
    # Validate API key
    if not config.API_KEY or config.API_KEY == "your-api-key-here":
        print("ERROR: API key not provided. Use --api-key or set OPENAI_API_KEY environment variable.")
        return
    
    # Run async event loop
    asyncio.run(process_domains(config, args.domains))


if __name__ == "__main__":
    main()