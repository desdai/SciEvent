import json
import os

def ablate_domains(input_file, output_dir):
    # List of domains
    domains = ["ACL", "bioinfo", "cscw", "dh", "jmir"]
    
    # Load the original full train file
    with open(input_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # For each domain, create a version with that domain removed
    for domain in domains:
        ablated_data = [
            instance for instance in data 
            if not instance.get("doc_id", "").startswith(domain)
        ]

        # Create subfolder like no_acl, no_bioinfo...
        subfolder = os.path.join(output_dir, f"no_{domain.lower()}")
        os.makedirs(subfolder, exist_ok=True)

        # Save the ablated data inside that subfolder
        output_path = os.path.join(subfolder, f"train_without_{domain.lower()}.json")
        with open(output_path, "w", encoding="utf-8") as out_f:
            for item in ablated_data:
                out_f.write(json.dumps(item) + "\n")
        
        print(f"Saved {len(ablated_data)} examples to {output_path}")

if __name__ == "__main__":
    # Example usage
    input_file = "SciEvent_data/DEGREE/all_splits/train.json"           # your input JSON file
    output_dir = "SciEvent_data/DEGREE/ablation"   # folder to save outputs
    ablate_domains(input_file, output_dir)
