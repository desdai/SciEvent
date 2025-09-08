"""
SemEval 2013, Task 9 (based on MUC):

****Legend****
N: No output
1: Some output
12: A similar output to "1", partially overlap
2: A different output to "1". no overlap

Feasible input: Type classification is only 1, or 2 or N
                Span identification is all 1, 12, or 2 or N

****Definiton From MUC****:
Correct (COR) : both are the same;
Incorrect (INC) : the output of a system and the golden annotation don’t match;
Partial (PAR) : system and the golden annotation are somewhat “similar” but not the same;
Missing (MIS) : a golden annotation is not captured by a system;
Spurius (SPU) : system produces a response which doesn’t exist in the golden annotation;


****Four options****:

**Strict: Strict evaluation (exact-boundary and type matching).

Span:N ; Type:N ~ Span:1 ; Type:1 -> SPU            Hypothesised span and section
Span:1 ; Type:1 ~ Span:N ; Type:N -> MIS            Missed span and section
Span:1 ; Type:1 ~ Span:1 ; Type:1 -> COR            Matched span and section
Span:1 ; Type:1 ~ Span:12 ; Type:1 -> INC           Partial span, correct section
Span:1 ; Type:1 ~ Span:1 ; Type:2 -> INC            Correct span, wrong section
Span:1 ; Type:1 ~ Span:12 ; Type:2 -> INC           Partial span, wrong section
Span:1 ; Type:1 ~ Span:2 ; Type:any -> INC          wrong span (not likely to appear, and not discussed with MUC, SemEval)


**Exact: Exact boundary matching (regardless to the type). (from here to do)

Span:N ; Type:N ~ Span:1 ; Type:1 -> SPU            Hypothesised span and section
Span:1 ; Type:1 ~ Span:N ; Type:N -> MIS            Missed span and section
Span:1 ; Type:1 ~ Span:1 ; Type:1 -> COR            Matched span and section
Span:1 ; Type:1 ~ Span:12 ; Type:1 -> INC           Partial span, correct section
Span:1 ; Type:1 ~ Span:1 ; Type:2 -> COR            Correct span, wrong section *
Span:1 ; Type:1 ~ Span:12 ; Type:2 -> INC           Partial span, wrong section
Span:1 ; Type:1 ~ Span:2 ; Type:any -> INC          wrong span


**Partial: Partial boundary matching (regardless to the type).

Span:N ; Type:N ~ Span:1 ; Type:1 -> SPU            Hypothesised span and section
Span:1 ; Type:1 ~ Span:N ; Type:N -> MIS            Missed span and section
Span:1 ; Type:1 ~ Span:1 ; Type:1 -> COR            Matched span and section
Span:1 ; Type:1 ~ Span:12 ; Type:1 -> PAR           Partial span, correct section
Span:1 ; Type:1 ~ Span:1 ; Type:2 -> COR            Correct span, wrong section
Span:1 ; Type:1 ~ Span:12 ; Type:2 -> PAR           Partial span, wrong section
Span:1 ; Type:1 ~ Span:2 ; Type:any -> INC          wrong span


**Type: Type matching (some span overlap is required).

Span:N ; Type:N ~ Span:1 ; Type:1 -> SPU            Hypothesised span and section
Span:1 ; Type:1 ~ Span:N ; Type:N -> MIS            Missed span and section
Span:1 ; Type:1 ~ Span:1 ; Type:1 -> COR            Matched span and section
Span:1 ; Type:1 ~ Span:12 ; Type:1 -> COR           Partial span, correct section
Span:1 ; Type:1 ~ Span:1 ; Type:2 -> INC            Correct span, wrong section
Span:1 ; Type:1 ~ Span:12 ; Type:2 -> INC           Partial span, wrong section
Span:1 ; Type:1 ~ Span:2 ; Type:any -> INC          wrong span



POS = COR + PAR + INC + MIS
ACT = COR + PAR + INC + SPU


precision = COR + (PAR x 0.5) / ACT

recall = COR + (PAR x 0.5) / POS



*** par is only non-zero for partial


****Reference****
@inproceedings{segura-bedmar-etal-2013-semeval,
    title = "{S}em{E}val-2013 Task 9 : Extraction of Drug-Drug Interactions from Biomedical Texts ({DDIE}xtraction 2013)",
    author = "Segura-Bedmar, Isabel  and
      Mart{\'i}nez, Paloma  and
      Herrero-Zazo, Mar{\'i}a",
    editor = "Manandhar, Suresh  and
      Yuret, Deniz",
    booktitle = "Second Joint Conference on Lexical and Computational Semantics (*{SEM}), Volume 2: Proceedings of the Seventh International Workshop on Semantic Evaluation ({S}em{E}val 2013)",
    month = jun,
    year = "2013",
    address = "Atlanta, Georgia, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/S13-2056/",
    pages = "341--350"
}

@inproceedings{chinchor-sundheim-1993-muc,
    title = "{MUC}-5 Evaluation Metrics",
    author = "Chinchor, Nancy  and
      Sundheim, Beth",
    booktitle = "Fifth Message Understanding Conference ({MUC}-5): Proceedings of a Conference Held in Baltimore, {M}aryland, August 25-27, 1993",
    year = "1993",
    url = "https://aclanthology.org/M93-1007/"
}
"""
import os
import re
from collections import Counter
from sklearn.metrics import precision_recall_fscore_support
from collections import defaultdict

def evaluate_eventtype_and_domain(true_folder, pred_folder):
    """Returns event-type-wise and domain-wise scores for both Exact and IoU match."""
    true_files = {f for f in os.listdir(true_folder) if f.endswith("_extracted.txt")}
    pred_files = {f for f in os.listdir(pred_folder) if f.endswith("_extracted.txt")}
    common_files = true_files.intersection(pred_files)

    # Initialize counters
    eventtype_stats = {
        "Exact": defaultdict(Counter),
        "IoU": defaultdict(Counter)
    }
    domain_stats = {
        "Exact": defaultdict(Counter),
        "IoU": defaultdict(Counter)
    }

    for filename in common_files:
        true_path = os.path.join(true_folder, filename)
        pred_path = os.path.join(pred_folder, filename)
        
        true_data = read_extracted_sections(true_path)
        pred_data = read_extracted_sections(pred_path)

        for paper_code in true_data.keys():
            domain = paper_code.split("_")[0]

            for section in true_data[paper_code].keys():
                true_text = true_data[paper_code][section]
                pred_text = pred_data.get(paper_code, {}).get(section, "<NONE>")

                # Exact Match
                label_e = exact_match(true_text, pred_text, section, section)
                eventtype_stats["Exact"][section][label_e] += 1
                domain_stats["Exact"][domain][label_e] += 1

                # IoU Match
                label_i = iou_match(true_text, pred_text, section, section)
                eventtype_stats["IoU"][section][label_i] += 1
                domain_stats["IoU"][domain][label_i] += 1

    def compute_metrics(stats_dict):
        results = {}
        for key, counts in stats_dict.items():
            POS = counts["COR"] + counts["PAR"] + counts["INC"] + counts["MIS"]
            ACT = counts["COR"] + counts["PAR"] + counts["INC"] + counts["SPU"]

            precision = (counts["COR"] + 0.5 * counts["PAR"]) / ACT if ACT > 0 else 0
            recall = (counts["COR"] + 0.5 * counts["PAR"]) / POS if POS > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            results[key] = {"Precision": precision, "Recall": recall, "F1": f1, "Counts": dict(counts)}
        return results

    eventtype_metrics = {
        "Exact": compute_metrics(eventtype_stats["Exact"]),
        "IoU": compute_metrics(eventtype_stats["IoU"])
    }
    domain_metrics = {
        "Exact": compute_metrics(domain_stats["Exact"]),
        "IoU": compute_metrics(domain_stats["IoU"])
    }

    return eventtype_metrics, domain_metrics

def read_extracted_sections(file_path):
    """Reads an extracted text file and returns a dictionary mapping paper codes to sections."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    papers = {}
    current_paper = None
    sections = {"Background": "<NONE>", "Method": "<NONE>", "Results": "<NONE>", "Implications": "<NONE>"}
    
    for line in content.splitlines():
        line = line.strip()
        
        if line.startswith("Paper Code:"):
            if current_paper:
                papers[current_paper] = sections
            
            current_paper = line.split(": ")[-1]
            sections = {"Background": "<NONE>", "Method": "<NONE>", "Results": "<NONE>", "Implications": "<NONE>"}
        
        for section in sections.keys():
            if line.startswith(f"[{section}]:"):
                sections[section] = line[len(f"[{section}]: "):].strip()
    
    if current_paper:
        papers[current_paper] = sections
    
    return papers

def strict_match(true_text, pred_text, true_label, pred_label):
    """Strict evaluation: Both span and section label must match exactly."""
    if true_text.strip() == "<NONE>" and pred_text.strip() != "<NONE>":
        return "SPU"  # Spurious
    elif true_text.strip() != "<NONE>" and pred_text.strip() == "<NONE>":
        return "MIS"  # Missing
    elif true_text.strip() == pred_text.strip() and true_label == pred_label:
        return "COR"  # Correct
    return "INC"  # Incorrect in any other case

def exact_match(true_text, pred_text, true_label, pred_label):
    """Exact boundary match: Span must match exactly, type ignored."""
    if true_text.strip() == "<NONE>" and pred_text.strip() != "<NONE>":
        return "SPU"
    elif true_text.strip() != "<NONE>" and pred_text.strip() == "<NONE>":
        return "MIS"
    elif true_text.strip() == pred_text.strip():
        return "COR"
    return "INC"

def iou_match(true_text, pred_text, true_label, pred_label):
    """IoU span match (IoU > 0.5) + label match required."""
    true_tokens = set(true_text.strip().split())
    pred_tokens = set(pred_text.strip().split())

    if true_text.strip() == "<NONE>" and pred_text.strip() != "<NONE>":
        return "SPU"
    elif true_text.strip() != "<NONE>" and pred_text.strip() == "<NONE>":
        return "MIS"
    elif true_text.strip() == pred_text.strip() and true_label == pred_label:
        return "COR"
    
    overlap = true_tokens & pred_tokens
    union = true_tokens | pred_tokens
    iou = len(overlap) / len(union) if union else 0.0

    if iou > 0.5 and true_label == pred_label:
        return "COR"
    return "INC"

def partial_match(true_text, pred_text, true_label, pred_label):
    """Partial boundary match: Any overlap in spans counts as partially correct."""
    true_tokens = set(true_text.strip().split())
    pred_tokens = set(pred_text.strip().split())
    overlap = true_tokens & pred_tokens
    
    if true_text.strip() == "<NONE>" and pred_text.strip() != "<NONE>":
        return "SPU"
    elif true_text.strip() != "<NONE>" and pred_text.strip() == "<NONE>":
        return "MIS"
    elif true_text.strip() == pred_text.strip():
        return "COR"
    elif len(overlap) > 0:
        return "PAR"
    return "INC"

def type_match(true_text, pred_text, true_label, pred_label):
    """Type match: Some span overlap is required, but section label matters."""
    true_tokens = set(true_text.strip().split())
    pred_tokens = set(pred_text.strip().split())
    overlap = true_tokens & pred_tokens
    
    if true_text.strip() == "<NONE>" and pred_text.strip() != "<NONE>":
        return "SPU"
    elif true_text.strip() != "<NONE>" and pred_text.strip() == "<NONE>":
        return "MIS"
    elif true_text.strip() == pred_text.strip() and true_label == pred_label:
        return "COR"
    elif len(overlap) > 0 and true_label == pred_label:
        return "COR"
    return "INC"

def evaluate_folders(true_folder, pred_folder, strategy):
    """Evaluates LLM predictions against the ground truth using the given strategy."""
    true_files = {f for f in os.listdir(true_folder) if f.endswith("_extracted.txt")}
    pred_files = {f for f in os.listdir(pred_folder) if f.endswith("_extracted.txt")}
    common_files = true_files.intersection(pred_files)
    
    classifications = Counter()
    for filename in common_files:
        true_path = os.path.join(true_folder, filename)
        pred_path = os.path.join(pred_folder, filename)
        
        true_data = read_extracted_sections(true_path)
        pred_data = read_extracted_sections(pred_path)
        
        for paper_code in true_data.keys():
            for section in true_data[paper_code].keys():
                match_type = strategy(true_data[paper_code][section], pred_data.get(paper_code, {}).get(section, "<NONE>"), section, section)
                classifications[match_type] += 1
    
    # Compute Precision, Recall, and F1-score
    POS = classifications["COR"] + classifications["PAR"] + classifications["INC"] + classifications["MIS"]
    ACT = classifications["COR"] + classifications["PAR"] + classifications["INC"] + classifications["SPU"]
    
    precision = (classifications["COR"] + 0.5 * classifications["PAR"]) / ACT if ACT > 0 else 0
    recall = (classifications["COR"] + 0.5 * classifications["PAR"]) / POS if POS > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1_score,
        "Counts": classifications
    }

if __name__ == "__main__":
    # put the folder here, in each folder, txt files are required, format as below:
    """
    Paper Code: ACL_23_P_xxx

    [Background]: <EXACT TEXT or `<NONE>`>

    [Method]: <EXACT TEXT or `<NONE>`>

    [Results]: <EXACT TEXT or `<NONE>`>

    [Implications]: <EXACT TEXT or `<NONE>`>
    """

    

    # pred_folder = "baselines/task_1/data/pred_qwen"
    # pred_folder = "baselines/task_1/data/pred_llama"
    pred_folder = "baselines/task_1/data/pred_distill_llama"
    # pred_folder = "baselines/task_1/data/pred_gpt"
    pred_folder = "/home/bofudong/baselines/chunk_parsing_cross_check/tiyasha_txt_new"


    # pred_folder = "baselines/task_1/data/pred_qwen_oneshot"
    # pred_folder = "baselines/task_1/data/pred_llama_oneshot"
    # pred_folder = "baselines/task_1/data/pred_distill_llama_oneshot"

    true_folder = "baselines/task_1/data/true"

    strategies = {
        "Strict (EM + Class)": strict_match,
        "IoU + Class": iou_match
        # "Exact": exact_match,
        # "Partial": partial_match,
        # "Type": type_match
    }
    with open("baselines/task_1/tiyasha.txt", "w", encoding="utf-8") as f:
        # Overall results (multiple strategies)
        for strategy_name, strategy_fn in strategies.items():
            results = evaluate_folders(true_folder, pred_folder, strategy_fn)
            output = f"\n{strategy_name} Evaluation:\n"
            output += f"  Precision: {results['Precision']:.4f}\n"
            output += f"  Recall: {results['Recall']:.4f}\n"
            output += f"  F1-score: {results['F1-score']:.4f}\n"
            output += f"  Counts: {results['Counts']}\n"
            f.write(output)
            print(output)

        # Add event-type and domain-wise for Exact and IoU
        eventtype_metrics, domain_metrics = evaluate_eventtype_and_domain(true_folder, pred_folder)

        f.write("\n=== EVENT-TYPE-WISE PERFORMANCE ===\n")
        for match_type in ["Exact", "IoU"]:
            f.write(f"\n[{match_type} Match]\n")
            for section, metrics in eventtype_metrics[match_type].items():
                f.write(f"{section:12s} | P: {metrics['Precision']:.4f} R: {metrics['Recall']:.4f} F1: {metrics['F1']:.4f} | {metrics['Counts']}\n")

        f.write("\n=== DOMAIN-WISE PERFORMANCE ===\n")
        for match_type in ["Exact", "IoU"]:
            f.write(f"\n[{match_type} Match]\n")
            for domain, metrics in domain_metrics[match_type].items():
                f.write(f"{domain:12s} | P: {metrics['Precision']:.4f} R: {metrics['Recall']:.4f} F1: {metrics['F1']:.4f} | {metrics['Counts']}\n")

    # for strategy_name, strategy_fn in strategies.items():
    #     results = evaluate_folders(true_folder, pred_folder, strategy_fn)
    #     print(f"\n{strategy_name} Evaluation:")
    #     print(f"  Precision: {results['Precision']:.4f}")
    #     print(f"  Recall: {results['Recall']:.4f}")
    #     print(f"  F1-score: {results['F1-score']:.4f}")
    #     print(f"  Counts: {results['Counts']}")



# python baselines/task_1/F1_using_NER.py
