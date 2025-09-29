import json
import random
from collections import defaultdict
from pathlib import Path

def get_event_types(instance):
    """Return a set of event types present in the instance."""
    return set(ev["event_type"] for ev in instance.get("event_mentions", []))

def split_data_by_event_type(
    input_path: str,
    output_dir: str,
    seed: int = 42
):
    # Load .jsonl data
    with open(input_path, "r") as f:
        data = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(data)} instances from {input_path}")

    # Group instances by event type
    event_groups = defaultdict(list)
    for instance in data:
        event_types = get_event_types(instance)
        for et in event_types:
            event_groups[et].append(instance)

    print(f"Grouped into {len(event_groups)} event types")

    # Deduplication helper
    def dedup(instances):
        seen = set()
        unique = []
        for item in instances:
            key = item["wnd_id"]
            if key not in seen:
                seen.add(key)
                unique.append(item)
        return unique

    # Split per event type
    train_all, dev_all, test_all = [], [], []
    random.seed(seed)

    for event_type, instances in event_groups.items():
        unique_instances = dedup(instances)
        random.shuffle(unique_instances)
        total = len(unique_instances)

        # Use 8:1:1 ratio
        n_train = int(0.8 * total)
        n_dev = int(0.1 * total)
        n_test = total - n_train - n_dev

        train_all.extend(unique_instances[:n_train])
        dev_all.extend(unique_instances[n_train:n_train + n_dev])
        test_all.extend(unique_instances[n_train + n_dev:])

        print(f"[{event_type}] → {total} total | {n_train} train | {n_dev} dev | {n_test} test")

    # Final deduplication and shuffle
    train_all = dedup(train_all)
    dev_all = dedup(dev_all)
    test_all = dedup(test_all)
    random.shuffle(train_all)
    random.shuffle(dev_all)
    random.shuffle(test_all)

    # Save to files
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    def write_jsonl(instances, filename):
        with open(Path(output_dir) / filename, "w") as f:
            for item in instances:
                f.write(json.dumps(item) + "\n")

    write_jsonl(train_all, "train.json")
    write_jsonl(dev_all, "dev.json")
    write_jsonl(test_all, "test.json")

    print(f"\n Saved: {len(train_all)} train, {len(dev_all)} dev, {len(test_all)} test")

if __name__ == "__main__":
    split_data_by_event_type(
        input_path="SciEvent_data/DEGREE/processed/all_data.json",
        output_dir="SciEvent_data/DEGREE/all_splits"
    )
