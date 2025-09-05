import json

test_file = "./aaa_my_data_conversion/splitted_data/test.oneie.json"  # <- change this to your actual test file path
total_arguments = 0
event_count = 0

with open(test_file, "r") as f:
    for line in f:
        example = json.loads(line)
        for event in example.get("event_mentions", []):
            total_arguments += len(event.get("arguments", []))
            event_count += 1

print(f"Total argument roles: {total_arguments}")
print(f"Total event mentions: {event_count}")
