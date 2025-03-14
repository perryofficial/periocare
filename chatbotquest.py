import json

# Read the incorrectly formatted JSON file
with open("your_dataset.jsonl", "r", encoding="utf-8") as file:
    lines = file.readlines()  # Read line by line

# Convert each line into a proper JSON object
corrected_data = []
for line in lines:
    try:
        # Convert single JSON objects and fix keys
        obj = json.loads(line.strip())
        corrected_data.append({
            "instruction": obj.get("instruction (string)", ""),  # Fix key names
            "output": obj.get("output (string)", "")  # Fix key names
        })
    except json.JSONDecodeError as e:
        print(f"Skipping invalid line: {line}")

# Save as a properly formatted JSON array
with open("fixed_data.json", "w", encoding="utf-8") as file:
    json.dump(corrected_data, file, indent=4, ensure_ascii=False)

print("✅ JSON file has been corrected and saved as 'fixed_data.json'")
