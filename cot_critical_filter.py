# extracts questions that need CoT to be answered

import json

def extract_questions(input_file, output_file):
    # Read the JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Filter questions based on criteria
    filtered_questions = [
        item for item in data 
        if item["direct_success_rate"] == 0.0 and item["cot_success_rate"] == 100.0
    ]
    
    # Write filtered questions to new JSON file
    with open(output_file, 'w') as f:
        json.dump(filtered_questions, f, indent=2)
    
    # Print summary
    print(f"Found {len(filtered_questions)} questions matching the criteria")
    print("\nExtracted questions:")
    for item in filtered_questions:
        print(f"- {item['question']}")

# Usage
input_file = "gsm8k_claude_3_5_sonnet_20241022_reliability/results.json"  # Replace with your input file name
output_file = input_file.replace('.json', '_cot_critical.json')

extract_questions(input_file, output_file)