import json

def filter_qa_by_success_rates(input_json_path, output_json_path=None):
    """
    Filter questions where CoT succeeded (100%) but direct approach failed (0%)
    
    Args:
        input_json_path (str): Path to input JSON file
        output_json_path (str, optional): Path for output JSON file.
                                        If None, adds '_cot_critical' before extension
    
    Returns:
        str: Path to the created filtered JSON file
        int: Number of questions that matched the criteria
    """
    # Generate output filename if not provided
    if output_json_path is None:
        base, ext = input_json_path.rsplit('.', 1)
        output_json_path = f"{base}_cot_critical.{ext}"
    
    # Read the input JSON file
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Filter questions based on success rates
    filtered_data = []
    for item in data:
        if (item.get('cot_success_rate') == 100.0 and 
            item.get('direct_success_rate') == 0.0):
            # Create new item with only required fields
            filtered_item = {
                'question': item.get('question'),
                'answer': item.get('answer'),
                'raw_answer': item.get('raw_answer')
            }
            filtered_data.append(filtered_item)
    
    # Save filtered data to new JSON file
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=2, ensure_ascii=False)
    
    return output_json_path, len(filtered_data)

def view_filtered_json(filename, num_samples=3):
    """
    View the contents of the filtered JSON file
    
    Args:
        filename (str): Path to JSON file
        num_samples (int): Number of samples to display
    """
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
        print(f"\nViewing {filename}")
        print(f"Total filtered items: {len(data)}")
        print(f"\nFirst {min(num_samples, len(data))} samples:")
        
        for i, item in enumerate(data[:num_samples]):
            print(f"\nItem {i+1}:")
            for key, value in item.items():
                print(f"{key}: {value}")
            print("-" * 50)

# Example usage
if __name__ == "__main__":
    # Filter the JSON file
    input_file = "reliability_results.json"
    output_file, num_filtered = filter_qa_by_success_rates(input_file)
    
    print(f"Found {num_filtered} questions where CoT succeeded but direct approach failed")
    print(f"Filtered data saved to: {output_file}")
    
    # View the filtered results
    view_filtered_json(output_file)