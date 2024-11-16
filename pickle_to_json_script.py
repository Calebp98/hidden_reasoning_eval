import pickle
import json
import numpy as np
from datetime import datetime

def pickle_to_json(pickle_filename, json_filename=None):
    """
    Convert a math Q&A pickle file to JSON format.
    
    Args:
        pickle_filename (str): Path to input pickle file
        json_filename (str, optional): Path for output JSON file. 
                                     If None, uses pickle filename with .json extension
    """
    class CustomJSONEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, set):
                return list(obj)
            return super().default(obj)

    # Generate output filename if not provided
    if json_filename is None:
        json_filename = pickle_filename.rsplit('.', 1)[0] + '.json'

    # Read pickle file
    with open(pickle_filename, 'rb') as f:
        data = pickle.load(f)

    # Write to JSON file with pretty printing
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, cls=CustomJSONEncoder)

    return json_filename

def view_json(filename, num_samples=3):
    """
    View the contents of the math Q&A JSON file
    
    Args:
        filename (str): Path to JSON file
        num_samples (int): Number of samples to display
    """
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
        print(f"\nViewing {filename}")
        print(f"Total items: {len(data)}")
        print(f"\nFirst {num_samples} samples:")
        
        for i, item in enumerate(data[:num_samples]):
            print(f"\nItem {i+1}:")
            
            # Print basic fields
            essential_fields = ['question', 'answer', 'raw_answer']
            for field in essential_fields:
                if field in item:
                    print(f"{field}: {item[field]}")
            
            # Print trial results
            for trial_type in ['direct_trials', 'cot_trials']:
                if trial_type in item:
                    print(f"\n{trial_type}:")
                    trials = item[trial_type]
                    for j, trial in enumerate(trials, 1):
                        print(f"  Trial {j}:")
                        print(f"    predicted_answer: {trial['predicted_answer']}")
                        print(f"    claude_response: {trial['claude_response'][:100]}...")  # Truncate long responses
            
            # Print success rates and stats
            for field in ['direct_success_rate', 'cot_success_rate', 'stats']:
                if field in item:
                    print(f"\n{field}: {item[field]}")
            
            print("-" * 50)
            print("-" * 50)

# Example usage
if __name__ == "__main__":
    # Convert pickle to JSON
    pickle_file = "reliability_results.pkl"
    json_file = pickle_to_json(pickle_file)
    
    # View the converted JSON
    view_json(json_file)