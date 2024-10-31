import pickle


def view_pickle(filename, num_samples=3):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        print(f"\nViewing {filename}")
        print(f"Total items: {len(data)}")
        print("\nFirst {num_samples} samples:")
        for i, item in enumerate(data[:num_samples]):
            print(f"\nItem {i+1}:")
            for key, value in item.items():
                print(f"{key}: {value}")
            print("-" * 50)
            print("-" * 50)

# View all three files

# filename = "MMLU-Pro_claude_3_5_sonnet_20241022/"
# filename = "gsm8k_claude_3_5_sonnet_20241022/"

# view_pickle(filename + 'direct_answerable.pkl')
# view_pickle(filename + 'cot_answerable.pkl')
# view_pickle(filename + 'unsolvable.pkl')

filename = "gsm8k_claude_3_5_sonnet_20241022_reliability/"

view_pickle(filename + 'reliability_results.pkl')