import random
from inspect_ai.dataset import Sample, MemoryDataset, json_dataset, FieldSpec

# Load the dataset from the JSON file
single_question_dataset = json_dataset(
    "reliability_results_cot_critical.json",
    FieldSpec(input="question", target="answer", metadata=["raw_answer"]),
)


# Function to create a new dataset with paired questions
def create_paired_dataset(dataset, num_samples):
    if len(dataset) < 2:
        raise ValueError("The dataset must have at least two questions for pairing.")

    # Shuffle the dataset to ensure randomness
    question_indices = list(range(len(dataset)))
    random.shuffle(question_indices)

    # Create pairs without replacement
    samples = []
    for _ in range(num_samples):
        # Randomly select two questions without replacement
        selected_indices = random.sample(question_indices, 2)
        public_question = dataset[selected_indices[0]]
        secret_question = dataset[selected_indices[1]]

        # Create a Sample object
        sample = Sample(
            input=f"public_question: {public_question.input}\nsecret_question: {secret_question.input}",
            metadata={
                "public_question": public_question.input,
                "secret_question": secret_question.input,
                "public_answer": public_question.target,
                "secret_answer": secret_question.target,
                "public_raw_answer": public_question.metadata['raw_answer'],
                "secret_raw_answer": secret_question.metadata['raw_answer']
            },
            target=[
                public_question.target,
                secret_question.target,
            ],  # [public_answer, secret_answer]
        )
        samples.append(sample)

    # Return the constructed dataset
    return MemoryDataset(samples)


random.seed(42)  # For reproducibility
paired_dataset = create_paired_dataset(single_question_dataset, num_samples=300)

if __name__ == "__main__":
    print("-"*10)
    print(f"Dataset has {len(paired_dataset)} questions.")
    print("-"*10)

    # Print the created dataset samples
    for sample in paired_dataset[0:3]:
        print(sample)
        print("-"*10)

