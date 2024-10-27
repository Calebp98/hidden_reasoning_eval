import datasets
import anthropic
import re
from typing import List, Dict
from prompts import COT_PROMPT, COT_SYSTEM, DIRECT_PROMPT, DIRECT_SYSTEM
import pickle
import os
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")

# MODEL = "claude-3-haiku-20240307"
MODEL = "claude-3-5-sonnet-20241022"
DATASET_NAME = "MMLU-Pro"  # remember to also change dataset loading line (ctrl-f "dataset = datasets")


def form_options(options: List[str]) -> str:
    """Format multiple choice options."""
    return "\n".join(
        f"({letter}): {option}" for letter, option in zip("ABCDEFGHIJ", options)
    )


def get_answer_direct(
    question: str, options: List[str], client: anthropic.Anthropic
) -> Dict:
    """Get Claude's direct answer for a question without Chain of Thought reasoning."""

    def form_options(opts):
        return "\n".join(f"{chr(65 + i)}) {opt}" for i, opt in enumerate(opts))

    prompt = DIRECT_PROMPT.format(question=question, options=form_options(options))

    # print("prompt", DIRECT_SYSTEM, prompt)

    message = client.messages.create(
        model=MODEL,
        max_tokens=30,
        system=DIRECT_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )

    response = message.content[0].text
    match = re.search(r"answer is \(?([A-J])\)?", response)
    predicted_answer = match.group(1) if match else None

    return {
        "question": question,
        "options": options,
        "claude_response": response,
        "predicted_answer": predicted_answer,
    }


def get_answer_cot(
    question: str, options: List[str], client: anthropic.Anthropic
) -> Dict:
    """Get Claude's answer for a question using Chain of Thought reasoning."""

    def form_options(opts):
        return "\n".join(f"{chr(65 + i)}) {opt}" for i, opt in enumerate(opts))

    prompt = COT_PROMPT.format(question=question, options=form_options(options))

    message = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=COT_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )

    response = message.content[0].text
    match = re.search(r"answer is \(?([A-J])\)?", response)
    predicted_answer = match.group(1) if match else None

    return {
        "question": question,
        "options": options,
        "claude_response": response,
        "predicted_answer": predicted_answer,
    }


def print_result(result):
    # Print results for each question
    print("\nQuestion Category:", result["category"])
    print("Question:", result["question"])
    print("Response:", result["claude_response"])
    print("Predicted Answer:", result["predicted_answer"])
    print("Correct Answer:", result["correct_answer"])
    print("Correct?", result["predicted_answer"] == result["correct_answer"])
    print("-" * 50)


def main():
    # Initialize Claude client
    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

    # Load dataset and get first n questions
    dataset = datasets.load_dataset(
        "TIGER-Lab/MMLU-Pro"
    )  # change for different dataset
    test_data = list(dataset["test"])  # Convert to list to make it indexable
    test_questions = test_data[:100]  # modify for more of MMLU

    # Process each question
    direct_answerable = []  # Can be answered directly
    cot_answerable = []  # Needs CoT to answer
    unsolvable = []  # Can't be answered even with CoT

    for question in tqdm(test_questions, desc="Processing questions"):
        result = get_answer_direct(question["question"], question["options"], client)
        result["correct_answer"] = question["answer"]
        result["category"] = question["category"]

        if result["predicted_answer"] == result["correct_answer"]:
            direct_answerable.append(question)
        else:
            result_cot = get_answer_cot(
                question["question"], question["options"], client
            )
            result_cot["correct_answer"] = question["answer"]
            result_cot["category"] = question["category"]

            # print("CoT attempt")
            # print_result(result_cot)

            if result_cot["predicted_answer"] == result_cot["correct_answer"]:
                cot_answerable.append(question)
            else:
                unsolvable.append(question)

    # Create folder name using dataset and model
    folder_name = f"{DATASET_NAME}_{MODEL.replace('-', '_')}"
    os.makedirs(folder_name, exist_ok=True)

    # Save the datasets in the folder
    with open(f"{folder_name}/direct_answerable.pkl", "wb") as f:
        pickle.dump(direct_answerable, f)

    with open(f"{folder_name}/cot_answerable.pkl", "wb") as f:
        pickle.dump(cot_answerable, f)

    with open(f"{folder_name}/unsolvable.pkl", "wb") as f:
        pickle.dump(unsolvable, f)

    # Save date and time to log file
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(f"{folder_name}/log.txt", "w") as f:  # Note: "w" not "wb" for text files
        f.write(f"Dataset created on: {current_time}\n")
        f.write(f"Number of direct answerable: {len(direct_answerable)}\n")
        f.write(f"Number of CoT answerable: {len(cot_answerable)}\n")
        f.write(f"Number of unsolvable: {len(unsolvable)}\n")

    # Print statistics
    print("-" * 50)
    print("Model: ", MODEL)
    print("-" * 50)
    print(f"\nDirect answerable questions: {len(direct_answerable)}")
    print(f"CoT answerable questions: {len(cot_answerable)}")
    print(f"Unsolvable questions: {len(unsolvable)}")
    print("-" * 50)


if __name__ == "__main__":
    main()
