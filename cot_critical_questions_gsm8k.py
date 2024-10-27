import datasets
import anthropic
import re
from typing import Dict
from datetime import datetime
import pickle
import os
from tqdm import tqdm
from prompts import (
    GSM8K_COT_SYSTEM,
    GSM8K_COT_PROMPT,
    GSM8K_DIRECT_SYSTEM,
    GSM8K_DIRECT_PROMPT,
)
from dotenv import load_dotenv

load_dotenv()
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")

MODEL = "claude-3-5-sonnet-20241022"
DATASET_NAME = "openai/gsm8k"


def get_answer_direct(question: str, client: anthropic.Anthropic) -> Dict:
    """Get Claude's direct answer for a question without Chain of Thought reasoning."""
    print("...getting direct answer")

    prompt = GSM8K_DIRECT_PROMPT.format(question=question)

    message = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=GSM8K_DIRECT_SYSTEM,
        messages=[
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "The answer is ####"},
        ],
        temperature=0.1,
    )

    response = message.content[0].text

    # Extract the numeric answer after ####
    match = re.search(r"####\s*(\d+)", response)
    predicted_answer = int(match.group(1)) if match else None

    return {
        "question": question,
        "claude_response": response,
        "predicted_answer": predicted_answer,
    }


def get_answer_cot(question: str, client: anthropic.Anthropic) -> Dict:
    """Get Claude's answer for a question using Chain of Thought reasoning."""
    print("...getting cot answer")

    prompt = GSM8K_COT_PROMPT.format(question=question)

    message = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=GSM8K_COT_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )

    response = message.content[0].text

    # Extract the numeric answer after ####
    match = re.search(r"####\s*(\d+)", response)
    predicted_answer = int(match.group(1)) if match else None

    return {
        "question": question,
        "claude_response": response,
        "predicted_answer": predicted_answer,
    }


def print_result(result):
    print("\nQuestion:", result["question"])
    print("Response:", result["claude_response"])
    print("Predicted Answer:", result["predicted_answer"])
    print("Correct Answer:", result["correct_answer"])
    print("Correct?", result["predicted_answer"] == result["correct_answer"])
    print("-" * 50)


def main():
    # Initialize Claude client
    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

    # Load dataset and get first n questions
    dataset = datasets.load_dataset("gsm8k", "main")
    test_data = list(dataset["test"])
    test_questions = test_data[:100]  # Adjust number as needed

    # Process each question
    direct_answerable = []
    cot_answerable = []
    unsolvable = []

    for item in tqdm(test_questions, desc="Processing questions"):
        # Extract question and answer
        question = item["question"]
        correct_answer = int(item["answer"].split("####")[1].strip())

        # Try direct answer first
        result = get_answer_direct(question, client)
        result["correct_answer"] = correct_answer
        print_result(result)

        if result["predicted_answer"] == result["correct_answer"]:
            direct_answerable.append(item)
        else:
            # Try with Chain of Thought
            result_cot = get_answer_cot(question, client)
            result_cot["correct_answer"] = correct_answer
            print_result(result_cot)

            if result_cot["predicted_answer"] == result_cot["correct_answer"]:
                cot_answerable.append(item)
            else:
                unsolvable.append(item)
                print("\nFailed to solve:")
                print_result(result_cot)

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
    with open(f"{folder_name}/log.txt", "w") as f:
        f.write(f"Dataset created on: {current_time}\n")
        f.write(f"Number of direct answerable: {len(direct_answerable)}\n")
        f.write(f"Number of CoT answerable: {len(cot_answerable)}\n")
        f.write(f"Number of unsolvable: {len(unsolvable)}\n")

    # Print statistics
    print("-" * 50)
    print("Model:", MODEL)
    print("-" * 50)
    print(f"Direct answerable questions: {len(direct_answerable)}")
    print(f"CoT answerable questions: {len(cot_answerable)}")
    print(f"Unsolvable questions: {len(unsolvable)}")
    print("-" * 50)


if __name__ == "__main__":
    main()
