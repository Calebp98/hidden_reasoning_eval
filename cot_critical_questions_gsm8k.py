import datasets
import anthropic
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
import pickle
import json
import os
import time
from tqdm import tqdm
from prompts import (
    GSM8K_COT_SYSTEM,
    GSM8K_DIRECT_SYSTEM,
    GSM8K_COT_PROMPT,
    GSM8K_DIRECT_PROMPT,
)
from anthropic.types.beta.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.beta.messages.batch_create_params import Request
from dotenv import load_dotenv

# Configuration Constants
MODEL = "claude-3-5-sonnet-20241022"
DATASET_NAME = "gsm8k"
NUM_TRIALS = 5  # Number of times to test each question
NUM_QUESTIONS = 1000
MAX_RETRIES = 3
POLL_INTERVAL = 5  # seconds



def setup_environment() -> Optional[str]:
    """Setup and validate environment variables."""
    load_dotenv()
    api_key = os.getenv("CLAUDE_API_KEY")
    if not api_key:
        print("Error: CLAUDE_API_KEY not found in environment variables")
        return None
    return api_key


def verify_prompts() -> bool:
    """Verify that all required prompts are loaded and non-empty."""
    required_prompts = [
        GSM8K_COT_SYSTEM,
        GSM8K_DIRECT_SYSTEM,
        GSM8K_COT_PROMPT,
        GSM8K_DIRECT_PROMPT,
    ]
    for prompt in required_prompts:
        if not prompt:
            print("Error: One or more required prompts are missing or empty")
            return False
    return True


def extract_answer(response: str) -> Optional[int]:
    """Extract numerical answer from Claude's response."""
    try:
        match = re.search(r"####\s*(\d+)", response)
        return int(match.group(1)) if match else None
    except (AttributeError, ValueError) as e:
        print(f"Error extracting answer from response: {str(e)}")
        return None


def create_all_requests(questions: List[Dict[str, Any]]) -> List[Request]:
    """Create a single list of batch requests for all questions and trials."""
    requests = []

    # Create direct answer requests
    for trial in range(NUM_TRIALS):
        for i, item in enumerate(questions):
            try:
                question = item["question"]
                custom_id = f"direct_q{i}_t{trial}"

                requests.append(
                    Request(
                        custom_id=custom_id,
                        params=MessageCreateParamsNonStreaming(
                            model=MODEL,
                            max_tokens=30,
                            system=GSM8K_DIRECT_SYSTEM,
                            messages=[
                                {
                                    "role": "user",
                                    "content": GSM8K_DIRECT_PROMPT.format(
                                        question=question
                                    ),
                                }
                            ],
                            temperature=0.7,
                        ),
                    )
                )
            except Exception as e:
                print(
                    f"Error creating direct request for question {i}, trial {trial}: {str(e)}"
                )
                continue

    # Create CoT requests
    for trial in range(NUM_TRIALS):
        for i, item in enumerate(questions):
            try:
                question = item["question"]
                custom_id = f"cot_q{i}_t{trial}"

                requests.append(
                    Request(
                        custom_id=custom_id,
                        params=MessageCreateParamsNonStreaming(
                            model=MODEL,
                            max_tokens=1024,
                            system=GSM8K_COT_SYSTEM,
                            messages=[
                                {
                                    "role": "user",
                                    "content": GSM8K_COT_PROMPT.format(
                                        question=question
                                    ),
                                }
                            ],
                            temperature=0.7,
                        ),
                    )
                )
            except Exception as e:
                print(
                    f"Error creating CoT request for question {i}, trial {trial}: {str(e)}"
                )
                continue

    return requests


def wait_for_batch(
    client: anthropic.Anthropic, batch_id: str, progress_bar: tqdm
) -> Dict[str, Any]:
    """Wait for a batch to complete and return results."""
    retry_count = 0

    while True:
        try:
            batch = client.beta.messages.batches.retrieve(batch_id)
            if batch.processing_status == "ended":
                results = {}
                for result in client.beta.messages.batches.results(batch_id):
                    try:
                        if result.result.type == "succeeded":
                            response = result.result.message.content[0].text
                            predicted_answer = extract_answer(response)
                            results[result.custom_id] = {
                                "predicted_answer": predicted_answer,
                                "claude_response": response,
                            }
                    except Exception as e:
                        print(f"Error processing result {result.custom_id}: {str(e)}")
                        continue
                return results

            time.sleep(POLL_INTERVAL)
            progress_bar.set_postfix({"status": "waiting for batch"})

        except Exception as e:
            retry_count += 1
            if retry_count >= MAX_RETRIES:
                print(
                    f"Failed to retrieve batch after {MAX_RETRIES} attempts: {str(e)}"
                )
                return {}
            print(f"Error retrieving batch (attempt {retry_count}): {str(e)}")
            time.sleep(POLL_INTERVAL)


def extract_answer_from_gsm8k(answer_text: str) -> Optional[int]:
    """Extract the numerical answer from GSM8K answer text format."""
    try:
        # Look for the last number after ####
        match = re.search(r"####\s*(\d+)", answer_text)
        if match:
            return int(match.group(1))
        # If no #### format, try to find the last number in the text
        numbers = re.findall(r"\d+", answer_text)
        if numbers:
            return int(numbers[-1])
        return None
    except (ValueError, AttributeError) as e:
        print(f"Error extracting answer from GSM8K text: {str(e)}")
        return None


def process_results(
    all_results: Dict[str, Any],
    test_questions: List[Dict[str, Any]],
    num_questions: int,
) -> List[Dict[str, Any]]:
    """Process batch results and calculate success rates."""
    final_results = []
    skipped_questions = 0


    for i in range(num_questions):
        try:
            # Extract correct answer from the GSM8K answer text
            correct_answer = extract_answer_from_gsm8k(test_questions[i]["answer"])
            if correct_answer is None:
                print(f"Warning: Could not extract answer for question {i}, skipping")
                skipped_questions += 1
                continue

            question_results = {
                "question": test_questions[i]["question"],
                "answer": correct_answer,
                "raw_answer": test_questions[i][
                    "answer"
                ],  # Store original answer for debugging
                "direct_trials": [],
                "cot_trials": [],
            }

            # Process trials
            valid_direct_trials = 0
            valid_cot_trials = 0
            direct_successes = 0
            cot_successes = 0

            for trial in range(NUM_TRIALS):
                direct_id = f"direct_q{i}_t{trial}"
                cot_id = f"cot_q{i}_t{trial}"

                # Process direct trials
                if direct_id in all_results:
                    trial_result = all_results[direct_id]
                    # claude_response = trial_result["claude_response"]
                    if trial_result["predicted_answer"] is not None:
                        question_results["direct_trials"].append(trial_result)
                        valid_direct_trials += 1
                        if trial_result["predicted_answer"] == correct_answer:
                            direct_successes += 1
                    else:
                        print(
                            f"Warning: Could not extract predicted answer for direct trial {trial} of question {i}"
                        )

                # Process CoT trials
                if cot_id in all_results:
                    trial_result = all_results[cot_id]
                    if trial_result["predicted_answer"] is not None:
                        question_results["cot_trials"].append(trial_result)
                        valid_cot_trials += 1
                        if trial_result["predicted_answer"] == correct_answer:
                            cot_successes += 1
                    else:
                        print(
                            f"Warning: Could not extract predicted answer for CoT trial {trial} of question {i}"
                        )

            # Calculate success rates only if we have valid trials
            question_results["direct_success_rate"] = (
                (direct_successes / valid_direct_trials * 100)
                if valid_direct_trials > 0
                else None
            )
            question_results["cot_success_rate"] = (
                (cot_successes / valid_cot_trials * 100)
                if valid_cot_trials > 0
                else None
            )

            # Add trial statistics
            question_results["stats"] = {
                "valid_direct_trials": valid_direct_trials,
                "valid_cot_trials": valid_cot_trials,
                "direct_successes": direct_successes,
                "cot_successes": cot_successes,
                "total_trials_attempted": NUM_TRIALS,
            }

            final_results.append(question_results)

        except Exception as e:
            print(f"Error processing results for question {i}: {str(e)}")
            skipped_questions += 1
            continue

    if skipped_questions > 0:
        print(
            f"\nWarning: Skipped {skipped_questions} out of {num_questions} questions due to errors"
        )

    return final_results


def save_results(folder_name: str, final_results: List[Dict[str, Any]]) -> None:
    """Save results and generate reports."""
    if not final_results:
        print("Warning: No results to save")
        return

    # Save full results
    with open(f"{folder_name}/reliability_results.pkl", "wb") as f:
        pickle.dump(final_results, f)

    # Save simplified results, handling None values
    simple_results = [
        {
            "question": result["question"],
            "answer": result["answer"],
            "direct_success_rate": result["direct_success_rate"]
            if result["direct_success_rate"] is not None
            else "N/A",
            "cot_success_rate": result["cot_success_rate"]
            if result["cot_success_rate"] is not None
            else "N/A",
            "stats": result["stats"],
        }
        for result in final_results
    ]

    with open(f"{folder_name}/results.json", "w", encoding="utf-8") as f:
        json.dump(simple_results, f, indent=2, ensure_ascii=False)

    # Calculate summary statistics, excluding None values
    valid_direct_rates = [
        r["direct_success_rate"]
        for r in final_results
        if r["direct_success_rate"] is not None
    ]
    valid_cot_rates = [
        r["cot_success_rate"]
        for r in final_results
        if r["cot_success_rate"] is not None
    ]

    total_trials = sum(r["stats"]["total_trials_attempted"] for r in final_results)
    valid_direct_trials = sum(r["stats"]["valid_direct_trials"] for r in final_results)
    valid_cot_trials = sum(r["stats"]["valid_cot_trials"] for r in final_results)

    summary_stats = {
        "num_questions": len(final_results),
        "trials_per_question": NUM_TRIALS,
        "total_trials_attempted": total_trials,
        "valid_direct_trials": valid_direct_trials,
        "valid_cot_trials": valid_cot_trials,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Only calculate averages if we have valid rates
    if valid_direct_rates:
        summary_stats["avg_direct_success_rate"] = sum(valid_direct_rates) / len(
            valid_direct_rates
        )
    if valid_cot_rates:
        summary_stats["avg_cot_success_rate"] = sum(valid_cot_rates) / len(
            valid_cot_rates
        )

    # Save summary stats
    with open(f"{folder_name}/log.txt", "w") as f:
        f.write(f"Analysis completed on: {summary_stats['timestamp']}\n")
        f.write(f"Number of questions tested: {summary_stats['num_questions']}\n")
        f.write(
            f"Trials per question attempted: {summary_stats['trials_per_question']}\n"
        )
        f.write(f"Total trials attempted: {summary_stats['total_trials_attempted']}\n")
        f.write(f"Valid direct trials: {summary_stats['valid_direct_trials']}\n")
        f.write(f"Valid CoT trials: {summary_stats['valid_cot_trials']}\n")

        if "avg_direct_success_rate" in summary_stats:
            f.write(
                f"Average direct success rate: {summary_stats['avg_direct_success_rate']:.1f}%\n"
            )
        else:
            f.write("Average direct success rate: N/A (no valid trials)\n")

        if "avg_cot_success_rate" in summary_stats:
            f.write(
                f"Average CoT success rate: {summary_stats['avg_cot_success_rate']:.1f}%\n"
            )
        else:
            f.write("Average CoT success rate: N/A (no valid trials)\n")

    print("\nFinal Statistics:")
    print("-" * 50)
    print(f"Total questions tested: {summary_stats['num_questions']}")
    print(f"Total trials attempted: {summary_stats['total_trials_attempted']}")
    print(f"Valid direct trials: {summary_stats['valid_direct_trials']}")
    print(f"Valid CoT trials: {summary_stats['valid_cot_trials']}")
    if "avg_direct_success_rate" in summary_stats:
        print(
            f"Average direct success rate: {summary_stats['avg_direct_success_rate']:.1f}%"
        )
    else:
        print("Average direct success rate: N/A (no valid trials)")
    if "avg_cot_success_rate" in summary_stats:
        print(f"Average CoT success rate: {summary_stats['avg_cot_success_rate']:.1f}%")
    else:
        print("Average CoT success rate: N/A (no valid trials)")
    print("-" * 50)


def main():
    """Main execution function."""
    try:
        # Setup and validation
        api_key = setup_environment()
        if not api_key:
            return

        if not verify_prompts():
            return

        # Initialize Claude client
        client = anthropic.Anthropic(api_key=api_key)

        # Create results folder
        folder_name = f"{DATASET_NAME}_{MODEL.replace('-', '_')}_reliability"
        os.makedirs(folder_name, exist_ok=True)

        # Load dataset
        try:
            dataset = datasets.load_dataset("gsm8k", "main")
            test_data = list(dataset["test"])
            test_questions = test_data[:NUM_QUESTIONS]
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            return

        # Create and process batch requests
        all_requests = create_all_requests(test_questions)
        if not all_requests:
            print("Error: No valid requests created")
            return

        # Process the batch
        progress_bar = tqdm(total=1, desc="Processing batch")
        try:
            batch = client.beta.messages.batches.create(requests=all_requests)
            all_results = wait_for_batch(client, batch.id, progress_bar)
            progress_bar.update(1)
        except Exception as e:
            print(f"Error processing batch: {str(e)}")
            return

        if not all_results:
            print("Error: No results received from batch processing")

            return

        with open(f"{folder_name}/transcript.json", "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        # Process and save results
        final_results = process_results(
            all_results, test_questions, len(test_questions)
        )
        save_results(folder_name, final_results)

    except Exception as e:
        print(f"An error occurred in main: {str(e)}")
        print("Traceback:")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()