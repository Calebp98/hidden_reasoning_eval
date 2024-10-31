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
                            temperature=0.1,
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
                            temperature=0.1,
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
# import datasets
# import anthropic
# import re
# from typing import Dict, List
# from datetime import datetime
# import pickle
# import json
# import os
# import time
# from tqdm import tqdm
# from prompts import (
#     GSM8K_COT_SYSTEM,
#     GSM8K_DIRECT_SYSTEM,
#     GSM8K_COT_PROMPT,
#     GSM8K_DIRECT_PROMPT,
# )
# from anthropic.types.beta.message_create_params import MessageCreateParamsNonStreaming
# from anthropic.types.beta.messages.batch_create_params import Request
# from dotenv import load_dotenv

# load_dotenv()
# CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")

# MODEL = "claude-3-5-sonnet-20241022"
# DATASET_NAME = "gsm8k"
# NUM_TRIALS = 2  # Number of times to test each question
# NUM_QUESTIONS = 2


# def create_all_requests(questions: List[Dict]) -> List[Request]:
#     """Create a single list of batch requests for all questions and trials."""
#     requests = []

#     # First create all direct answer requests
#     for trial in range(NUM_TRIALS):
#         for i, item in enumerate(questions):
#             try:
#                 question = item["question"]
#                 custom_id = f"direct_q{i}_t{trial}"

#                 requests.append(
#                     Request(
#                         custom_id=custom_id,
#                         params=MessageCreateParamsNonStreaming(
#                             model=MODEL,
#                             max_tokens=30,
#                             system=GSM8K_DIRECT_SYSTEM,
#                             messages=[
#                                 {
#                                     "role": "user",
#                                     "content": GSM8K_DIRECT_PROMPT.format(
#                                         question=question
#                                     ),
#                                 }
#                             ],
#                             temperature=0.1,
#                         ),
#                     )
#                 )
#             except Exception as e:
#                 print(
#                     f"Error creating direct request for question {i}, trial {trial}: {str(e)}"
#                 )
#                 continue

#     # Then create all CoT requests
#     for trial in range(NUM_TRIALS):
#         for i, item in enumerate(questions):
#             try:
#                 question = item["question"]
#                 custom_id = f"cot_q{i}_t{trial}"

#                 requests.append(
#                     Request(
#                         custom_id=custom_id,
#                         params=MessageCreateParamsNonStreaming(
#                             model=MODEL,
#                             max_tokens=1024,
#                             system=GSM8K_COT_SYSTEM,
#                             messages=[
#                                 {
#                                     "role": "user",
#                                     "content": GSM8K_COT_PROMPT.format(
#                                         question=question
#                                     ),
#                                 }
#                             ],
#                             temperature=0.1,
#                         ),
#                     )
#                 )
#             except Exception as e:
#                 print(
#                     f"Error creating CoT request for question {i}, trial {trial}: {str(e)}"
#                 )
#                 continue

#     return requests


# def wait_for_batch(
#     client: anthropic.Anthropic, batch_id: str, progress_bar: tqdm
# ) -> Dict:
#     """Wait for a batch to complete and return results."""
#     max_retries = 3
#     retry_count = 0

#     while True:
#         try:
#             batch = client.beta.messages.batches.retrieve(batch_id)
#             if batch.processing_status == "ended":
#                 results = {}
#                 for result in client.beta.messages.batches.results(batch_id):
#                     try:
#                         if result.result.type == "succeeded":
#                             response = result.result.message.content[0].text
#                             match = re.search(r"####\s*(\d+)", response)
#                             predicted_answer = int(match.group(1)) if match else None
#                             results[result.custom_id] = {
#                                 "predicted_answer": predicted_answer,
#                                 "claude_response": response,
#                             }
#                     except Exception as e:
#                         print(f"Error processing result {result.custom_id}: {str(e)}")
#                         continue
#                 return results

#             time.sleep(5)  # Poll every 5 seconds
#             progress_bar.set_postfix({"status": "waiting for batch"})

#         except Exception as e:
#             retry_count += 1
#             if retry_count >= max_retries:
#                 print(
#                     f"Failed to retrieve batch after {max_retries} attempts: {str(e)}"
#                 )
#                 return {}
#             print(f"Error retrieving batch (attempt {retry_count}): {str(e)}")
#             time.sleep(5)


# def main():
#     try:
#         # Initialize Claude client
#         client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

#         # Create results folder
#         folder_name = f"{DATASET_NAME}_{MODEL.replace('-', '_')}_reliability"
#         os.makedirs(folder_name, exist_ok=True)

#         # Load dataset and get questions
#         dataset = datasets.load_dataset("gsm8k", "main")
#         test_data = list(dataset["test"])
#         test_questions = test_data[:NUM_QUESTIONS]

#         # Create all requests in a single batch
#         all_requests = create_all_requests(test_questions)

#         # Process the single large batch
#         progress_bar = tqdm(total=1, desc="Processing batch")
#         batch = client.beta.messages.batches.create(requests=all_requests)
#         all_results = wait_for_batch(client, batch.id, progress_bar)
#         progress_bar.update(1)

#         # Process results
#         final_results = process_results(
#             all_results, test_questions, len(test_questions)
#         )

#         # Save all results and generate reports (rest of the code remains the same)
#         with open(f"{folder_name}/reliability_results.pkl", "wb") as f:
#             pickle.dump(final_results, f)

#         simple_results = [
#             {
#                 "question": result["question"],
#                 "direct_success_rate": result["direct_success_rate"],
#                 "cot_success_rate": result["cot_success_rate"],
#             }
#             for result in final_results
#         ]

#         with open(f"{folder_name}/results.json", "w", encoding="utf-8") as f:
#             json.dump(simple_results, f, indent=2, ensure_ascii=False)

#         # Calculate and save summary statistics
#         summary_stats = {
#             "avg_direct_success_rate": sum(
#                 r["direct_success_rate"] for r in final_results
#             )
#             / len(final_results),
#             "avg_cot_success_rate": sum(r["cot_success_rate"] for r in final_results)
#             / len(final_results),
#             "num_questions": len(final_results),
#             "trials_per_question": NUM_TRIALS,
#             "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#         }

#         # Save summary stats
#         with open(f"{folder_name}/log.txt", "w") as f:
#             f.write(f"Analysis completed on: {summary_stats['timestamp']}\n")
#             f.write(f"Number of questions tested: {summary_stats['num_questions']}\n")
#             f.write(f"Trials per question: {summary_stats['trials_per_question']}\n")
#             f.write(
#                 f"Average direct success rate: {summary_stats['avg_direct_success_rate']:.1f}%\n"
#             )
#             f.write(
#                 f"Average CoT success rate: {summary_stats['avg_cot_success_rate']:.1f}%\n"
#             )

#         print("\nFinal Statistics:")
#         print("-" * 50)
#         print(f"Total questions tested: {len(final_results)}")
#         print(f"Trials per question: {NUM_TRIALS}")
#         print(
#             f"Average direct success rate: {summary_stats['avg_direct_success_rate']:.1f}%"
#         )
#         print(f"Average CoT success rate: {summary_stats['avg_cot_success_rate']:.1f}%")
#         print("-" * 50)

#     except Exception as e:
#         print(f"An error occurred in main: {str(e)}")
#         print("Traceback:")
#         import traceback

#         traceback.print_exc()


# if __name__ == "__main__":
#     main()

# # import datasets
# # import anthropic
# # import re
# # from typing import Dict, List
# # from datetime import datetime
# # import pickle
# # import json
# # import os
# # import time
# # from tqdm import tqdm
# # from prompts import (
# #     GSM8K_COT_SYSTEM,
# #     GSM8K_DIRECT_SYSTEM,
# #     GSM8K_COT_PROMPT,
# #     GSM8K_DIRECT_PROMPT,
# # )
# # from anthropic.types.beta.message_create_params import MessageCreateParamsNonStreaming
# # from anthropic.types.beta.messages.batch_create_params import Request
# # from dotenv import load_dotenv

# # load_dotenv()
# # CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")

# # MODEL = "claude-3-5-sonnet-20241022"
# # DATASET_NAME = "gsm8k"
# # NUM_TRIALS = 2  # Number of times to test each question
# # NUM_QUESTIONS = 2


# # def create_all_requests(questions: List[Dict]) -> List[Request]:
# #     """Create a single list of batch requests for all questions and trials."""
# #     requests = []

# #     # First create all direct answer requests
# #     for trial in range(NUM_TRIALS):
# #         for i, item in enumerate(questions):
# #             try:
# #                 question = item["question"]
# #                 custom_id = f"direct_q{i}_t{trial}"

# #                 requests.append(
# #                     Request(
# #                         custom_id=custom_id,
# #                         params=MessageCreateParamsNonStreaming(
# #                             model=MODEL,
# #                             max_tokens=30,
# #                             system=GSM8K_DIRECT_SYSTEM,
# #                             messages=[
# #                                 {
# #                                     "role": "user",
# #                                     "content": GSM8K_DIRECT_PROMPT.format(
# #                                         question=question
# #                                     ),
# #                                 }
# #                             ],
# #                             temperature=0.1,
# #                         ),
# #                     )
# #                 )
# #             except Exception as e:
# #                 print(
# #                     f"Error creating direct request for question {i}, trial {trial}: {str(e)}"
# #                 )
# #                 continue

# #     # Then create all CoT requests
# #     for trial in range(NUM_TRIALS):
# #         for i, item in enumerate(questions):
# #             try:
# #                 question = item["question"]
# #                 custom_id = f"cot_q{i}_t{trial}"

# #                 requests.append(
# #                     Request(
# #                         custom_id=custom_id,
# #                         params=MessageCreateParamsNonStreaming(
# #                             model=MODEL,
# #                             max_tokens=1024,
# #                             system=GSM8K_COT_SYSTEM,
# #                             messages=[
# #                                 {
# #                                     "role": "user",
# #                                     "content": GSM8K_COT_PROMPT.format(
# #                                         question=question
# #                                     ),
# #                                 }
# #                             ],
# #                             temperature=0.1,
# #                         ),
# #                     )
# #                 )
# #             except Exception as e:
# #                 print(
# #                     f"Error creating CoT request for question {i}, trial {trial}: {str(e)}"
# #                 )
# #                 continue

# #     return requests


# # def wait_for_batch(
# #     client: anthropic.Anthropic, batch_id: str, progress_bar: tqdm
# # ) -> Dict:
# #     """Wait for a batch to complete and return results."""
# #     max_retries = 3
# #     retry_count = 0

# #     while True:
# #         try:
# #             batch = client.beta.messages.batches.retrieve(batch_id)
# #             if batch.processing_status == "ended":
# #                 results = {}
# #                 for result in client.beta.messages.batches.results(batch_id):
# #                     try:
# #                         if result.result.type == "succeeded":
# #                             response = result.result.message.content[0].text
# #                             match = re.search(r"####\s*(\d+)", response)
# #                             predicted_answer = int(match.group(1)) if match else None
# #                             results[result.custom_id] = {
# #                                 "predicted_answer": predicted_answer,
# #                                 "claude_response": response,
# #                             }
# #                     except Exception as e:
# #                         print(f"Error processing result {result.custom_id}: {str(e)}")
# #                         continue
# #                 return results

# #             time.sleep(5)  # Poll every 5 seconds
# #             progress_bar.set_postfix({"status": "waiting for batch"})

# #         except Exception as e:
# #             retry_count += 1
# #             if retry_count >= max_retries:
# #                 print(
# #                     f"Failed to retrieve batch after {max_retries} attempts: {str(e)}"
# #                 )
# #                 return {}
# #             print(f"Error retrieving batch (attempt {retry_count}): {str(e)}")
# #             time.sleep(5)


# # def main():
# #     try:
# #         # Initialize Claude client
# #         client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

# #         # Create results folder
# #         folder_name = f"{DATASET_NAME}_{MODEL.replace('-', '_')}_reliability"
# #         os.makedirs(folder_name, exist_ok=True)

# #         # Load dataset and get questions
# #         dataset = datasets.load_dataset("gsm8k", "main")
# #         test_data = list(dataset["test"])
# #         test_questions = test_data[:NUM_QUESTIONS]

# #         # Create all requests in a single batch
# #         all_requests = create_all_requests(test_questions)

# #         # Process the single large batch
# #         progress_bar = tqdm(total=1, desc="Processing batch")
# #         batch = client.beta.messages.batches.create(requests=all_requests)
# #         all_results = wait_for_batch(client, batch.id, progress_bar)
# #         progress_bar.update(1)

# #         # Save intermediate results
# #         save_intermediate_results(all_results, folder_name, 0)

# #         # Process results
# #         final_results = process_results(
# #             all_results, test_questions, len(test_questions)
# #         )

# #         # Save all results and generate reports (rest of the code remains the same)
# #         with open(f"{folder_name}/reliability_results.pkl", "wb") as f:
# #             pickle.dump(final_results, f)

# #         simple_results = [
# #             {
# #                 "question": result["question"],
# #                 "direct_success_rate": result["direct_success_rate"],
# #                 "cot_success_rate": result["cot_success_rate"],
# #             }
# #             for result in final_results
# #         ]

# #         with open(f"{folder_name}/results.json", "w", encoding="utf-8") as f:
# #             json.dump(simple_results, f, indent=2, ensure_ascii=False)

# #         # Calculate and save summary statistics
# #         summary_stats = {
# #             "avg_direct_success_rate": sum(
# #                 r["direct_success_rate"] for r in final_results
# #             )
# #             / len(final_results),
# #             "avg_cot_success_rate": sum(r["cot_success_rate"] for r in final_results)
# #             / len(final_results),
# #             "num_questions": len(final_results),
# #             "trials_per_question": NUM_TRIALS,
# #             "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
# #         }

# #         # Save summary stats
# #         with open(f"{folder_name}/log.txt", "w") as f:
# #             f.write(f"Analysis completed on: {summary_stats['timestamp']}\n")
# #             f.write(f"Number of questions tested: {summary_stats['num_questions']}\n")
# #             f.write(f"Trials per question: {summary_stats['trials_per_question']}\n")
# #             f.write(
# #                 f"Average direct success rate: {summary_stats['avg_direct_success_rate']:.1f}%\n"
# #             )
# #             f.write(
# #                 f"Average CoT success rate: {summary_stats['avg_cot_success_rate']:.1f}%\n"
# #             )

# #         print("\nFinal Statistics:")
# #         print("-" * 50)
# #         print(f"Total questions tested: {len(final_results)}")
# #         print(f"Trials per question: {NUM_TRIALS}")
# #         print(
# #             f"Average direct success rate: {summary_stats['avg_direct_success_rate']:.1f}%"
# #         )
# #         print(f"Average CoT success rate: {summary_stats['avg_cot_success_rate']:.1f}%")
# #         print("-" * 50)

# #     except Exception as e:
# #         print(f"An error occurred in main: {str(e)}")
# #         print("Traceback:")
# #         import traceback

# #         traceback.print_exc()


# # if __name__ == "__main__":
# #     main()


# # # import datasets
# # # import anthropic
# # # import re
# # # from typing import Dict, List
# # # from datetime import datetime
# # # import pickle
# # # import json
# # # import os
# # # import time
# # # from tqdm import tqdm
# # # from prompts import (
# # #     GSM8K_COT_SYSTEM,
# # #     GSM8K_COT_PROMPT,
# # #     GSM8K_DIRECT_SYSTEM,
# # #     GSM8K_DIRECT_PROMPT,
# # # )
# # # from anthropic.types.beta.message_create_params import MessageCreateParamsNonStreaming
# # # from anthropic.types.beta.messages.batch_create_params import Request
# # # from dotenv import load_dotenv

# # # load_dotenv()
# # # CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")

# # # MODEL = "claude-3-5-sonnet-20241022"
# # # DATASET_NAME = "gsm8k"
# # # NUM_TRIALS = 2  # Number of times to test each question
# # # NUM_QUESTIONS = 2


# # # def save_intermediate_results(results: Dict, folder_name: str, batch_num: int):
# # #     """Save intermediate results to prevent data loss."""
# # #     os.makedirs(folder_name, exist_ok=True)
# # #     with open(f"{folder_name}/intermediate_results_batch_{batch_num}.pkl", "wb") as f:
# # #         pickle.dump(results, f)


# # # def load_intermediate_results(folder_name: str) -> Dict:
# # #     """Load and combine all intermediate results."""
# # #     all_results = {}
# # #     for filename in os.listdir(folder_name):
# # #         if filename.startswith("intermediate_results_batch_") and filename.endswith(
# # #             ".pkl"
# # #         ):
# # #             with open(os.path.join(folder_name, filename), "rb") as f:
# # #                 batch_results = pickle.load(f)
# # #                 all_results.update(batch_results)
# # #     return all_results


# # # def clean_answer(answer: str) -> int:
# # #     """Clean and convert answer string to integer."""
# # #     try:
# # #         # First split by #### and take the second part
# # #         numeric_part = answer.split("####")[1].strip()
# # #         # Remove any commas and convert to int
# # #         numeric_part = numeric_part.replace(",", "")
# # #         return int(numeric_part)
# # #     except (IndexError, ValueError) as e:
# # #         raise ValueError(f"Could not parse answer: {answer}") from e


# # # def create_batch_requests(
# # #     questions: List[Dict], trial_num: int, is_cot: bool
# # # ) -> List[Request]:
# # #     """Create a list of batch requests for a set of questions."""
# # #     requests = []

# # #     for i, item in enumerate(questions):
# # #         try:
# # #             question = item["question"]
# # #             prompt = (
# # #                 GSM8K_COT_PROMPT.format(question=question)
# # #                 if is_cot
# # #                 else GSM8K_DIRECT_PROMPT.format(question=question)
# # #             )
# # #             system = GSM8K_COT_SYSTEM if is_cot else GSM8K_DIRECT_SYSTEM
# # #             max_tokens = 1024 if is_cot else 30

# # #             custom_id = f"{'cot' if is_cot else 'direct'}_q{i}_t{trial_num}"

# # #             requests.append(
# # #                 Request(
# # #                     custom_id=custom_id,
# # #                     params=MessageCreateParamsNonStreaming(
# # #                         model=MODEL,
# # #                         max_tokens=max_tokens,
# # #                         system=system,
# # #                         messages=[{"role": "user", "content": prompt}],
# # #                         temperature=0.1,
# # #                     ),
# # #                 )
# # #             )
# # #         except Exception as e:
# # #             print(f"Error creating request for question {i}: {str(e)}")
# # #             continue

# # #     return requests


# # # def wait_for_batch(
# # #     client: anthropic.Anthropic, batch_id: str, progress_bar: tqdm
# # # ) -> Dict:
# # #     """Wait for a batch to complete and return results."""
# # #     max_retries = 3
# # #     retry_count = 0

# # #     while True:
# # #         try:
# # #             batch = client.beta.messages.batches.retrieve(batch_id)
# # #             if batch.processing_status == "ended":
# # #                 results = {}
# # #                 for result in client.beta.messages.batches.results(batch_id):
# # #                     try:
# # #                         if result.result.type == "succeeded":
# # #                             response = result.result.message.content[0].text
# # #                             match = re.search(r"####\s*(\d+)", response)
# # #                             predicted_answer = int(match.group(1)) if match else None
# # #                             results[result.custom_id] = {
# # #                                 "predicted_answer": predicted_answer,
# # #                                 "claude_response": response,
# # #                             }
# # #                     except Exception as e:
# # #                         print(f"Error processing result {result.custom_id}: {str(e)}")
# # #                         continue
# # #                 return results

# # #             time.sleep(5)  # Poll every 5 seconds
# # #             progress_bar.set_postfix({"status": "waiting for batch"})

# # #         except Exception as e:
# # #             retry_count += 1
# # #             if retry_count >= max_retries:
# # #                 print(
# # #                     f"Failed to retrieve batch after {max_retries} attempts: {str(e)}"
# # #                 )
# # #                 return {}
# # #             print(f"Error retrieving batch (attempt {retry_count}): {str(e)}")
# # #             time.sleep(5)  # Wait before retrying


# # # def process_results(
# # #     all_results: Dict, questions: List[Dict], num_questions: int
# # # ) -> List[Dict]:
# # #     """Process batch results into final format."""
# # #     final_results = []

# # #     for i in range(num_questions):
# # #         try:
# # #             question = questions[i]["question"]
# # #             correct_answer = clean_answer(questions[i]["answer"])

# # #             # Count successes for direct and CoT approaches
# # #             direct_successes = sum(
# # #                 1
# # #                 for t in range(NUM_TRIALS)
# # #                 if f"direct_q{i}_t{t}" in all_results
# # #                 and all_results[f"direct_q{i}_t{t}"]["predicted_answer"]
# # #                 == correct_answer
# # #             )
# # #             cot_successes = sum(
# # #                 1
# # #                 for t in range(NUM_TRIALS)
# # #                 if f"cot_q{i}_t{t}" in all_results
# # #                 and all_results[f"cot_q{i}_t{t}"]["predicted_answer"] == correct_answer
# # #             )

# # #             # Collect all responses
# # #             direct_responses = [
# # #                 {
# # #                     "question": question,
# # #                     "claude_response": all_results.get(f"direct_q{i}_t{t}", {}).get(
# # #                         "claude_response", ""
# # #                     ),
# # #                     "predicted_answer": all_results.get(f"direct_q{i}_t{t}", {}).get(
# # #                         "predicted_answer", None
# # #                     ),
# # #                 }
# # #                 for t in range(NUM_TRIALS)
# # #                 if f"direct_q{i}_t{t}" in all_results
# # #             ]

# # #             cot_responses = [
# # #                 {
# # #                     "question": question,
# # #                     "claude_response": all_results.get(f"cot_q{i}_t{t}", {}).get(
# # #                         "claude_response", ""
# # #                     ),
# # #                     "predicted_answer": all_results.get(f"cot_q{i}_t{t}", {}).get(
# # #                         "predicted_answer", None
# # #                     ),
# # #                 }
# # #                 for t in range(NUM_TRIALS)
# # #                 if f"cot_q{i}_t{t}" in all_results
# # #             ]

# # #             final_results.append(
# # #                 {
# # #                     "question": question,
# # #                     "correct_answer": correct_answer,
# # #                     "direct_success_rate": (direct_successes / NUM_TRIALS) * 100,
# # #                     "cot_success_rate": (cot_successes / NUM_TRIALS) * 100,
# # #                     "direct_responses": direct_responses,
# # #                     "cot_responses": cot_responses,
# # #                 }
# # #             )
# # #         except Exception as e:
# # #             print(f"Error processing results for question {i}: {str(e)}")
# # #             continue

# # #     return final_results


# # # def main():
# # #     try:
# # #         # Initialize Claude client
# # #         client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

# # #         # Create results folder
# # #         folder_name = f"{DATASET_NAME}_{MODEL.replace('-', '_')}_reliability"
# # #         os.makedirs(folder_name, exist_ok=True)

# # #         # Load dataset and get first n questions
# # #         dataset = datasets.load_dataset("gsm8k", "main")
# # #         test_data = list(dataset["test"])
# # #         test_questions = test_data[:NUM_QUESTIONS]  # Adjust number as needed

# # #         # Try to load any existing intermediate results
# # #         all_results = load_intermediate_results(folder_name)

# # #         progress_bar = tqdm(total=2 * NUM_TRIALS, desc="Processing batches")

# # #         # Skip already completed trials
# # #         completed_trials = len(all_results) // (2 * len(test_questions))
# # #         progress_bar.update(completed_trials * 2)

# # #         # Process remaining direct and CoT trials in batches
# # #         for trial in range(completed_trials, NUM_TRIALS):
# # #             try:
# # #                 # Create and process direct batch
# # #                 direct_batch = client.beta.messages.batches.create(
# # #                     requests=create_batch_requests(test_questions, trial, is_cot=False)
# # #                 )
# # #                 direct_results = wait_for_batch(client, direct_batch.id, progress_bar)
# # #                 all_results.update(direct_results)
# # #                 save_intermediate_results(all_results, folder_name, 2 * trial)
# # #                 progress_bar.update(1)

# # #                 # Create and process CoT batch
# # #                 cot_batch = client.beta.messages.batches.create(
# # #                     requests=create_batch_requests(test_questions, trial, is_cot=True)
# # #                 )
# # #                 cot_results = wait_for_batch(client, cot_batch.id, progress_bar)
# # #                 all_results.update(cot_results)
# # #                 save_intermediate_results(all_results, folder_name, 2 * trial + 1)
# # #                 progress_bar.update(1)

# # #             except Exception as e:
# # #                 print(f"Error processing trial {trial}: {str(e)}")
# # #                 continue

# # #         # Process results
# # #         final_results = process_results(
# # #             all_results, test_questions, len(test_questions)
# # #         )

# # #         # Save the detailed results as pickle
# # #         with open(f"{folder_name}/reliability_results.pkl", "wb") as f:
# # #             pickle.dump(final_results, f)

# # #         # Save the simplified results as JSON
# # #         simple_results = [
# # #             {
# # #                 "question": result["question"],
# # #                 "direct_success_rate": result["direct_success_rate"],
# # #                 "cot_success_rate": result["cot_success_rate"],
# # #             }
# # #             for result in final_results
# # #         ]

# # #         with open(f"{folder_name}/results.json", "w", encoding="utf-8") as f:
# # #             json.dump(simple_results, f, indent=2, ensure_ascii=False)

# # #         # Calculate summary statistics
# # #         summary_stats = {
# # #             "avg_direct_success_rate": sum(
# # #                 r["direct_success_rate"] for r in final_results
# # #             )
# # #             / len(final_results),
# # #             "avg_cot_success_rate": sum(r["cot_success_rate"] for r in final_results)
# # #             / len(final_results),
# # #             "num_questions": len(final_results),
# # #             "trials_per_question": NUM_TRIALS,
# # #             "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
# # #         }

# # #         # Save summary stats to log file
# # #         with open(f"{folder_name}/log.txt", "w") as f:
# # #             f.write(f"Analysis completed on: {summary_stats['timestamp']}\n")
# # #             f.write(f"Number of questions tested: {summary_stats['num_questions']}\n")
# # #             f.write(f"Trials per question: {summary_stats['trials_per_question']}\n")
# # #             f.write(
# # #                 f"Average direct success rate: {summary_stats['avg_direct_success_rate']:.1f}%\n"
# # #             )
# # #             f.write(
# # #                 f"Average CoT success rate: {summary_stats['avg_cot_success_rate']:.1f}%\n"
# # #             )

# # #         # Clean up intermediate results
# # #         for filename in os.listdir(folder_name):
# # #             if filename.startswith("intermediate_results_batch_"):
# # #                 os.remove(os.path.join(folder_name, filename))

# # #         # Print final statistics
# # #         print("\nFinal Statistics:")
# # #         print("-" * 50)
# # #         print(f"Total questions tested: {len(final_results)}")
# # #         print(f"Trials per question: {NUM_TRIALS}")
# # #         print(
# # #             f"Average direct success rate: {summary_stats['avg_direct_success_rate']:.1f}%"
# # #         )
# # #         print(f"Average CoT success rate: {summary_stats['avg_cot_success_rate']:.1f}%")
# # #         print("-" * 50)

# # #     except Exception as e:
# # #         print(f"An error occurred in main: {str(e)}")
# # #         print("Traceback:")
# # #         import traceback

# # #         traceback.print_exc()


# # # if __name__ == "__main__":
# # #     main()


# # # # import datasets
# # # # import anthropic
# # # # import re
# # # # from typing import Dict, List
# # # # from datetime import datetime
# # # # import pickle
# # # # import json
# # # # import os
# # # # import time
# # # # from tqdm import tqdm
# # # # from prompts import (
# # # #     GSM8K_COT_SYSTEM,
# # # #     GSM8K_COT_PROMPT,
# # # #     GSM8K_DIRECT_SYSTEM,
# # # #     GSM8K_DIRECT_PROMPT,
# # # # )
# # # # from anthropic.types.beta.message_create_params import MessageCreateParamsNonStreaming
# # # # from anthropic.types.beta.messages.batch_create_params import Request
# # # # from dotenv import load_dotenv

# # # # load_dotenv()
# # # # CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")

# # # # MODEL = "claude-3-5-sonnet-20241022"
# # # # DATASET_NAME = "gsm8k"
# # # # NUM_TRIALS = 10  # Number of times to test each question
# # # # NUM_QUESTIONS = 200


# # # # def create_batch_requests(
# # # #     questions: List[Dict], trial_num: int, is_cot: bool
# # # # ) -> List[Request]:
# # # #     """Create a list of batch requests for a set of questions."""
# # # #     requests = []

# # # #     for i, item in enumerate(questions):
# # # #         question = item["question"]
# # # #         prompt = (
# # # #             GSM8K_COT_PROMPT.format(question=question)
# # # #             if is_cot
# # # #             else GSM8K_DIRECT_PROMPT.format(question=question)
# # # #         )
# # # #         system = GSM8K_COT_SYSTEM if is_cot else GSM8K_DIRECT_SYSTEM
# # # #         max_tokens = 1024 if is_cot else 30

# # # #         custom_id = f"{'cot' if is_cot else 'direct'}_q{i}_t{trial_num}"

# # # #         requests.append(
# # # #             Request(
# # # #                 custom_id=custom_id,
# # # #                 params=MessageCreateParamsNonStreaming(
# # # #                     model=MODEL,
# # # #                     max_tokens=max_tokens,
# # # #                     system=system,
# # # #                     messages=[{"role": "user", "content": prompt}],
# # # #                     temperature=0.1,
# # # #                 ),
# # # #             )
# # # #         )

# # # #     return requests


# # # # def wait_for_batch(
# # # #     client: anthropic.Anthropic, batch_id: str, progress_bar: tqdm
# # # # ) -> Dict:
# # # #     """Wait for a batch to complete and return results."""
# # # #     while True:
# # # #         batch = client.beta.messages.batches.retrieve(batch_id)
# # # #         if batch.processing_status == "ended":
# # # #             results = {}
# # # #             for result in client.beta.messages.batches.results(batch_id):
# # # #                 if result.result.type == "succeeded":
# # # #                     # Extract the numeric answer after ####
# # # #                     response = result.result.message.content[0].text
# # # #                     match = re.search(r"####\s*(\d+)", response)
# # # #                     predicted_answer = int(match.group(1)) if match else None
# # # #                     results[result.custom_id] = {
# # # #                         "predicted_answer": predicted_answer,
# # # #                         "claude_response": response,
# # # #                     }
# # # #             return results
# # # #         time.sleep(5)  # Poll every 5 seconds
# # # #         progress_bar.set_postfix({"status": "waiting for batch"})


# # # # def process_results(
# # # #     all_results: Dict, questions: List[Dict], num_questions: int
# # # # ) -> List[Dict]:
# # # #     """Process batch results into final format."""
# # # #     final_results = []

# # # #     for i in range(num_questions):
# # # #         question = questions[i]["question"]
# # # #         correct_answer = int(questions[i]["answer"].split("####")[1].strip())

# # # #         # Count successes for direct and CoT approaches
# # # #         direct_successes = sum(
# # # #             1
# # # #             for t in range(NUM_TRIALS)
# # # #             if all_results[f"direct_q{i}_t{t}"]["predicted_answer"] == correct_answer
# # # #         )
# # # #         cot_successes = sum(
# # # #             1
# # # #             for t in range(NUM_TRIALS)
# # # #             if all_results[f"cot_q{i}_t{t}"]["predicted_answer"] == correct_answer
# # # #         )

# # # #         # Collect all responses
# # # #         direct_responses = [
# # # #             {
# # # #                 "question": question,
# # # #                 "claude_response": all_results[f"direct_q{i}_t{t}"]["claude_response"],
# # # #                 "predicted_answer": all_results[f"direct_q{i}_t{t}"][
# # # #                     "predicted_answer"
# # # #                 ],
# # # #             }
# # # #             for t in range(NUM_TRIALS)
# # # #         ]

# # # #         cot_responses = [
# # # #             {
# # # #                 "question": question,
# # # #                 "claude_response": all_results[f"cot_q{i}_t{t}"]["claude_response"],
# # # #                 "predicted_answer": all_results[f"cot_q{i}_t{t}"]["predicted_answer"],
# # # #             }
# # # #             for t in range(NUM_TRIALS)
# # # #         ]

# # # #         final_results.append(
# # # #             {
# # # #                 "question": question,
# # # #                 "correct_answer": correct_answer,
# # # #                 "direct_success_rate": (direct_successes / NUM_TRIALS) * 100,
# # # #                 "cot_success_rate": (cot_successes / NUM_TRIALS) * 100,
# # # #                 "direct_responses": direct_responses,
# # # #                 "cot_responses": cot_responses,
# # # #             }
# # # #         )

# # # #     return final_results


# # # # def main():
# # # #     # Initialize Claude client
# # # #     client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

# # # #     # Load dataset and get first n questions
# # # #     dataset = datasets.load_dataset("gsm8k", "main")
# # # #     test_data = list(dataset["test"])
# # # #     test_questions = test_data[:NUM_QUESTIONS]  # Adjust number as needed

# # # #     all_results = {}
# # # #     progress_bar = tqdm(total=2 * NUM_TRIALS, desc="Processing batches")

# # # #     # Process direct and CoT trials in batches
# # # #     for trial in range(NUM_TRIALS):
# # # #         # Create and process direct batch
# # # #         direct_batch = client.beta.messages.batches.create(
# # # #             requests=create_batch_requests(test_questions, trial, is_cot=False)
# # # #         )
# # # #         direct_results = wait_for_batch(client, direct_batch.id, progress_bar)
# # # #         all_results.update(direct_results)
# # # #         progress_bar.update(1)

# # # #         # Create and process CoT batch
# # # #         cot_batch = client.beta.messages.batches.create(
# # # #             requests=create_batch_requests(test_questions, trial, is_cot=True)
# # # #         )
# # # #         cot_results = wait_for_batch(client, cot_batch.id, progress_bar)
# # # #         all_results.update(cot_results)
# # # #         progress_bar.update(1)

# # # #     # Process results
# # # #     final_results = process_results(all_results, test_questions, len(test_questions))

# # # #     # Create folder name using dataset and model
# # # #     folder_name = f"{DATASET_NAME}_{MODEL.replace('-', '_')}_reliability"
# # # #     os.makedirs(folder_name, exist_ok=True)

# # # #     # Save the detailed results as pickle
# # # #     with open(f"{folder_name}/reliability_results.pkl", "wb") as f:
# # # #         pickle.dump(final_results, f)

# # # #     # Save the simplified results as JSON
# # # #     simple_results = [
# # # #         {
# # # #             "question": result["question"],
# # # #             "direct_success_rate": result["direct_success_rate"],
# # # #             "cot_success_rate": result["cot_success_rate"],
# # # #         }
# # # #         for result in final_results
# # # #     ]

# # # #     with open(f"{folder_name}/results.json", "w", encoding="utf-8") as f:
# # # #         json.dump(simple_results, f, indent=2, ensure_ascii=False)

# # # #     # Calculate summary statistics
# # # #     summary_stats = {
# # # #         "avg_direct_success_rate": sum(r["direct_success_rate"] for r in final_results)
# # # #         / len(final_results),
# # # #         "avg_cot_success_rate": sum(r["cot_success_rate"] for r in final_results)
# # # #         / len(final_results),
# # # #         "num_questions": len(final_results),
# # # #         "trials_per_question": NUM_TRIALS,
# # # #         "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
# # # #     }

# # # #     # Save summary stats to log file
# # # #     with open(f"{folder_name}/log.txt", "w") as f:
# # # #         f.write(f"Analysis completed on: {summary_stats['timestamp']}\n")
# # # #         f.write(f"Number of questions tested: {summary_stats['num_questions']}\n")
# # # #         f.write(f"Trials per question: {summary_stats['trials_per_question']}\n")
# # # #         f.write(
# # # #             f"Average direct success rate: {summary_stats['avg_direct_success_rate']:.1f}%\n"
# # # #         )
# # # #         f.write(
# # # #             f"Average CoT success rate: {summary_stats['avg_cot_success_rate']:.1f}%\n"
# # # #         )

# # # #     # Print final statistics
# # # #     print("\nFinal Statistics:")
# # # #     print("-" * 50)
# # # #     print(f"Total questions tested: {len(final_results)}")
# # # #     print(f"Trials per question: {NUM_TRIALS}")
# # # #     print(
# # # #         f"Average direct success rate: {summary_stats['avg_direct_success_rate']:.1f}%"
# # # #     )
# # # #     print(f"Average CoT success rate: {summary_stats['avg_cot_success_rate']:.1f}%")
# # # #     print("-" * 50)


# # # # if __name__ == "__main__":
# # # #     main()

# # # # # import datasets
# # # # # import anthropic
# # # # # import re
# # # # # from typing import Dict, List
# # # # # from datetime import datetime
# # # # # import pickle
# # # # # import json
# # # # # import os
# # # # # from tqdm import tqdm
# # # # # from prompts import (
# # # # #     GSM8K_COT_SYSTEM,
# # # # #     GSM8K_COT_PROMPT,
# # # # #     GSM8K_DIRECT_SYSTEM,
# # # # #     GSM8K_DIRECT_PROMPT,
# # # # # )
# # # # # from dotenv import load_dotenv

# # # # # load_dotenv()
# # # # # CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")

# # # # # MODEL = "claude-3-5-sonnet-20241022"
# # # # # DATASET_NAME = "gsm8k"
# # # # # NUM_TRIALS = 10 # Number of times to test each question
# # # # # NUM_QUESTIONS = 200

# # # # # def get_answer_direct(question: str, client: anthropic.Anthropic) -> Dict:
# # # # #     """Get Claude's direct answer for a question without Chain of Thought reasoning."""
# # # # #     prompt = GSM8K_DIRECT_PROMPT.format(question=question)

# # # # #     message = client.messages.create(
# # # # #         model=MODEL,
# # # # #         max_tokens=30,
# # # # #         system=GSM8K_DIRECT_SYSTEM,
# # # # #         messages=[
# # # # #             {"role": "user", "content": prompt},
# # # # #         ],
# # # # #         temperature=0.7,
# # # # #     )

# # # # #     response = message.content[0].text

# # # # #     # Extract the numeric answer after ####
# # # # #     match = re.search(r"####\s*(\d+)", response)
# # # # #     predicted_answer = int(match.group(1)) if match else None

# # # # #     return {
# # # # #         "question": question,
# # # # #         "claude_response": response,
# # # # #         "predicted_answer": predicted_answer,
# # # # #     }


# # # # # def get_answer_cot(question: str, client: anthropic.Anthropic) -> Dict:
# # # # #     """Get Claude's answer for a question using Chain of Thought reasoning."""
# # # # #     prompt = GSM8K_COT_PROMPT.format(question=question)

# # # # #     message = client.messages.create(
# # # # #         model=MODEL,
# # # # #         max_tokens=1024,
# # # # #         system=GSM8K_COT_SYSTEM,
# # # # #         messages=[{"role": "user", "content": prompt}],
# # # # #         temperature=0.7,
# # # # #     )

# # # # #     response = message.content[0].text

# # # # #     # Extract the numeric answer after ####
# # # # #     match = re.search(r"####\s*(\d+)", response)
# # # # #     predicted_answer = int(match.group(1)) if match else None

# # # # #     return {
# # # # #         "question": question,
# # # # #         "claude_response": response,
# # # # #         "predicted_answer": predicted_answer,
# # # # #     }


# # # # # def test_question_reliability(
# # # # #     question: str, correct_answer: int, client: anthropic.Anthropic
# # # # # ) -> Dict:
# # # # #     """Test a question multiple times with both direct and CoT approaches."""
# # # # #     direct_successes = 0
# # # # #     cot_successes = 0

# # # # #     direct_responses = []
# # # # #     cot_responses = []

# # # # #     # Test direct approach
# # # # #     for _ in range(NUM_TRIALS):
# # # # #         result = get_answer_direct(question, client)
# # # # #         if result["predicted_answer"] == correct_answer:
# # # # #             direct_successes += 1
# # # # #         direct_responses.append(result)

# # # # #     # Test CoT approach
# # # # #     for _ in range(NUM_TRIALS):
# # # # #         result = get_answer_cot(question, client)
# # # # #         if result["predicted_answer"] == correct_answer:
# # # # #             cot_successes += 1
# # # # #         cot_responses.append(result)

# # # # #     return {
# # # # #         "question": question,
# # # # #         "correct_answer": correct_answer,
# # # # #         "direct_success_rate": direct_successes / NUM_TRIALS * 100,
# # # # #         "cot_success_rate": cot_successes / NUM_TRIALS * 100,
# # # # #         "direct_responses": direct_responses,
# # # # #         "cot_responses": cot_responses,
# # # # #     }


# # # # # def main():
# # # # #     # Initialize Claude client
# # # # #     client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

# # # # #     # Load dataset and get first n questions
# # # # #     dataset = datasets.load_dataset("gsm8k", "main")
# # # # #     test_data = list(dataset["test"])
# # # # #     test_questions = test_data[:NUM_QUESTIONS]  # Adjust number as needed

# # # # #     # Process each question
# # # # #     results = []
# # # # #     simple_results = []  # For JSON output

# # # # #     progress_bar = tqdm(test_questions, desc="Testing questions")
# # # # #     for item in progress_bar:
# # # # #         # Extract question and answer
# # # # #         question = item["question"]
# # # # #         correct_answer = int(item["answer"].split("####")[1].strip())

# # # # #         # Test reliability
# # # # #         result = test_question_reliability(question, correct_answer, client)
# # # # #         results.append(result)

# # # # #         # Store simplified result for JSON
# # # # #         simple_results.append(
# # # # #             {
# # # # #                 "question": question,
# # # # #                 "direct_success_rate": result["direct_success_rate"],
# # # # #                 "cot_success_rate": result["cot_success_rate"],
# # # # #             }
# # # # #         )

# # # # #         # Update progress bar description
# # # # #         progress_bar.set_postfix(
# # # # #             {
# # # # #                 "Direct": f"{result['direct_success_rate']:.1f}%",
# # # # #                 "CoT": f"{result['cot_success_rate']:.1f}%",
# # # # #             }
# # # # #         )

# # # # #     # Create folder name using dataset and model
# # # # #     folder_name = f"{DATASET_NAME}_{MODEL.replace('-', '_')}_reliability"
# # # # #     os.makedirs(folder_name, exist_ok=True)

# # # # #     # Save the detailed results as pickle
# # # # #     with open(f"{folder_name}/reliability_results.pkl", "wb") as f:
# # # # #         pickle.dump(results, f)

# # # # #     # Save the simplified results as JSON
# # # # #     with open(f"{folder_name}/results.json", "w", encoding="utf-8") as f:
# # # # #         json.dump(simple_results, f, indent=2, ensure_ascii=False)

# # # # #     # Calculate and save summary statistics
# # # # #     summary_stats = {
# # # # #         "avg_direct_success_rate": sum(r["direct_success_rate"] for r in results)
# # # # #         / len(results),
# # # # #         "avg_cot_success_rate": sum(r["cot_success_rate"] for r in results)
# # # # #         / len(results),
# # # # #         "num_questions": len(results),
# # # # #         "trials_per_question": NUM_TRIALS,
# # # # #         "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
# # # # #     }

# # # # #     # Save summary stats to log file
# # # # #     with open(f"{folder_name}/log.txt", "w") as f:
# # # # #         f.write(f"Analysis completed on: {summary_stats['timestamp']}\n")
# # # # #         f.write(f"Number of questions tested: {summary_stats['num_questions']}\n")
# # # # #         f.write(f"Trials per question: {summary_stats['trials_per_question']}\n")
# # # # #         f.write(
# # # # #             f"Average direct success rate: {summary_stats['avg_direct_success_rate']:.1f}%\n"
# # # # #         )
# # # # #         f.write(
# # # # #             f"Average CoT success rate: {summary_stats['avg_cot_success_rate']:.1f}%\n"
# # # # #         )

# # # # #     # Print final statistics
# # # # #     print("\nFinal Statistics:")
# # # # #     print("-" * 50)
# # # # #     print(f"Total questions tested: {len(results)}")
# # # # #     print(f"Trials per question: {NUM_TRIALS}")
# # # # #     print(
# # # # #         f"Average direct success rate: {summary_stats['avg_direct_success_rate']:.1f}%"
# # # # #     )
# # # # #     print(f"Average CoT success rate: {summary_stats['avg_cot_success_rate']:.1f}%")
# # # # #     print("-" * 50)


# # # # # if __name__ == "__main__":
# # # # #     main()
