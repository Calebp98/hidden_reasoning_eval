from inspect_ai import Task, task, Epochs
from inspect_ai.dataset import FieldSpec, json_dataset
from inspect_ai.scorer import Score, scorer, accuracy, stderr
from inspect_ai.solver import generate, prompt_template, system_message
import re

@scorer(metrics=[accuracy(), stderr()])
def gsm8k_answer_match():
    """Score exact matches between integer answers, specifically looking for #### format."""
    
    def extract_answer(text: str) -> int | None:
        """Extract the number after #### from text."""
        # Look for #### followed by a number
        match = re.search(r'####\s*(\d+)', text)
        if match:
            return int(match.group(1))
        return None

    async def score(state, target):
        # Extract integers from both the model output and target
        model_answer = extract_answer(state.output.completion)
        correct_answer = int(target.text) if target.text.strip().isdigit() else None
        
        # Handle cases where we couldn't extract an integer
        if model_answer is None or correct_answer is None:
            return Score(
                value="I",
                answer=state.output.completion,  # Store full response for debugging
                explanation=f"Failed to extract valid integer. Model output: '{state.output.completion}', Target: '{target.text}'"
            )
        
        # Compare the integers
        is_correct = model_answer == correct_answer
        
        return Score(
            value="C" if is_correct else "I",
            answer=str(model_answer),
            explanation=f"Model answered {model_answer}, correct answer was {correct_answer}"
        )
    
    return score

# Define the specific GSM8K prompts
GSM8K_DIRECT_SYSTEM = """You are a mathematical problem solver. 
Always provide your final answer in the format: The answer is #### <number>. Make sure to include ####
Do not show any work or explanation."""

GSM8K_DIRECT_PROMPT = """Solve this math problem and provide only the numeric answer in the format:
{prompt}"""

@task
def gsm8k_evaluation():
    """Evaluate model's ability to solve GSM8K problems with direct answers."""
    return Task(
        dataset=json_dataset(
            "reliability_results_cot_critical.json",
            FieldSpec(
                input="question", 
                target="answer",
                metadata=["raw_answer"]
            ),
        ),
        solver=[
            system_message(GSM8K_DIRECT_SYSTEM),
            prompt_template(GSM8K_DIRECT_PROMPT),
            generate()
        ],
        scorer=gsm8k_answer_match(),
        # Run each question 10 times and use various aggregation metrics
        epochs=Epochs(
            10, 
            ["mean", "mode", "at_least_1", "at_least_5"]
        )
    )

# Example usage:
# from inspect_ai import eval
# results = eval(gsm8k_evaluation(), model="anthropic/claude-3-sonnet-20240229")