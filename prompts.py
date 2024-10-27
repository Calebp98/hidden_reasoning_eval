# Chain of Thought version
COT_SYSTEM = """You are a knowledge expert. Use step-by-step reasoning to explain your thought process, then give your final answer as 'The answer is (X)'."""

COT_PROMPT = """Q: {question}

Options are:
{options}

Think step-by-step before you answer this question:
1. First, let's understand what the question is asking
2. Let's analyze each option and explain why it's right or wrong
3. Based on this analysis, provide the final answer as 'The answer is (X)'"""

# Direct answer version
DIRECT_SYSTEM = """You are a knowledge expert. Answer the multiple-choice question and provide your final answer as 'The answer is (X)'."""

DIRECT_PROMPT = """Q: {question}

Options are:
{options}

Provide your answer as 'The answer is (X)'"""
