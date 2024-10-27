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

# GSM8K Prompts
# Chain of Thought version
GSM8K_COT_SYSTEM = """You are a math expert specialized in solving GSM8K (Grade School Math 8K) problems. Break down each problem step by step, showing your calculations in <<>> tags, then provide your final answer after '####'."""

GSM8K_COT_PROMPT = """This is a GSM8K (Grade School Math 8K) word problem. Solve it step by step, showing your calculations in <<>> tags and give your final answer after '####':

{question}

Follow the GSM8K format:
1. Break down the problem into steps
2. Show each calculation step in <<>> tags (example: 5 × 4 = <<5*4=20>>20)
3. Write your final numeric answer after '####'"""

# Direct answer version
GSM8K_DIRECT_SYSTEM = """You are a mathematical problem solver. 
Always provide your final answer in the format: The answer is #### <number>
Do not show any work or explanation."""

# Or in your user prompt:
GSM8K_DIRECT_PROMPT = """Solve this math problem and provide only the numeric answer in the format 'The answer is #### <number>':

{question}"""
