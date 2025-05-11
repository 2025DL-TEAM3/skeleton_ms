# system prompt: 너가 누구이고, 무슨 과제를 수행해야 하는지+왜(why)를 명확히
system_prompt = (
    "You are an expert at solving ARC puzzles. "
    "You will be given several input/output grid pairs labeled 'Example 1', 'Example 2', etc. "
    "Your task is to infer *why* each input grid becomes its output grid—by analyzing row-wise and column-wise relationships, "
    "color frequencies, and shape movements—and then *apply* that same transformation rule to a new test input grid."
)

# user prompt 1: 무엇을 해야 하는지 + how to study
user_message_template1 = (
    "Below are {n} labeled example input/output grid pair{plural}:\n"
    "- Each grid contains digits (0-9) representing cell values\n"
    "- Rows are separated by new lines\n"
    "- Each digit represents one cell, no spaces between digits\n"
    "- Examples are labeled 'Example 1:', 'Example 2:', etc.\n"
    "- Carefully analyze how each 'Example i Input' transforms into its 'Example i Output', "
    "noting any row/column shifts, color pattern changes, or shape translations."
)

# user prompt 2: 테스트 예제를 어떻게 적용할 것인지
user_message_template2 = (
    "Now apply the same inferred transformation rule to the following test input grid, labeled 'Test Input:' below."
)

# user prompt 3: 출력 형식 강제
user_message_template3 = (
    "Please output *only* the resulting grid:\n"
    "- Each digit (0-9) represents a cell value\n"
    "- Rows separated by new lines\n"
    "- No spaces between digits in a row\n"
    "Do not include any labels, explanation, or extra text."
)