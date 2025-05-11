# system prompt
system_prompt = (
    "You are an expert at solving ARC puzzles. "
    "You will be given several labeled example input/output grid pairs. "
    "Learn the transformation rule from those examples, "
    "then apply it to the test input grid."
)

# user prompt 1
user_message_template1 = (
    "Here are {n} example input/output pair{plural}:\n"
    "- Digits 0–9 represent cell values\n"
    "- Rows separated by newline characters\n"
    "- No spaces between digits in a row\n"
    "- Examples labeled 'Example 1:', 'Example 2:', etc.\n"
    "Study how each input becomes its output."
)

# user prompt 2
user_message_template2 = (
    "Now apply the learned rule to this test input grid (labeled 'Test Input:'):"
)

# user prompt 3: 출력 형식 강제
user_message_template3 = (
    "Output format example: "
    "1234\n"
    "5678\n"
    "Use line breaks at the end of each row. "
    "Only output digits and line breaks for each row; no extra blank lines, no spaces, no labels, no explanation."
)