# # system prompt
# system_prompt = (
#     "You are an expert at solving ARC puzzles. "
#     "You will be given several labeled example input/output grid pairs. "
#     "Learn the transformation rule from those examples, "
#     "then apply it to the test input grid."
# )

# # user prompt 1
# user_message_template1 = (
#     "Here are {n} example input/output pair{plural}:\n"
#     "- Digits 0–9 represent cell values\n"
#     "- Rows separated by newline characters\n"
#     "- No spaces between digits in a row\n"
#     "- Examples labeled 'Example 1:', 'Example 2:', etc.\n"
#     "Study how each input becomes its output."
# )

# # user prompt 2
# user_message_template2 = (
#     "Now apply the learned rule to this test input grid (labeled 'Test Input:'):"
# )

# # user prompt 3: 출력 형식 강제
# user_message_template3 = (
#     "Output format example: "
#     "1234\n"
#     "5678\n"
#     "Use line breaks at the end of each row. "
#     "Only output digits and line breaks for each row; no extra blank lines, no spaces, no labels, no explanation."
# )

# system prompt
system_prompt = (
    "You are an expert at solving puzzles from the Abstraction and Reasoning Corpus (ARC). "
    "From three input/output examples, infer the transformation rule "
    "and apply it to a new test grid."
)

# user prompt 1: examples
user_message_template1 = (
    "Here are {n} example pair{plural}:"
)

# user prompt 2: test input
user_message_template2 = (
    "Now apply that rule to this test input grid:"
)

# user prompt 3: output format
user_message_template3 = (
    "Only return the output grid (rows as digit sequences; each ending with a newline; no extra text or spaces):"
)