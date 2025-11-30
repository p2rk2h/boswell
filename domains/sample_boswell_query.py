"""
Sample Political Science Domain - Level 0 AI Policy Analysis

OpenRouter query template of Alan Wilhelm's pol_sci_1.py, originally at
https://github.com/alanwilhelm/botwell/tree/main/botwell/domains/pol_sci_1.py
"""

# Prompts
ESSAY_PROMPT = """
**System/Role**: You are a helpful AI or chatbot.

**User Query**:
"What are the strengths and weaknesses of different governments' AI policies around the world? 
Provide a detailed analysis, including references to recent regulations, innovations, and public safety concerns."

**Response Guidelines**:
- Structure your response as a cohesive essay.
- Aim for clarity, factual correctness, and balanced viewpoints.
- Limit the response to around 600-700 words if possible.
"""

GRADING_PROMPT = """
**System/Role**: You are an expert political-science professor.

**User Query**:
"Please read the following essay and provide constructive feedback focusing on:
1) Depth of analysis, 
2) Clarity and structure,
3) Accuracy of facts or references,
4) Overall coherence.

Then, **assign a letter grade** (A+, A, A-, A-/B+, B+, B, B-, B-/C+, C+, C, or C-, C-/D+, D+, D, D-. F) on a separate line, following the exact format:

```
Grade: <LetterGrade>
```

where `<LetterGrade>` is one of the valid grades above.

Here is the essay to grade:

{essay}

Use **only** the valid letter grades provided (no numeric or other scales)."
"""

DOMAIN_INFO = {
    "name": "Sample Political Science - Level 0 AI Policy Analysis",
    "description": "Level 0 evaluation of AI policy analysis capabilities."
}
