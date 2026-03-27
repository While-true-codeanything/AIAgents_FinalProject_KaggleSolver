from src.tools.llm_api_connector import ask_model_response


def run_engineer(dataset_info_text, explorer_output, model, critic_feedback=None):
    system_prompt = """
You are a strong ML engineer working on a tabular machine learning competition.

Your task:
- write a full Python script
- use pandas, numpy, scikit-learn, catboost
- read data from:
  data/train.csv
  data/test.csv
  data/submission_sample.csv
- build a baseline solution
- infer whether the task is classification or regression from target statistics and explorer plan
- create simple, practical preprocessing
- use cross-validation on train
- print validation score in this exact format:
  CV_SCORE=<number>
- save submission to:
  artifacts/submissions/submission.csv

Rules:
- return only pure Python code
- no markdown
- no explanations
- no backticks
- code must be complete and runnable
- keep the solution simple and robust
"""

    user_prompt = f"""
Dataset summary:
{dataset_info_text}

Explorer plan:
{explorer_output}
"""

    if critic_feedback:
        user_prompt += f"""

Critic feedback from previous iteration:
{critic_feedback}

Update the previous baseline and improve it according to the feedback.
"""

    response = ask_model_response(
        user_prompt,
        model=model,
        system_prompt=system_prompt,
        temperature=0.2,
        max_tokens=5000,
        timeout=120,
    )

    code_text = response["text"].strip()

    if code_text.startswith("```python"):
        code_text = code_text[len("```python"):].strip()
    elif code_text.startswith("```"):
        code_text = code_text[len("```"):].strip()

    if code_text.endswith("```"):
        code_text = code_text[:-3].strip()

    return code_text
