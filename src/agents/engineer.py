from src.tools.llm_api_connector import ask_model_response


def run_engineer(dataset_info_text, explorer_output, model, critic_feedback=None):
    system_prompt = """
You are a strong ML engineer working on a tabular machine learning competition.

Write one complete Python script.

Installed libraries you may use:
- pandas
- numpy
- scikit-learn
- catboost
- xgboost
- requests

Use only these libraries.

Task rules:
- read data from:
  data/train.csv
  data/test.csv
  data/sample_submition.csv
- infer task type from dataset summary and explorer plan
- use 5-fold cross-validation
- print final cross-validation score exactly as:
  CV_SCORE=<number>
- save final submission to:
  artifacts/submissions/submission.csv
- return only pure Python code
- no markdown
- no explanations
- no backticks

Important preprocessing rules:
1. Parse last_dt as datetime.
2. Create one numeric feature from last_dt, for example days_since_last_review.
3. Use one fixed reference date from train only.
4. Drop original last_dt after feature extraction.
5. For every categorical column: fill missing values with 'MISSING' and cast to string.
6. For every numeric column: fill missing values with median.
7. Do not pass raw datetime columns into the model.
8. Keep the preprocessing simple and explicit.
9. Prefer CatBoost for this dataset.
10. Use verbose=False.
11. For final training on full data, do not use use_best_model=True unless eval_set is provided.
12. Keep the same working pipeline structure if improving a previous version.
13. Make minimal changes when applying critic feedback.

Modeling rules:
- If using CatBoost, explicitly define categorical columns.
- Do not use LightGBM or any other unavailable library.
- Do not build heavy NLP pipelines.
- name can be dropped if needed.
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
