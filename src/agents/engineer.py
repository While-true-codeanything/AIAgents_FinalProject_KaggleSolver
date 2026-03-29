from src.config import ensure_directories
from src.tools.llm_api_connector import ask_model_response


def run_engineer(dataset_info_text, explorer_output, model, submission_output_path, critic_feedback=None):
    ensure_directories()

    system_prompt = f"""
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

Data files:
- data/train.csv
- data/test.csv
- data/sample_submition.csv
- artifacts/data_splits/train_inner.csv
- artifacts/data_splits/valid_holdout.csv

Task rules:
- infer task type from dataset summary and explorer plan
- for local validation, train on artifacts/data_splits/train_inner.csv
- evaluate on artifacts/data_splits/valid_holdout.csv
- print validation score exactly as:
  CV_SCORE=<number>
- save final submission to:
  artifacts/submissions/submission.csv
- return only pure Python code
- no markdown
- no explanations
- no backticks

Validation rules:
- Do not create new train/validation splits inside the script.
- Use only:
  - artifacts/data_splits/train_inner.csv
  - artifacts/data_splits/valid_holdout.csv
- The validation score must be computed on valid_holdout only.
- save submission to this exact path: {submission_output_path}

Submission rules:
- After local validation, fit final model on full data/train.csv
- Then predict on data/test.csv
- Build submission strictly from data/sample_submition.csv/ Use it structure and data/test.csv predictions
- Save submission to this exact path:
  {submission_output_path}
- Do not use test["_id"] as submission ID
- Preserve the first column from sample_submition.csv exactly as-is
- Fill only the prediction column
- Validate that:
  - submission row count matches sample_submition.csv
  - first submission column has no duplicates
  - first submission column has no missing values

Important preprocessing rules:
1. Parse last_dt as datetime.
2. Create one numeric feature from last_dt, for example days_since_last_review.
3. Use one fixed reference date from train only.
4. Drop original last_dt after feature extraction.
5. For every categorical column: fill missing values with 'MISSING' and cast to string.
6. For every numeric column: fill missing values with median.
7. Do not pass raw datetime columns into the model.
8. Keep preprocessing simple and explicit.
9. Prefer CatBoost for this dataset.
10. Use verbose=False.
11. For final training on full data, do not use use_best_model=True unless eval_set is provided.
12. Keep the same working pipeline structure if improving a previous version.
13. Make minimal changes when applying critic feedback.

Runtime constraint:
- The hard timeout is 300 seconds.
- Aim to finish safely within 200-250 seconds total.
- This includes preprocessing, local validation, final training, and submission generation.
- Avoid large models, expensive feature engineering, and hyperparameter search.
- If using CatBoost or XGBoost, choose parameters that are likely to finish within the time budget.

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

Submission output path for this iteration:
{submission_output_path}

"""

    if critic_feedback:
        user_prompt += f"""

Critic feedback:
{critic_feedback}

Modify the previous working solution with minimal changes.
Do not rewrite the whole pipeline from scratch unless necessary.
"""

    response = ask_model_response(
        promt=user_prompt,
        system_prompt=system_prompt,
        model=model,
        temperature=0.15,
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