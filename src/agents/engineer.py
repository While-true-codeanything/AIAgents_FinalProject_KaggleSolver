from src.config import ensure_directories, CONFIG
from src.tools.llm_api_connector import ask_model_response


def run_engineer(dataset_info_text, explorer_output, model, submission_output_path, critic_feedback=None):
    ensure_directories()

    system_prompt = system_prompt = f"""
You are a strong ML engineer working on a tabular machine learning competition.

Write one complete Python script.

Allowed libraries:
- pandas
- numpy
- scikit-learn
- catboost
- xgboost
- requests

Data:
- data/train.csv
- data/test.csv
- data/sample_submition.csv
- artifacts/data_splits/train_inner.csv
- artifacts/data_splits/valid_holdout.csv

Pipeline structure (STRICT):
- load data
- define preprocess()
- apply preprocess() to train_inner, valid_holdout, train_full, test
- define feature_columns once
- use same feature_columns everywhere
- train model
- evaluate
- retrain on full data
- predict test
- save submission

Do not change pipeline structure. Modify only small parts.

Preprocessing:
- Fit ONLY on train_inner
- Save stats (medians, rare mapping, reference date)
- Reuse same stats for validation and test
- Do NOT recompute stats on validation/test
- Parse last_dt → create numeric feature → drop original
- Fill categorical NaN with 'MISSING' and cast to str
- Fill numeric NaN with median
- Do not pass datetime columns into model

Feature consistency:
- Same columns for train/valid/test
- Do not use dropped columns

Validation:
- Use {CONFIG['run']['main_metric']} metric
- Train on train_inner, evaluate on valid_holdout
- Do not create new splits
- Print exactly: SCORE=<number>


Submission rules:
- After local validation, fit final model on full data/train.csv
- Then predict on data/test.csv
- Build submission strictly from data/sample_submition.csv/ Use it structure and data/test.csv predictions
- Do not use test["_id"] as submission ID
- Preserve the first column from sample_submition.csv exactly as-is
- Fill only the prediction column


Submission rules:
- After local validation, fit final model on full data/train.csv
- Predict test.csv
- Build submission strictly from data/sample_submition.csv/ Use it structure and data/test.csv predictions
- Do NOT use test["_id"] as submission ID
- Preserve the first column from sample_submition.csv exactly as-is. Fill only the prediction column. Resulting df should have same column names and number as sample_submition.
- Save to: {submission_output_path}

Modeling:
- Prefer CatBoost
- Define categorical columns explicitly
- No LightGBM or other libraries
- No heavy NLP
- verbose=False

Runtime:
- Hard limit: {CONFIG['run']['executor_timeout']} sec
- Target: {CONFIG['run']['executor_timeout'] - 60} sec This includes preprocessing, local validation, final training, and submission generation
- Keep model simple, no hyperparameter search. Or conduct it in described time limit
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
        timeout=180,
    )

    code_text = response["text"].strip()

    if code_text.startswith("```python"):
        code_text = code_text[len("```python"):].strip()
    elif code_text.startswith("```"):
        code_text = code_text[len("```"):].strip()

    if code_text.endswith("```"):
        code_text = code_text[:-3].strip()

    return code_text