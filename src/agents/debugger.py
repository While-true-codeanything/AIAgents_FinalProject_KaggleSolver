from src.config import CONFIG
from src.tools.code_executor import clean_code_text
from src.tools.llm_api_connector import ask_model_response


def run_debugger(dataset_info_text, explorer_output, code_text, execution_result, model):
    system_prompt = f"""
You are a Python ML debugging agent.

Your only task is to fix runtime errors in the provided code.

Rules:
- fix the code with minimal changes
- do not redesign the pipeline
- do not change the overall modeling approach unless it is strictly necessary to remove the error
- preserve the existing structure as much as possible
- keep the same data paths
- keep the same submission logic
- return only pure Python code
- no markdown
- no explanations
- no backticks

CRITICAL:
- Return ONLY valid Python code
- Do not include any explanations
- Do not include any text before or after code
- Do not explain the error

Submission rules:
- After local validation, fit final model on full data/train.csv
- Predict test.csv
- Build submission strictly from data/sample_submition.csv/ Use it structure and data/test.csv predictions
- Do NOT use test["_id"] as submission ID
- Preserve the first column from sample_submition.csv exactly as-is. Fill only the prediction column. Format: index,prediction
- Fill only the prediction column
  
Validation rules:
- Use {CONFIG['run']['main_metric']} as validation metric
- Do not create new train/validation splits inside the script.


If unsure, keep original code and fix only the broken lines
Focus only on making the script run successfully.
Do not try to improve model quality unless required to fix the error.
"""

    stdout_text = execution_result.get("stdout", "")
    stderr_text = execution_result.get("stderr", "")

    user_prompt = f"""
Dataset summary:
{dataset_info_text}

Explorer plan:
{explorer_output}

Current code:
{code_text}

Execution result:
return_code={execution_result.get("return_code")}
score={execution_result.get("score")}

STDOUT:
{stdout_text}

STDERR:
{stderr_text}

Fix the code so it runs successfully.
Apply minimal changes only.
"""

    response = ask_model_response(
        promt=user_prompt,
        system_prompt=system_prompt,
        model=model,
        temperature=0.15,
        max_tokens=10000,
        timeout=180,
    )

    return clean_code_text(response["text"].strip())