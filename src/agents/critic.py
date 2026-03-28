from src.tools.llm_api_connector import ask_model_response


def run_critic(dataset_info_text, explorer_output, code_text, execution_result, model):
    system_prompt = """
You are a strong ML reviewer for tabular ML competitions.

Your task:
- analyze the dataset summary
- analyze the explorer plan
- analyze the current code
- analyze execution results
- identify the most important blocker or the most promising next improvement

Available libraries:
- pandas
- numpy
- scikit-learn
- catboost
- xgboost
- requests

Rules:
- suggest only improvements compatible with these libraries
- do not suggest LightGBM or any unavailable library
- prefer minimal, targeted changes
- preserve the current working pipeline if it already runs
- if the code failed, focus only on the failure cause
- if the code worked, suggest at most 3 small improvements
- be concise

Runtime constraint:
- The full script must stay within 300 seconds.
- Suggest only improvements that are realistic within this runtime budget.
- Prefer small, targeted improvements over heavier models or more expensive validation.

Return your answer in this format:

MAIN_PROBLEMS:
- ...
- ...

IMPROVEMENTS:
- ...
- ...
- ...

DECISION:
improve
"""

    stdout_text = execution_result.get("stdout", "")
    stderr_text = execution_result.get("stderr", "")
    return_code = execution_result.get("return_code", None)
    cv_score = execution_result.get("cv_score", None)

    user_prompt = f"""
Dataset summary:
{dataset_info_text}

Explorer plan:
{explorer_output}

Generated code:
{code_text}

Execution result:
return_code={return_code}
cv_score={cv_score}

STDOUT:
{stdout_text}

STDERR:
{stderr_text}
"""

    response = ask_model_response(
        user_prompt,
        model=model,
        system_prompt=system_prompt,
        temperature=0.2,
        max_tokens=2000,
        timeout=60,
    )

    return response["text"].strip()
