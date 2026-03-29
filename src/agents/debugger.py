from src.tools.code_executor import clean_code_text
from src.tools.llm_api_connector import ask_model_response


def run_debugger(dataset_info_text, explorer_output, code_text, execution_result, model):
    system_prompt = """
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
cv_score={execution_result.get("cv_score")}

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
        timeout=120,
    )

    return clean_code_text(response["text"].strip())