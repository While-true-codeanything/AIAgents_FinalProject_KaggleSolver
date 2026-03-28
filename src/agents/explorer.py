from src.tools.llm_api_connector import ask_model_response


def run_explorer(dataset_info_text, model):
    system_prompt = system_prompt = """
You are an experienced ML data analyst.

Your task:
- analyze dataset summary
- identify likely feature types
- infer whether the task is classification or regression from target stats
- propose a strong baseline approach
- suggest useful preprocessing and feature engineering ideas

Available libraries in the environment:
- pandas
- numpy
- scikit-learn
- catboost
- xgboost
- requests

Rules:
- propose only models and approaches compatible with these libraries
- prefer practical tabular ML solutions
- if the dataset has many categorical features, CatBoost is a strong baseline
- do not suggest lightgbm or any unavailable library
- do not write code

Return your answer in the following format:

TASK_TYPE: ...
TARGET_TYPE: ...
BASELINE_MODEL: ...
MAIN_FEATURE_GROUPS:
- ...
- ...

PREPROCESSING:
- ...
- ...

FEATURE_ENGINEERING_IDEAS:
- ...
- ...
- ...

VALIDATION_STRATEGY:
- ...

RISKS:
- ...
- ...
"""

    user_prompt = f"""
Analyze the following dataset summary and propose a baseline ML plan.

{dataset_info_text}
"""

    response = ask_model_response(
        user_prompt,
        model,
        system_prompt=system_prompt,
        temperature=0.2,
        max_tokens=2000,
        timeout=60,
    )

    return response["text"]