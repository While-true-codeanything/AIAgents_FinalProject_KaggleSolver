from src.agents.engineer import run_engineer
from src.agents.explorer import run_explorer
from src.config import CONFIG
from src.tools.code_executor import save_code, execute_code
from src.tools.dataset_inputer import load_data, get_dataset_info, \
    format_dataset_info
from src.tools.llm_api_connector import ask_model_response


def test_datatools():
    train_df, test_df, sample_submition = load_data()

    summary = get_dataset_info(train_df, test_df, 'target')
    print(format_dataset_info(summary))


def test_modelapi():
    response = ask_model_response(
        promt=" Write script to print (10 * 10)",
        system_prompt="You are a strong ML engineer. Return only Python code.",
        model="qwen/qwen3-coder-next",
    )
    print(response["text"])


def test_explorer():
    train_df, test_df, sample_submition = load_data()
    summary = get_dataset_info(train_df, test_df, 'target')
    response = run_explorer(summary, CONFIG["models"]["explorer"])
    print(response)


def test_engineer():
    train_df, test_df, sample_submition = load_data()
    summary = get_dataset_info(train_df, test_df, 'target')
    explorer_response = run_explorer(summary, CONFIG["models"]["explorer"])
    engineer_response = run_engineer(summary, explorer_response, CONFIG["models"]["engineer"], critic_feedback=None)
    print(engineer_response)


def test_code_executor():
    response = ask_model_response(
        '''
Write a Python script that:
1. computes 10 * 10
2. prints the result
3. sets mean_cv_score = 0.0
4. prints it exactly as:
print(f"CV_SCORE={mean_cv_score}")

Return only Python code.
''',
        model="qwen/qwen3-coder-next",
        system_prompt="You are a strong ML engineer. Return only Python code.",
    )

    print(response["text"])
    file_path = "test_ce.py"
    result = execute_code(response["text"], file_path, timeout=300)
    print(result)
