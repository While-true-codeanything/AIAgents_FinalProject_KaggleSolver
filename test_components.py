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
