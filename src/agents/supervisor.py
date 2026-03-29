from src.config import CONFIG, ensure_directories
from src.tools.dataset_inputer import load_data, get_dataset_info, format_dataset_info
from src.tools.code_executor import execute_code
from src.tools.data_splits import save_train_valid_split
from src.agents.explorer import run_explorer
from src.agents.engineer import run_engineer
from src.agents.critic import run_critic


def run_supervisor():
    ensure_directories()

    train_df, test_df, submission_df = load_data(
        train_path=CONFIG["paths"]["train"],
        test_path=CONFIG["paths"]["test"],
        submission_sample_path=CONFIG["paths"]["submission_sample"],
    )

    dataset_info = get_dataset_info(
        train_df=train_df,
        test_df=test_df,
        target_col=CONFIG["run"]["target_col"],
    )
    dataset_info_text = format_dataset_info(dataset_info)

    split_dir = CONFIG["paths"]["data_splits"]
    train_inner_path = split_dir / "train_inner.csv"
    valid_holdout_path = split_dir / "valid_holdout.csv"

    if not train_inner_path.exists() or not valid_holdout_path.exists():
        save_train_valid_split(
            train_df=train_df,
            output_dir=split_dir,
            target_col=CONFIG["run"]["target_col"],
            valid_size=CONFIG["run"]["valid_size"],
            random_state=CONFIG["run"]["random_seed"],
            task_type="regression",
        )
        print(f"Saved train/valid split to: {split_dir}")
    else:
        print(f"Using existing split files from: {split_dir}")

    print("=== EXPLORER ===")
    explorer_output = run_explorer(
        dataset_info_text=dataset_info_text,
        model=CONFIG["models"]["explorer"],
    )
    print(explorer_output)

    max_iters = CONFIG["run"]["max_iters"]
    timeout = CONFIG["run"]["executor_timeout"]

    all_results = []
    critic_feedback = None

    best_result = None
    best_code = None
    best_iteration = None
    best_score = None

    for iteration in range(1, max_iters + 1):
        print(f"\n=== ENGINEER: ITERATION {iteration} ===")
        code_text = run_engineer(
            dataset_info_text=dataset_info_text,
            explorer_output=explorer_output,
            model=CONFIG["models"]["engineer"],
            critic_feedback=critic_feedback,
        )

        code_path = CONFIG["paths"]["generated_code"] / f"iteration_{iteration}.py"

        execution_result = execute_code(
            code_text=code_text,
            file_path=code_path,
            timeout=timeout,
        )

        print(f"\n=== EXECUTION RESULT: ITERATION {iteration} ===")
        print(execution_result)

        current_score = execution_result.get("cv_score")
        current_ok = execution_result.get("return_code") == 0 and current_score is not None

        all_results.append(
            {
                "iteration": iteration,
                "code_text": code_text,
                "code_path": str(code_path),
                "execution_result": execution_result,
            }
        )

        if current_ok and (best_score is None or current_score < best_score):
            best_score = current_score
            best_result = execution_result
            best_code = code_text
            best_iteration = iteration

        if iteration < max_iters:
            critic_feedback = run_critic(
                dataset_info_text=dataset_info_text,
                explorer_output=explorer_output,
                code_text=code_text,
                execution_result=execution_result,
                model=CONFIG["models"]["critic"],
            )

            print(f"\n=== CRITIC: AFTER ITERATION {iteration} ===")
            print(critic_feedback)

    if best_code is not None:
        best_code_path = CONFIG["paths"]["generated_code"] / "best_pipeline.py"
        with open(best_code_path, "w", encoding="utf-8") as f:
            f.write(best_code)
    else:
        best_code_path = None

    print("\n=== FINAL RESULT ===")
    print(f"Best iteration: {best_iteration}")
    print(f"Best score: {None if best_result is None else best_result.get('cv_score')}")
    print(f"Best script path: {best_code_path}")

    return {
        "dataset_info_text": dataset_info_text,
        "explorer_output": explorer_output,
        "all_results": all_results,
        "best_result": best_result,
        "best_iteration": best_iteration,
        "best_code_path": None if best_code_path is None else str(best_code_path),
        "train_inner_path": str(train_inner_path),
        "valid_holdout_path": str(valid_holdout_path),
    }