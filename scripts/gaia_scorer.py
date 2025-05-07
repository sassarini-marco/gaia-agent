# gaia_scorer_sequential.py : used to test gaia agent directly agains GAIA samples

import argparse
import json
import time
import traceback
import pandas as pd
import re
import string
import warnings
from pathlib import Path
from typing import Dict, Any, List, Tuple

from tqdm import tqdm

from gaia_agent.utils.agent_wrapper import GAIABenchmarkAgentWrapper

def process_single_task(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """Processes a single task/question with the agent system."""
    start_time = time.time()
    # Reset agent state before processing each task

    task_id = task_data.get("task_id", "MISSING_ID")
    question = task_data.get("Question")
    true_answer = task_data.get("Final Answer") # Renamed from 'Final answer' for consistency
    file_name = task_data.get("file_name")
    level = task_data.get("Level") # Keep level info if present

    agent_memory_dump = []
    predicted_answer = "ERROR: Agent Did Not Respond"
    error_message = None

    if question is None:
        error_message = "Missing 'Question' field in input data"
        predicted_answer = f"ERROR: {error_message}"
    else:
        try:
            additional_args = {}
            # Handle file if present - Assuming data structure from prompt
            if file_name:
                # Adjust the base path according to your project structure
                potential_path = Path("./data/gaia") / file_name # Example path
                if potential_path.exists():
                    file_path_to_pass = str(potential_path.resolve())
                    additional_args['file_path'] = file_path_to_pass
                    # print(f"Task {task_id}: Providing file '{file_path_to_pass}'") # Optional: reduce verbosity
                else:
                    print(f"Warning: Task {task_id}: File '{file_name}' not found at '{potential_path}'")

            # Run the agent
            agent_wrapper = GAIABenchmarkAgentWrapper()
            predicted_answer = agent_wrapper(task_data)

        except Exception as e:
            print(f"\nError during agent run for task_id {task_id}: {e}")
            print(traceback.format_exc())
            error_message = f"Agent execution error: {type(e).__name__}: {e}"
            predicted_answer = f"ERROR: {error_message}"

            # Try to get memory even on error


    end_time = time.time()

    return {
        "task_id": task_id,
        "level": level,
        "question": question,
        "true_answer": true_answer,
        "predicted_answer": predicted_answer,
        "agent_memory": agent_memory_dump,
        "error": error_message,
        "start_time": start_time,
        "end_time": end_time,
        "duration": end_time - start_time,
    }

def save_results_to_jsonl(results: List[Dict], output_path: str):
    """Save results list to JSONL file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f_out:
            for entry in results:
                f_out.write(json.dumps(entry) + "\n")
    except IOError as e:
        print(f"Error saving results to {output_path}: {e}")


def run_agent_on_dataset_sequential(
        data_path: str,
        output_path: str,
        max_questions: int | None = None,
) -> str:
    """
    Runs the agent system sequentially on a dataset subset and saves results.

    Args:
        data_path (str): Path to the input JSONL dataset file.
        output_path (str): Path to save the raw agent results JSONL file.
        max_questions (int, optional): Max questions to process. Defaults to all.

    Returns:
        str: The path to the saved results file.

    Raises:
        RuntimeError: If agent initialization fails.
        FileNotFoundError: If the data file is not found.
        ValueError: If the data file is empty or invalid.
    """
    # Load dataset using pandas
    print(f"Loading dataset from {data_path}...")
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Data file not found at {data_path}")
    try:
        df = pd.read_json(data_path, lines=True)
        df.rename(columns={"Final answer": "Final Answer", "question": "Question"}, inplace=True, errors='ignore')

        if df.empty:
            raise ValueError(f"No valid data loaded from {data_path}")
        if "Question" not in df.columns or "Final Answer" not in df.columns:
             warnings.warn(f"Input file {data_path} might be missing 'Question' or 'Final Answer' columns.", UserWarning)

        print(f"Loaded {len(df)} examples from {data_path}")
    except Exception as e:
        print(f"Error loading or processing data from {data_path}: {e}")
        raise ValueError(f"Failed to load data: {e}") from e

    # Limit questions if needed
    if max_questions is not None and max_questions > 0:
        df = df.head(max_questions)
    actual_max = len(df)
    if actual_max == 0:
        print("No questions to process after filtering.")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        save_results_to_jsonl([], output_path)
        return output_path

    print(f"Processing {actual_max} questions sequentially...")

    # Ensure output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create a backup file for incremental saving
    backup_path = f"{output_path}.backup"

    # Execute evaluation sequentially
    results = []
    error_count = 0

    for index, row in tqdm(df.iterrows(), total=actual_max, desc="Running Agent (Sequential)"):
        task_data = row.to_dict()
        task_id = task_data.get("task_id", f"Row_{index}")

        try:
            # Process the task using the single, shared agent instance
            result_entry = process_single_task(task_data)
            results.append(result_entry)
            if result_entry.get("error"):
                error_count += 1
        except Exception as e:
            # This catches errors *outside* process_single_task, unlikely but possible
            print(f"\nCritical Error processing task_id {task_id}: {e}")
            print(traceback.format_exc())
            results.append({
                "task_id": task_id,
                "level": task_data.get("Level"),
                "question": task_data.get("Question", "ERROR: Question Key Missing"),
                "true_answer": task_data.get("Final Answer", None),
                "predicted_answer": "SCRIPT_ERROR",
                "agent_memory": [],
                "error": f"Evaluation framework error: {type(e).__name__}: {e}",
                "start_time": time.time(), # Approximate
                "end_time": time.time(),
                "duration": 0,
            })
            error_count += 1
        finally:
            # Save backup incrementally
            if (index + 1) % 10 == 0 or (index + 1) == actual_max:
                 save_results_to_jsonl(results, backup_path)


    # Save final results
    print("Agent run complete. Saving final results...")
    save_results_to_jsonl(results, output_path)

    # Remove backup file if final save succeeded
    backup_file = Path(backup_path)
    if backup_file.exists():
        try:
            backup_file.unlink()
        except OSError as e:
            print(f"Warning: Could not remove backup file {backup_path}: {e}")

    print("\n--- Agent Run Summary ---")
    print(f"Total questions processed: {len(results)}")
    print(f"Agent/Script Errors during run: {error_count}")
    print(f"Raw results saved to: {output_path}")

    return output_path


# --- Scoring Logic (Copied and potentially adapted from provided script) ---
# [Keep the scoring functions: normalize_number_str, split_string, is_float_str,
#  normalize_str, gaia_level1_score exactly as they were in the previous version]

def normalize_number_str(number_str: str) -> float | None:
    """Normalizes a string to a float, handling common units/commas. Returns None on failure."""
    if not isinstance(number_str, str):
        number_str = str(number_str) # Attempt conversion if not string

    for char in ["$", "%", ",", "€", "£", "¥"]:
        number_str = number_str.replace(char, "")
    number_str = number_str.strip()

    try:
        return float(number_str)
    except (ValueError, TypeError):
        return None

def split_string(s: str, char_list: list[str] = [",", ";"]) -> list[str]:
    """Splits a string by multiple delimiters and strips whitespace."""
    if not isinstance(s, str):
        s = str(s)
    pattern = f"[{''.join(re.escape(c) for c in char_list)}]"
    return [item.strip() for item in re.split(pattern, s) if item.strip()]

def is_float_str(element: any) -> bool:
    """Checks if a string can be reliably converted to a float after normalization."""
    if not isinstance(element, str):
        element = str(element)
    if not element or element.isspace():
        return False
    return normalize_number_str(element) is not None

def normalize_str(input_str: Any, remove_punct: bool = True) -> str:
    """Normalize a string by removing whitespace, optionally punctuation, and lowercasing."""
    if not isinstance(input_str, str):
        input_str = str(input_str)
    no_spaces = re.sub(r"\s+", "", input_str)
    if remove_punct:
        translator = str.maketrans("", "", string.punctuation)
        normalized = no_spaces.lower().translate(translator)
    else:
        normalized = no_spaces.lower()
    return normalized

def gaia_level1_score(model_answer: Any, ground_truth: Any) -> bool:
    """Scores the agent's answer against the ground truth based on GAIA Level 1 criteria."""
    ma_str = str(model_answer) if model_answer is not None else ""
    gt_str = str(ground_truth) if ground_truth is not None else ""

    if gt_str == "":
        warnings.warn("Ground truth is empty, cannot score.", UserWarning)
        return False
    if ma_str == "" or ma_str.startswith("ERROR:") or ma_str == "SCRIPT_ERROR":
        return False

    gt_float = normalize_number_str(gt_str)
    if gt_float is not None:
        ma_float = normalize_number_str(ma_str)
        return ma_float is not None and ma_float == gt_float

    list_delimiters = [",", ";"]
    contains_delimiter = any(char in gt_str for char in list_delimiters)
    if contains_delimiter and normalize_number_str(gt_str.replace(",", "").replace(";", "")) is None:
        gt_elems = split_string(gt_str, char_list=list_delimiters)
        ma_elems = split_string(ma_str, char_list=list_delimiters)

        if len(gt_elems) != len(ma_elems):
            return False

        comparisons = []
        for ma_elem, gt_elem in zip(ma_elems, gt_elems):
            gt_elem_float = normalize_number_str(gt_elem)
            if gt_elem_float is not None:
                ma_elem_float = normalize_number_str(ma_elem)
                comparisons.append(ma_elem_float is not None and ma_elem_float == gt_elem_float)
            else:
                comparisons.append(normalize_str(ma_elem, remove_punct=True) == normalize_str(gt_elem, remove_punct=True))
        return all(comparisons)

    return normalize_str(ma_str, remove_punct=True) == normalize_str(gt_str, remove_punct=True)


# --- Main Scorer Orchestration ---

def score_results(results_path: str, failures_path: str) -> Tuple[float, int, int]:
    """
    Loads agent results, scores them, saves failures, and returns accuracy.
    (Identical to the previous version)
    """
    print(f"\nScoring results from: {results_path}")
    try:
        results_df = pd.read_json(results_path, lines=True)
        if results_df.empty:
            print("Result file is empty. No scores to calculate.")
            return 0.0, 0, 0
    except FileNotFoundError:
        print(f"Error: Results file not found at {results_path}")
        return 0.0, 0, 0
    except Exception as e:
        print(f"Error loading results file {results_path}: {e}")
        return 0.0, 0, 0

    total_scored = 0
    correct_answers = 0
    failures = []

    for index, row in tqdm(results_df.iterrows(), total=len(results_df), desc="Scoring Answers"):
        task_id = row.get("task_id", f"Row_{index}")
        pred = row.get("predicted_answer")
        true = row.get("true_answer")
        question = row.get("question")
        level = row.get("level")
        agent_memory = row.get("agent_memory", [])
        run_error = row.get("error")

        if true is None:
            print(f"Warning: Skipping Task ID {task_id} due to missing 'true_answer'.")
            continue

        total_scored += 1
        is_correct = False
        try:
            is_correct = gaia_level1_score(pred, true)
        except Exception as e:
            print(f"\nError scoring Task ID {task_id}: {e}")
            print(f"  Predicted: {pred}")
            print(f"  True: {true}")
            is_correct = False

        if is_correct:
            correct_answers += 1
        else:
            failures.append({
                "task_id": task_id,
                "level": level,
                "question": question,
                "true_answer": true,
                "predicted_answer": pred,
                "agent_run_error": run_error,
                "agent_memory": agent_memory,
            })

    accuracy = (correct_answers / total_scored) * 100 if total_scored > 0 else 0.0

    if failures:
        print(f"Saving {len(failures)} failure details to: {failures_path}")
        save_results_to_jsonl(failures, failures_path)
    else:
        print("No failures recorded.")

    print("\n--- Scoring Summary ---")
    print(f"Total questions scored: {total_scored}")
    print(f"Correct answers: {correct_answers}")
    print(f"Accuracy: {accuracy:.2f}%")

    return accuracy, correct_answers, total_scored


def main():
    parser = argparse.ArgumentParser(description="Run GAIA Agent Evaluation and Scoring (Sequentially).")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the input validation dataset file (JSONL format). Example: ./data/gaia/gaia_level1_validation.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./gaia_eval_output_sequential", # Changed default output dir slightly
        help="Directory to save the evaluation results and failure analysis.",
    )
    parser.add_argument(
        "--max_questions",
        type=int,
        default=None,
        help="Maximum number of questions to process (optional, process all if None).",
    )
    # Removed num_workers argument
    parser.add_argument(
        "--results_filename",
        type=str,
        default="agent_results_sequential.jsonl", # Changed default filename
        help="Filename for the raw agent results.",
    )
    parser.add_argument(
        "--failures_filename",
        type=str,
        default="failures_sequential.jsonl", # Changed default filename
        help="Filename for the failure analysis.",
    )

    args = parser.parse_args()

    # --- Setup Paths ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / args.results_filename
    failures_path = output_dir / args.failures_filename

    print("--- Starting GAIA Agent Evaluation (Sequential) ---")
    print(f"Input data: {args.data_path}")
    print(f"Output directory: {output_dir}")
    if args.max_questions:
        print(f"Max questions: {args.max_questions}")
    # No num_workers to print

    # --- Step 1: Run Agent on Dataset (Sequentially) ---
    try:
        # Call the sequential version of the runner function
        actual_results_path = run_agent_on_dataset_sequential(
            data_path=args.data_path,
            output_path=str(results_path),
            max_questions=args.max_questions,
            # No num_workers argument passed
        )
    except (RuntimeError, FileNotFoundError, ValueError) as e:
        print(f"\nFatal error during agent execution phase: {e}")
        print("Scoring cannot proceed.")
        return # Exit if agent run failed critically

    # --- Step 2: Score the Results ---
    if Path(actual_results_path).exists():
         score_results(
            results_path=actual_results_path,
            failures_path=str(failures_path)
        )
    else:
        print(f"Agent results file '{actual_results_path}' not found. Skipping scoring.")

    print("\n--- Evaluation Complete ---")


if __name__ == "__main__":
    main()