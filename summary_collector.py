import os

base_directory = "no_greedy_eval_results"

for root, dirs, files in os.walk(base_directory):
    for directory in dirs:

        # summary_file_path = os.path.join(root, directory, "EVAL", "summary.txt")
        summary_file_path = os.path.join(root, directory, "summary.txt")

        if os.path.exists(summary_file_path):
            with open(summary_file_path, "r") as summary_file:
                summary_content = summary_file.read()
            
            with open(f"all_summaries_{base_directory}.txt", "a") as summary:
                summary.write(f"{directory}\n{summary_content}\n\n")