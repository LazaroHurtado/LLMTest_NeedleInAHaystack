import pandas as pd
import json
import sys
import glob
import time
import os
import yaml

sys.path.insert(1, '/home/lazaro/projects/LLMTest_NeedleInAHaystack')

from dotenv import load_dotenv

from src.evaluators.openai import OpenAIEvaluator
from src.models.huggingface_tester import HuggingFaceTester
from src.llm_needle_haystack_tester import LLMNeedleHaystackTester

load_dotenv()

SAMPLES = 100
OFFSET = 0

CSV_FILE = "/home/lazaro/projects/LLMTest_NeedleInAHaystack/viz/stronger_prompts_results.csv"
RESULTS_FOLDER_PATH = "/home/lazaro/projects/LLMTest_NeedleInAHaystack/results"
LLM_TESTS_FOLDER_PATH = "/home/lazaro/needle_in_haystack_results/phi2_stronger_prompts"
PHI2_FINETUNED_FOLDER_PATH = "/home/lazaro/needle_in_haystack_results/phi2_finetuned_results"
PROMPTS_FILE_PATH = "/home/lazaro/projects/LLMTest_NeedleInAHaystack/viz/prompts.yml"

NEEDLE = "\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n"
RETRIEVAL_QUESTION = "What is the best thing to do in San Francisco?"
GENERATION_WARMING = "The best thing to do in San Franciso is"

MODEL_NAME = "cognitivecomputations/dolphin-2_6-phi-2"
PROMPT_STRUCTURE = """<|im_start|>system
You are a helpful AI bot that answers questions for a user. Keep your response short and direct.<|im_end|>
<|im_start|>user
{user_input}<|im_end|>
<|im_start|>assistant"""

def df_from_folder(folder):
    # Using glob to find all json files in the directory
    json_files = glob.glob(f"{folder}/*.json")

    # List to hold the data
    data = []

    # Iterating through each file and extract the 3 columns we need
    for file in json_files:
        with open(file, 'r') as f:
            json_data = json.load(f)
            # Extracting the required fields
            document_depth = json_data.get("depth_percent", None)
            context_length = json_data.get("context_length", None)
            score = json_data.get("score", None)
            # Appending to the list
            data.append({
                "Document Depth": document_depth,
                "Context Length": context_length,
                "Score": score
            })

    # Creating a DataFrame
    return pd.DataFrame(data)

def test_prompt_variant(prompt, sampled_df: pd.DataFrame):
    evaluator = OpenAIEvaluator("gpt-4-0125-preview", true_answer=NEEDLE, question_asked=RETRIEVAL_QUESTION)
    model = HuggingFaceTester(MODEL_NAME, device="cpu", prompt_structure=prompt)

    for _, row in sampled_df.iterrows():
        depth = int(row["Document Depth"])
        context_length = int(row["Context Length"])
        
        tester = LLMNeedleHaystackTester(model_to_test=model,
                                evaluator=evaluator,
                                needle=NEEDLE,
                                retrieval_question=RETRIEVAL_QUESTION,
                                document_depth_percents=[depth],
                                context_lengths=[context_length])
        tester.start_test()

def log_trial(prompt, time_taken):
    results_df = df_from_folder(RESULTS_FOLDER_PATH)

    good_scores = len(results_df[results_df["Score"] >= 5.0])
    good_scores_ratio = round(good_scores / len(results_df), 2)

    if os.path.exists(CSV_FILE):
        trials = pd.read_csv(CSV_FILE)
    else:
        trials = pd.DataFrame(columns=["Prompt", "Time Taken", "Good Scores", "Good Scores Ratio"])

    new_data = pd.DataFrame([{
        "Prompt": prompt,
        "Time Taken": time_taken,
        "Good Scores": good_scores,
        "Good Scores Ratio": good_scores_ratio
    }])
    trials = pd.concat([trials, new_data], ignore_index=True)

    trials.to_csv(CSV_FILE, index=False)

def move_results_folder(version):
    folder_name = "stronger_prompt_v{version}_results".format(version=version)
    command = "mv {results_folder} {llm_tests_folder}/{folder_name}".format(
        results_folder=RESULTS_FOLDER_PATH,
        llm_tests_folder=LLM_TESTS_FOLDER_PATH,
        folder_name=folder_name)
    
    os.system(command)

def main(df, prompts_file):
    with open(prompts_file, 'r') as f:
        prompts = yaml.safe_load(f)["test"]

    for i, prompt in enumerate(prompts):
        sampled_df = df.sample(SAMPLES)
        depths = sampled_df["Document Depth"].to_list()
        context_lengths = sampled_df["Context Length"].to_list()

        with open('stronger_prompt.log', 'a') as f:
            f.write(f"\n\nRunning prompt {i+1}\n")
            f.write(f"\t Depths: {depths}\n")
            f.write(f"\t Context Lengths: {context_lengths}\n")

        final_prompt = PROMPT_STRUCTURE.format(user_input=prompt)
        
        start = time.monotonic()
        test_prompt_variant(final_prompt, sampled_df)
        end = time.monotonic()

        total_time = round(end - start, 2)
        log_trial(prompt, total_time)

        move_results_folder(OFFSET+i+1)


if __name__ == "__main__":
    phi2_df = df_from_folder(PHI2_FINETUNED_FOLDER_PATH)
    bad_scores_data = phi2_df[phi2_df["Score"] == 1.0]

    main(bad_scores_data, PROMPTS_FILE_PATH)