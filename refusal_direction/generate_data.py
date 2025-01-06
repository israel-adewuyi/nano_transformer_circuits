import os
import json
import random
import warnings
import requests
import pandas as pd

# from dotenv import load_dotenv
from datasets import load_dataset

warnings.filterwarnings('ignore')

load_dotenv()

hf_token =  os.getenv("HF_TOKEN")

current_dir = os.path.dirname(os.path.abspath(__file__))

def load_config(config_path):
    with open(config_path, 'r') as file:
        return json.load(file)

def read_malicious_instruct_repo():
    url = 'https://raw.githubusercontent.com/Princeton-SysML/Jailbreak_LLM/refs/heads/main/data/MaliciousInstruct.txt'

    response = requests.get(url)

    return response.text


def read_harmbench_val():
    url = 'https://raw.githubusercontent.com/centerforaisafety/HarmBench/refs/heads/main/data/behavior_datasets/harmbench_behaviors_text_val.csv'

    val_df = pd.read_csv(url)

    return val_df

def read_alpaca_dataset():
    url = 'https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/refs/heads/main/alpaca_data.json'

    response = requests.get(url)
    response.raise_for_status()

    # Parse the JSON data
    data = response.json()

    return data


def read_harmful_dataset_list():
    # Construct the path to the JSON file in the root directory
    dataset_file_path = os.path.join(current_dir, 'datasets.json')
    
    dataset = load_config(dataset_file_path)

    harmful_datasets = dataset['dataset']['Harmful']

    return harmful_datasets

def select_train_harmful_prompts():
    list_of_harmful_dataset = read_harmful_dataset_list()

    LIST = []

    for harmful_dataset in list_of_harmful_dataset:
        data = load_dataset(harmful_dataset)['train']['prompt']
        LIST.extend(data)

    malicious_instruct_data = read_malicious_instruct_repo()
    malicious_instruct_data = malicious_instruct_data.split('\n')

    LIST.extend(malicious_instruct_data)

    selected_prompts = random.sample(LIST, 128)
    selected_prompts = [f"{prompt}\n" for prompt in selected_prompts]

    with open('train_harmful_prompts.txt', 'w') as file:
        file.writelines(selected_prompts)


def select_val_harmful_prompts():
    dataframe = read_harmbench_val()

    dataframe = dataframe[dataframe['FunctionalCategory'] == 'standard']

    val_set = dataframe.sample(n=32, random_state=42)

    selected_prompts = val_set['Behavior'].tolist()

    selected_prompts = [f"{prompt}\n" for prompt in selected_prompts]

    with open("val_harmful_prompts.txt", "w") as file:
        file.writelines(selected_prompts)


def select_harmless_prompts():
    data = read_alpaca_dataset()
    data = [item['instruction'] for item in data]
    data = set(data)

    train_instructions = random.sample(data, 128)

    data.difference_update(train_instructions)

    val_instructions = random.sample(data, 32)

    data.difference_update(val_instructions)

    evaluation_instructions = random.sample(data, 100)

    train_instructions = [f"{prompt}\n" for prompt in train_instructions]
    val_instructions = [f"{prompt}\n" for prompt in val_instructions]
    evaluation_instructions = [f"{prompt}\n" for prompt in evaluation_instructions]

    with open("train_harmless_prompts.txt", "w") as file:
        file.writelines(train_instructions)

    with open("val_harmless_prompts.txt", "w") as file:
        file.writelines(val_instructions)

    with open("evaluation_harmless_prompts.txt", 'w') as file:
        file.writelines(evaluation_instructions)
    

def main():
    select_train_harmful_prompts()
    select_val_harmful_prompts()

    select_harmless_prompts()




if __name__ == '__main__':
    main()