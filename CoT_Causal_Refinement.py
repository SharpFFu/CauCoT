from os import system
import csv
import torch
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from modelscope import snapshot_download
from modelscope.msdatasets import MsDataset
from modelscope.utils.constant import DownloadMode
from collections import Counter
import json

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line_num, line in enumerate(file, 1):
            stripped_line = line.strip()
            obj = json.loads(stripped_line)
            data.append(obj)
    return data
def save_json_array(data_list, file_path, indent=4):
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, 
                    ensure_ascii=False,
                    indent=indent, 
                    separators=(',', ': ') if indent else None)
        print(f"Successfully wrote the {len(jsonl_data)} objects to {file_name}")
    except Exception as e:
        print(f"Write failure: {str(e)}")
def save_json_simple(jsonl_data, file_name, indent=4):
    with open(file_name, 'w', encoding='utf8') as file:
        for obj in jsonl_data:
            if "\n" in str(obj):
                obj = {k: v.replace("\n", "") if isinstance(v, str) else v 
                    for k, v in obj.items()}
            line = json.dumps(obj, ensure_ascii=False)
            file.write(line + "\n")
        print(f"Successfully wrote the {len(jsonl_data)} objects to {file_name}")
def save_json_obj(jsonl_data, file_name, indent=4):
    with open(file_name, 'a', encoding='utf8') as file:
        if "\n" in str(jsonl_data):
            jsonl_data = {k: v.replace("\n", "") if isinstance(v, str) else v 
                for k, v in jsonl_data.items()}
        line = json.dumps(jsonl_data, ensure_ascii=False)
        file.write(line + "\n")
    print(f"Successfully wrote one object to {file_name}")

file_name = './CoT Errors/Measure_error/Measure_error.jsonl'

output_file = './Refined CoT/Measure_error/Measure_error.jsonl'

ds_train = read_jsonl(file_name)

model_dir = "your_model_dir/Qwen2.5-72B-Instruct/Qwen/Qwen2___5-72B-Instruct"

pipe = pipeline(
    "text-generation",
    model=model_dir,
    torch_dtype=torch.bfloat16,
    device_map="auto", 
)

def analyse(Cpa, Ci):
    # Cpa + Ci
    qu1 = "[Question]" + instruction + "[Hint]" + Cpa + '.' + Ci + '.'
    # Ci
    qu2 = "[Question]" + instruction + "[Hint]" + Ci + '.'

    # ---- qu1
    messages = [
        {"role": "system", "content": """You are a mathematical problem solving model. Now you need to accurately answer the result of the mathematical problem according to the mathematical problem I provided you and some prompt information. You don't need to give the process of solving the problem, just answer me the correct result. The questions will be presented in the following format:
            [Question] Question text
            [Hint] Hint content
        In the end, you will need to answer the questions and prompts based on the options, with the answer format:
        Note: you should answer and only answer the question without any other words."""},
        {"role": "user", "content": qu1},
    ]
    outputs = pipe(
        messages,
        max_new_tokens=256,
    )
    print(outputs[0]["generated_text"][-1])

    # ---- qu2
    messages = [
        {"role": "system", "content": """You are a mathematical problem solving model. Now you need to accurately answer the result of the mathematical problem according to the mathematical problem I provided you and some prompt information. You don't need to give the process of solving the problem, just answer me the correct result. The questions will be presented in the following format:
            [Question] Question text
            [Hint] Hint content
        In the end, you will need to answer the questions and prompts based on the options, with the answer format:
        Note: you should answer and only answer the question without any other words."""},
        {"role": "user", "content": qu2},
    ]
    outputs = pipe(
        messages,
        max_new_tokens=256,
    )
    print(outputs[0]["generated_text"][-1])

    # ---- Asking whether the influence of the large model Cpa on Ci is significant for answering this question
    qu3 =  "[Question]" + instruction + "[Hint1]" + Cpa + '.' + "[Hint2]" + Ci + '.'
    messages = [
        {"role": "system", "content": """You are a chatbot.Now, you need to determine the impact of the first statement on answering the question based on the questions I have provided, along with the corresponding options (labeled (A), (B), etc.), using my two hints. The questions will be presented in the following format:
            [Question] Question text
            [Hint1] Hint content
            [Hint2] Hint content
        You need to carefully weigh the impact of Hint1 content on Hint2 answers. If the full score is 100, you need to score the size of the impact. If the score is higher than 50, output 1, if it is less than or equal to 50, output 0.".
        Note: You should answer and Only answer 0 or 1 without any other words."""},
        # Score judgment
        {"role": "user", "content": qu3},
    ]
    outputs = pipe(
        messages,
        max_new_tokens=1,
    )
    score_A = outputs[0]["generated_text"][-1]["content"]
    
    # ---- Inquire whether the influence of Cpa of the large model on Ci is significant
    qu3 =  "[Question]" + instruction + "[Hint1]" + Cpa + '.' + "[Hint2]" + Ci + '.'
    messages = [
        {"role": "system", "content": """You are a chatbot.Now, you need to determine the impact of the first statement on answering the question based on the questions I have provided, along with the corresponding options (labeled (A), (B), etc.), using my two hints. The questions will be presented in the following format:
            [Question] Question text
            [Hint1] Hint content
            [Hint2] Hint content
        You need to carefully weigh the impact of Hint1's content to Hint2's content. If the full score is 100, you need to score the size of the impact. If the score is higher than 50, output 1, if it is less than or equal to 50, output 0.".
        Note: You should Only answer 0 or 1 without any other words."""},
        # Score judgment
        {"role": "user", "content": qu3},
    ]
    outputs = pipe(
        messages,
        max_new_tokens=1,
    )
    score_C = outputs[0]["generated_text"][-1]["content"]
    print(score_A)
    print(score_C)
    return score_A, score_C

def update_cot(Cpa, Ci):
    # Let the large model update the CoT that has no causal relationship
    quseach =  "[Question]" + instruction + "[Hint1]" + Cpa + '.' + "[Hint2]" + Ci + '.'
    messages = [
        {"role": "system", "content": """You are a chatbot.Now, you need to determine the impact of the first statement on answering the question based on the questions I have provided, along with the corresponding options (labeled (A), (B), etc.), using my two hints. The questions will be presented in the following format:
            [Question] Question text
            [Hint1] Hint content
            [Hint2] Hint content
        Imagine you are a master who is answering question and trying to think about the problem step by step with reasoning path following Hint1 to Hint2, where Hint1 and Hint2 should have strong causal relation. Now Hin2 is wrong step,  what is this step Hint2 that can meet the strong causal relationship with the previous step Hint1? Now, you need to generate several Hint2s that have a strong causal relationship with Hint1 according to your understanding.
        Note that your answer should follow the format:
        [1] Hint2-1 text.
        [2] Hint2-2 text.
        ...
        [5] Hint2-5 text."""},
        {"role": "user", "content": quseach},
    ]#Search for chain: Identify several possible correct ci based on the corresponding content of the paper
    outputs = pipe(
        messages,
        max_new_tokens = 256,
    )
    Search_for_chain = outputs[0]["generated_text"][-1]["content"]
    quseach =  "[Question]" + instruction + "[Hint1]" + Cpa + '.' + "[possible Answer]" + Search_for_chain + '.'
    messages = [
        {"role": "system", "content": """You are a chatbot.Now, you need to determine the impact of the first statement on answering the question based on the questions I have provided, along with the corresponding options (labeled (A), (B), etc.), using my two hints. The questions will be presented in the following format:
            [Question] Question text
            [Hint1] Hint content
            [Search_for_chain] Search_for_chain content
        Imagine you are a master who is answering question and trying to think about the problem step by step with reasoning path following Hint1 to Hint2, where Hint1 and Hint2 should have strong causal relation. Among the possible Hint2 listed in Search_for_chain, choose the result that are most likely to have strong reasoning to answering question. Return the chosen Hint2's content.
        Note: You should only answer the Hint2's text without any other words."""},
        {"role": "user", "content": quseach},
    ]#Refine the Search for chain: Identify the ci that best fits the problem
    outputs = pipe(
        messages,
        max_new_tokens=256,
    )
    updated = outputs[0]["generated_text"][-1]["content"]
    return updated

cnt = 0
testtimes = 2   # total epoch
causal_sigscore = 0.5
for row in ds_train:
    instruction = row["problem"]
    cnt += 1
    print(f"---{cnt}---")
    
    CoT = row["Error CoT"]
    n = len(CoT)

    alpha = 0.5
    belt = 0.5
    
    for i in range (0, n - 1):
        index_score = 0
        Cpa = CoT[i]
        Ci = CoT[i + 1]
        for _ in range(0, testtimes):
            index_a, index_c = analyse(Cpa, Ci) # Return two causal effect scores
            # Calculate the comprehensive causal score
            index_causalscore = alpha * float(index_a[0]) + belt * float(index_c[0])
            index_score = index_score + index_causalscore
            
        index_score_av = index_score / testtimes
        if index_score_av > causal_sigscore:
            continue
        if index_score_av <= causal_sigscore:
            index_ci = update_cot(Cpa, Ci)
            CoT[i + 1] = index_ci
    row["Error CoT"] = CoT
    save_json_obj(row, output_file)
