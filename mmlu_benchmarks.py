import argparse
import asyncio
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
import tiktoken
from tqdm import tqdm

choices = ["A", "B", "C", "D"]

tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")


def format_subject(subject):
    return " " + " ".join(subject.split("_"))


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = f"The following are multiple choice questions (with answers) about{format_subject(subject)}.\n\n"
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


def call_vllm_chat(prompt, server_url="http://localhost:8000/v1/chat/completions", temperature=0, max_tokens=1):
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "model": "vllm_cpu_offload",  # Must match --served-model-name
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    response = requests.post(server_url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def evaluate(args, subject, dev_df, test_df):
    prompts = []
    labels = []

    k = args.ntrain
    train_prompt = gen_prompt(dev_df, subject, k)
    while len(tokenizer.encode(train_prompt)) > 1536:
        k -= 1
        train_prompt = gen_prompt(dev_df, subject, k)

    for i in range(test_df.shape[0]):
        prompt_end = format_example(test_df, i, include_answer=False)
        prompt = train_prompt + prompt_end
        prompts.append(prompt)

        label = test_df.iloc[i, test_df.shape[1] - 1]
        labels.append(label)

    preds = [None] * len(prompts)
    max_tokens = 1

    def get_one_answer(i):
        try:
            pred = call_vllm_chat(prompts[i], server_url=args.server_url, temperature=0, max_tokens=max_tokens)
            preds[i] = pred.strip()[0]
        except Exception as e:
            print(f"Error processing prompt {i}: {e}")
            preds[i] = "A"

    tic = time.time()
    if args.parallel == 1:
        for i in range(len(prompts)):
            get_one_answer(i)
    else:
        with ThreadPoolExecutor(args.parallel) as executor:
            executor.map(get_one_answer, list(range(len(prompts))))
    latency = time.time() - tic

    cors = [pred == label for pred, label in zip(preds, labels)]
    acc = np.mean(cors)
    cors = np.array(cors)

    print(
        "Average accuracy {:.3f}, latency {:.2f}, #q: {} - {}".format(
            acc, latency, len(prompts), subject
        )
    )

    return cors, acc, latency


def main(args):
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(args.data_dir, "test"))
            if "_test.csv" in f
        ]
    )

    all_cors = []
    all_latencies = []
    num_requests = 0

    for subject in tqdm(subjects[: args.nsub]):
        dev_df = pd.read_csv(
            os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None
        )[: args.ntrain]
        test_df = pd.read_csv(
            os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None
        )

        cors, acc, latency = evaluate(args, subject, dev_df, test_df)
        all_cors.append(cors)
        all_latencies.append(latency)
        num_requests += len(test_df)

    total_latency = np.sum(all_latencies)
    print("Total latency: {:.3f}".format(total_latency))

    weighted_acc = np.mean(np.concatenate(all_cors))
    print("Average accuracy: {:.3f}".format(weighted_acc))

    with open(args.result_file, "a") as fout:
        value = {
            "task": "mmlu",
            "backend": "vllm",
            "num_gpus": 1,
            "latency": round(total_latency, 3),
            "accuracy": round(weighted_acc, 3),
            "num_requests": num_requests,
            "other": {
                "nsub": args.nsub,
                "parallel": args.parallel,
            },
        }
        fout.write(json.dumps(value) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", type=int, default=5)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--nsub", type=int, default=60)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--result_file", type=str, default="results.jsonl")
    parser.add_argument("--server_url", type=str, default="http://localhost:8000/v1/chat/completions")
    parser.add_argument("--backend", type=str, default="vllm")  # Just to allow --backend vllm without crashing
    args = parser.parse_args()
    main(args)
