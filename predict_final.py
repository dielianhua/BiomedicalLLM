#!/usr/bin/env python
# coding=utf-8
import argparse
import json
import os
import random
import time
import evaluate

from tqdm import tqdm
import datasets
from openai import OpenAI
import numpy as np
from typing import List, Dict
import re
import string
from sklearn.metrics import f1_score
from collections import Counter

tests = {
     # 'NER': [
     #         "BioNLP-2013-GRO"
     # ],
         'NER': [
                 "NCBI-disease","BC5CDR","BioNLP-2011-GE","BioNLP-2013-GRO"
        ],
         'TXTCLASS': ['MedDialog'],
         'RE': ['AnEM-RE', 'BC5CDR-RE', 'BioInfer-RE'],
         'EE': ['MLEE-EE'],
}
bio = ['NCBI-disease', 'BC5CDR']
cls = ['MedDialog']
entity = [
          'BioNLP-2011-GE',
          'BioNLP-2013-GRO'
          'AnEM-RE', 'BC5CDR-RE', 'BioInfer-RE',
          'MLEE-EE'
          ]

def predict(ex, client, model, flag=False):
    messages = []
    if 'history' in ex:
        for h in ex['history']:
            messages.append({"role": "user", "content": h[0]})
            messages.append({"role": "assistant", "content": h[1]})
    prompt = ex['instruction'] + '\n' + ex['input']

    messages.append({"role": "user", "content": prompt})

    try_times = 20
    while try_times > 0:
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=1024,
                temperature=0
            )
            response, num_input, num_output = completion.choices[
                0].message.content, completion.usage.prompt_tokens, completion.usage.completion_tokens
            if flag:
                print(messages)
                #print(completion.choices[0].message.content)
            return response, num_input, num_output
        except:
            print("Waiting for the server...")

            time.sleep(3)
            try_times -= 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument('--key', type=str, required=True)
    parser.add_argument('--base_url', type=str, default='https://api.openai.com/v1')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo')
    args = parser.parse_args()
    name = args.name
    dir = args.dir
    key = args.key
    model = args.model
    base_url = args.base_url
    client = OpenAI(api_key=key, base_url=base_url)


    with open(f'{dir}/{name}.json', 'w', encoding='utf-8') as out:
        results = {}
        for cat, test_names in tests.items():
            results[cat] = {}
            for test in test_names:
                flag = True
                print(f'------------------------ {cat}: {test} ------------------------')
                results[cat][test] = {}
                config = ''
                print(test + config+"_test.json")
                # 正确拼接路径

                file_path = os.path.join('MedINST/all_history_filter_all', f"{test}{config}_test.json")

                # 加载数据集
                data = datasets.load_dataset('json', data_files=file_path)['train']
                print(data)
                targets = []
                predictions = []
                tqdm_data = tqdm(data)
                for d in tqdm_data:
                    #print(d)
                    targets.append(d['output'])
                    if not flag:
                        flag = random.randint(1, 50) == 1
                    pre, len_prompt, len_gen = predict(d, client, model, flag=flag)
                    predictions.append(pre)
                    tqdm_data.set_description(f'Inp: {len_prompt} Gen: {len_gen}')
                    flag = False
                # print(targets, predictions)
                results[cat][test]['generated'] = [{'prediction': pre, 'target': target} for pre, target in
                                                   zip(predictions, targets)]
                types = []
                if test in cls:
                    types.append('cls')
                elif test in entity:
                    types.append('entity')
                elif test in bio:
                    types.append('bio')


        json.dump(results, out, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()
