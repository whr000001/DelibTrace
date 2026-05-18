import os
import json
from tqdm import tqdm
from utils import obtain_json, obtain_response
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--dataset', type=str, default='scruples')
args = parser.parse_args()

dataset_name = args.dataset
assert dataset_name in ['scruples', 'article']


def obtain_raw_facts(text):
    with open('prompts/facts_initial.txt') as f:
        prompt = f.read()
    inputs = prompt.replace('<===text===>', text)
    raw_facts = obtain_response(inputs, model='gpt-5', temperature=0.0, n=1)

    raw_facts = obtain_json(raw_facts)
    return raw_facts


def obtain_refined_facts(text, raw_facts):
    with open('prompts/facts_refine.txt') as f:
        prompt = f.read()
    inputs = prompt.replace('<===text===>', text)
    fact_text = ''
    for item in raw_facts:
        fact_text += f'{item}\n'
    inputs = inputs.replace('<===facts===>', fact_text)
    refined_facts = obtain_response(inputs, model='gpt-5', temperature=0.0, n=1)
    refined_facts = obtain_json(refined_facts)
    return refined_facts


def obtain_important_facts(question, refined_facts):
    with open('prompts/facts_select.txt') as f:
        prompt = f.read()
    inputs = prompt.replace('<===question===>', question)
    fact_text = ''
    for item in refined_facts:
        fact_text += f'{item}\n'
    inputs = inputs.replace('<===facts===>', fact_text)
    important_facts = obtain_response(inputs, model='gpt-5', temperature=0.0, n=1)
    important_facts = obtain_json(important_facts)
    return important_facts


def main():
    data = json.load(open(f'datasets/{dataset_name}.json'))
    save_path = f'data/facts_{dataset_name}.json'
    if os.path.exists(save_path):
        out = json.load(open(save_path))
    else:
        out = []

    for item in tqdm(data[len(out):]):
        raw_facts = obtain_raw_facts(item['description'])
        refined_facts = obtain_refined_facts(item['description'], raw_facts)
        important_facts = obtain_important_facts(item['question'], refined_facts)
        out.append([raw_facts, refined_facts, important_facts])
        json.dump(out, open(save_path, 'w'), indent=2)


if __name__ == '__main__':
    main()
