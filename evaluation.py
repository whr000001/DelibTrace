import json
import os.path
from tqdm import tqdm
from utils import obtain_response, obtain_json
from argparse import ArgumentParser
import os
import time

parser = ArgumentParser()
parser.add_argument('--dataset', type=str, default='article')
parser.add_argument('--model', type=str, default='gpt')
parser.add_argument('--structure', type=str, default='initial')
args = parser.parse_args()

dataset_name = args.dataset
model = args.model
structure = args.structure

assert structure in ['initial', 'full', 'tree', 'line']


def evaluate_facts(text, fact):
    with open('prompts/evaluate_fact.txt') as f:
        prompt = f.read()
    inputs = prompt.replace('<===text===>', text)
    fact_text = ''
    for index, item in enumerate(fact):
        fact_text += f'{index}: {item}\n'
    inputs = inputs.replace('<===facts===>', fact_text)
    obtained_facts = obtain_response(inputs, temperature=0.0, model='gpt-5')
    obtained_facts = obtain_json(obtained_facts)
    # print(obtained_facts)
    return obtained_facts


def evaluate_stance(text, question):
    with open('prompts/evaluate_stance.txt') as f:
        prompt = f.read()
    inputs = prompt.replace('<===text===>', text)
    inputs = inputs.replace('<===question===>', question)
    return obtain_response(inputs, temperature=0.0, model='gpt-5')


def evaluate(name):
    dataset = json.load(open(f'datasets/{dataset_name}.json'))
    facts = json.load(open(f'data/facts_{dataset_name}.json'))
    perspective = json.load(open(f'data/perspective_{dataset_name}.json'))
    indices = list(range(len(dataset)))
    # indices = list(range(1))

    save_path = f'evaluate_res/facts_{dataset_name}_{model}_{name}.json'

    if os.path.exists(save_path):
        out = json.load(open(save_path))
    else:
        out = []

    for index in tqdm(indices[len(out):], desc=name):
        _, refined_facts, _ = facts[index]
        if not os.path.exists(f'content/{dataset_name}_{model}/{index}/{name}.json'):
            out.append(None)
            continue
        data = json.load(open(f'content/{dataset_name}_{model}/{index}/{name}.json'))
        obtained_facts = [evaluate_facts(item, refined_facts) for item in data]
        obtained_stance = [evaluate_stance(item, dataset[index]['question']) for item in data]
        out.append({
            'facts': obtained_facts,
            'stance': obtained_stance
        })
        json.dump(out, open(save_path, 'w'), indent=2)


def main():
    if structure == 'initial':
        evaluate('initial')
    else:
        for num_round in range(3):
            evaluate(f'debate_{structure}_{num_round}')


if __name__ == '__main__':
    main()
