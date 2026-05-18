import os
import json
from tqdm import tqdm
from argparse import ArgumentParser
from utils import obtain_json, obtain_response
import time

parser = ArgumentParser()
parser.add_argument('--dataset', type=str, default='scruples')
args = parser.parse_args()

dataset_name = args.dataset
assert dataset_name in ['scruples', 'article']


def obtain_perspective(refined_facts, important_facts):
    with open('prompts/perspective.txt') as f:
        prompt = f.read()
    fact_text = ''
    for index, (r_fact, i_fact) in enumerate(zip(refined_facts, important_facts)):
        i_fact = '(Important)' if i_fact else '(Not Important)'
        fact_text += f'{index} {i_fact}: {r_fact}\n'
    inputs = prompt.replace('<===facts===>', fact_text)
    perspective = obtain_response(inputs, model='gpt-5', temperature=0.0, n=1)
    perspective = obtain_json(perspective)
    return perspective


def main():
    data = json.load(open(f'datasets/{dataset_name}.json'))
    facts = json.load(open(f'data/facts_{dataset_name}.json'))

    indices = list(range(len(data)))

    save_path = f'data/perspective_{dataset_name}.json'
    if os.path.exists(save_path):
        out = json.load(open(save_path))
    else:
        out = []

    for index in tqdm(indices[len(out):]):
        _, refined_facts, important_facts = facts[index]
        perspective = obtain_perspective(refined_facts, important_facts)
        out.append(perspective)
        json.dump(out, open(save_path, 'w'), indent=2)


if __name__ == '__main__':
    main()
