import os
import json
from tqdm import tqdm
from utils import obtain_json, obtain_response
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('--dataset', type=str, default='scruples')
parser.add_argument('--model', type=str, default='gpt')
parser.add_argument('--structure', type=str, default='full')
args = parser.parse_args()

dataset_name = args.dataset
model = args.model
structure = args.structure
assert dataset_name in ['scruples', 'article']
assert model in ['gpt', 'gemini', 'qwen']
assert structure in ['full', 'tree', 'line']

if model == 'gpt':
    model_name = 'gpt-4.1'
elif model == 'gemini':
    model_name = 'gemini-3-flash-preview'
elif model == 'qwen':
    model_name = 'qwen3.5-flash'
else:
    raise KeyError


def discussion_preprocess():
    import random
    random.seed(20260601)
    perspective = json.load(open(f'data/perspective_{dataset_name}.json'))
    out = []
    for item in perspective:
        random_index = list(range(8))
        random.shuffle(random_index)
        out.append(random_index)
    json.dump(out, open(f'data/random_index_{dataset_name}.json', 'w'))


def check_available(text, question, facts, pers):
    if not isinstance(text, str):
        return False
    if not isinstance(question, str):
        return False
    if not isinstance(facts, list):
        return False
    if not isinstance(pers, list):
        return False
    if len(pers) != 4:
        return False
    return True


def obtain_discussion_initial_each(question, fact_text, answer):
    with open('prompts/discussion_initial.txt') as f:
        prompt = f.read()
    inputs = prompt.replace('<===facts===>', fact_text)
    inputs = inputs.replace('<===answer===>', answer)
    inputs = inputs.replace('<===question===>', question)
    return obtain_response(inputs, model=model_name, temperature=1.2, n=1)


def obtain_discussion_initial(question, facts, perspective, save_dir, index):
    save_dir = f'{save_dir}/{index}'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_path = f'{save_dir}/initial.json'
    if os.path.exists(save_path):
        return
    out = []
    for item in perspective:
        fact_text = ''
        for _ in item:
            fact_text += f'{facts[_]}\n'
        out.append(obtain_discussion_initial_each(question, fact_text, 'yes'))
        out.append(obtain_discussion_initial_each(question, fact_text, 'no'))
    json.dump(out, open(save_path, 'w'), indent=2)


def discussion_initial():
    dataset = json.load(open(f'datasets/{dataset_name}.json'))
    facts = json.load(open(f'data/facts_{dataset_name}.json'))
    perspective = json.load(open(f'data/perspective_{dataset_name}.json'))

    indices = list(range(len(dataset)))
    # indices = list(range(200))

    save_dir = f'content/{dataset_name}_{model}'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for index in tqdm(indices, desc='initial'):
        text = dataset[index]['description']
        question = dataset[index]['question']
        _, refined_facts, _ = facts[index]
        pers = perspective[index]
        if not check_available(text, question, refined_facts, pers):
            continue
        obtain_discussion_initial(question, refined_facts, pers, save_dir, index)


def discussion_continue(initial, settings, edges, question, save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    with open('prompts/discussion_continue.txt') as f:
        debate_prompt = f.read()
    setting_describe = json.load(open('prompts/discussion_setting.json', encoding='utf-8'))

    length = len(edges)

    debate_round = 3
    previous = initial.copy()
    for _ in range(debate_round):
        save_path = f'{save_dir}/debate_{structure}_{_}.json'
        if os.path.exists(save_path):
            current = json.load(open(save_path))
        else:
            current = []
            for i in range(length):
                previous_view = previous[i]
                if previous_view is None:
                    previous_view = ''
                other_views = ''
                other_index = 0
                for j in edges[i]:
                    other_views += f'View {other_index + 1}: {previous[j]}\n'
                    other_index += 1
                inputs = debate_prompt.replace('<===others===>', other_views)
                inputs = inputs.replace('<===previous===>', previous_view)
                inputs = inputs.replace('<===setting===>', setting_describe[settings[i]])
                inputs = inputs.replace('<===question===>', question)
                current.append(obtain_response(inputs, model_name, temperature=1.2, n=1))
            json.dump(current, open(save_path, 'w'), indent=2)
        previous = current


def discussion():
    data = json.load(open(f'datasets/{dataset_name}.json'))
    indices = list(range(len(data)))
    # indices = list(range(400))
    save_dir = f'content/{dataset_name}_{model}'
    random_indices = json.load(open(f'data/random_index_{dataset_name}.json'))
    for index in tqdm(indices, desc='discussing..'):
        question = data[index]['question']
        if not os.path.exists(f'{save_dir}/{index}/initial.json'):
            continue
        initial = json.load(open(f'{save_dir}/{index}/initial.json'))
        ri = random_indices[index]
        initial = [initial[_] for _ in ri]

        initial = [_ if _ is not None else ' ' for _ in initial]

        length = len(initial)

        settings = ['default'] * length

        edges = []
        if structure == 'full':
            edges = [[j for j in range(length) if i != j] for i in range(length)]
        elif structure == 'line':
            for i in range(length):
                if i == 0:
                    edges.append([i + 1])
                elif i == length - 1:
                    edges.append([i - 1])
                else:
                    edges.append([i - 1, i + 1])
        elif structure == 'tree':
            for i in range(length):
                edge = []
                father = (i + 1) // 2 - 1
                left = (i + 1) * 2 - 1
                right = (i + 1) * 2 + 1 - 1
                if i != father and 0 <= father < length:
                    edge.append(father)
                if i != left and 0 <= left < length:
                    edge.append(left)
                if i != right and 0 <= right < length:
                    edge.append(right)
                edges.append(edge)
        else:
            raise KeyError

        discussion_continue(initial, settings, edges, question, f'{save_dir}/{index}')


def main():
    discussion_preprocess()
    discussion_initial()
    discussion()


if __name__ == '__main__':
    main()
