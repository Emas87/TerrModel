import itertools
import random

def get_all_combinations(a, b, c):
    combinations = list(itertools.product(a, b, c))
    random.shuffle(combinations)
    return combinations

# example usage
a = ('mcts', 'rhea')
b = (0,1,2,3,4,5,6,7)
c = (2, 3)

for i in [1,2,3,4,5,6,7,8,9,10]:
    lines = ["algorithm seed time result\n"]
    combinations = get_all_combinations(a, b, c)
    for combination in combinations:
        lines.append(f'{combination[0]} {combination[1]} {combination[2]}\n')

    with open(f'experiments{i}.txt', 'w') as f:
        f.writelines(lines)

    print(combinations)
    print(len(combinations))