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

lines = ["Algorithm seed time result\n"]
for _ in [0,1,2,3,4]:
    combinations = get_all_combinations(a, b, c)
    for combination in combinations:
        lines.append(f'{combination[0]} {combination[1]} {combination[2]}\n')

with open('experiments.txt', 'w') as f:
    f.writelines(lines)

print(combinations)
print(len(combinations))