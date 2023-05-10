from MCTS import MCTS
from RHEA import RHEA

if __name__ == "__main__":
    #mcts = MCTS(exploration=3)
    #rhea = RHEA(horizon=1, rollouts_per_step=2)
    mcts = None
    rhea = None
    with open("experiments.txt") as f:
        lines = f.readlines()
    # format: Algorithm seed time result

    for i in range(len(lines)):
        experiment = lines[i].strip().split(" ")
        if len(experiment) > 3: # already done
            continue
        algorithm = experiment[0]
        seed = experiment[1]
        max_time = experiment[2]
        #lunch experiment
        if algorithm == 'mcts':
            result = mcts.run(seed, max_time)
            lines[i] = lines[i].strip() + f' {str(result)}\n'
        elif algorithm == 'rhea':
            result = rhea.run(seed, max_time)
            lines[i] = lines[i].strip() + f' {str(result)}\n'
        with open("experiments.txt", "w") as f:
            f.writelines(lines)

    
    
