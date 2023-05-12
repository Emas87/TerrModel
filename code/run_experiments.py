import logging
from MCTS import MCTS
from RHEA import RHEA
from TerrEnv import TerrEnv
from configure_logging import configure_logging

# Configure the shared logger
logger = configure_logging('run_experiments.log')

if __name__ == "__main__":
    i = 2
    experiment_file = f"experiments{i}.txt"
    game_env = TerrEnv()
    mcts = MCTS(game_env, exploration=3)
    rhea = RHEA(game_env, horizon=1, rollouts_per_step=2)
    #mcts = None
    #rhea = None
    with open(experiment_file) as f:
        lines = f.readlines()
    # format: Algorithm seed time result

    for i in range(len(lines)):
        experiment = lines[i].strip().split(" ")
        if len(experiment) > 3: # already done
            continue
        algorithm = experiment[0]
        seed = experiment[1]
        max_time = experiment[2]
        logger.info(f'Running {algorithm} {seed} {max_time}')
        #lunch experiment
        if algorithm == 'mcts':
            result = mcts.run(int(seed), int(max_time))
            lines[i] = lines[i].strip() + f' {str(result)}\n'
        elif algorithm == 'rhea':
            result = rhea.run(int(seed), int(max_time))
            lines[i] = lines[i].strip() + f' {str(result)}\n'
        with open(experiment_file, "w") as f:
            f.writelines(lines)

    
    
