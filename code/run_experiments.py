import logging
from MCTS import MCTS
from RHEA import RHEA
from TerrEnv import TerrEnv
from configure_logging import configure_logging

# Configure the shared logger
#logger = configure_logging('run_experiments.log')
TIMEOUT = 500
if __name__ == "__main__":
    jinitial = 1
    for j in range(jinitial, 11):
        experiment_file = f"experiments{j}.txt"
        game_env = TerrEnv()
        mcts = MCTS(game_env, exploration=3)
        rhea = RHEA(game_env, horizon=2)
        #mcts = None
        #rhea = None
        with open(experiment_file) as f:
            lines = f.readlines()
        # format: algorithm seed time result

        for i in range(len(lines)):
            experiment = lines[i].strip().split(" ")
            if len(experiment) > 3: # reading first line
                continue
            algorithm = experiment[0]
            seed = experiment[1]
            max_time = experiment[2]
            #logger.info(f'Running {algorithm} {seed} {max_time}')
            while True:
                #launch experiment
                if algorithm == 'mcts':
                    result = mcts.run(int(seed), int(max_time), timeout=TIMEOUT)
                    if result == 0:
                        continue
                    lines[i] = lines[i].strip() + f' {str(result)}\n'
                    break
                elif algorithm == 'rhea':
                    result = rhea.run(int(seed), int(max_time), timeout=TIMEOUT)
                    if result == 0:
                        continue
                    lines[i] = lines[i].strip() + f' {str(result)}\n'
                    break
            with open(experiment_file, "w") as f:
                f.writelines(lines)

    
    
