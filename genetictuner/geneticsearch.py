from keras_tuner.engine import oracle as oracle_module
from keras_tuner.engine import trial as trial_module
from keras_tuner.engine import tuner as tuner_module

from typing import List

import random


#####
## TO FIX
## THE CROSSOVER AND MUTATE METHODS IN THE GENETICPOOL CLASS
## NEED TO DEAL WITH TRIALS SO THAT WE CAN USE THE SCORE TO SORT
## BUT THE POPULATE METHOD RETURNS A DICTIONARY
#####

class GeneticPool:
    def __init__(
        self,
        oracle : oracle_module.Oracle,
        trial_creator,
        objective : oracle_module.Objective,
        init_size : int = 100,
        stable_size : int = 20,
        dominant_crossover_prob : float = 0.5,
        backfill_rate : float = 0.2,
        cull_rate : float = 0.2,
        mutate_prob : float = 0.1,
    ):
        self._oracle = oracle
        self._init_size : int = init_size
        self._stable_size : int = stable_size
        self._cull_rate : float = cull_rate
        self._mutate_prob : float = mutate_prob
        self._dominant_crossover_prob : float = dominant_crossover_prob
        self._objective = objective
        self._backfill_rate = backfill_rate
        self._trial_creator = trial_creator
        
        #populate pool from chromo_creator
        self._pool :List = [ self._trial_creator() for _ in range(self._init_size)]
        self._used_pool : List = []
        
    def get_trial(self, trial_id):
        #if the pool is empty, repopulate it
        if len(self._pool) == 0:
            self._repopulate_pool()

        #get a trial from the pool            
        trial = self._pool.pop()
        self._used_pool.append(trial_id)
        return trial
        
    def _repopulate_pool(self):
        
        used_trials = [self._oracle.trials[trial_id] for trial_id in self._used_pool]
        used_trials.sort(key=lambda x: x.score, reverse=(self._objective.direction == "max"))
        
        #remove the worst chromos based on cull rate
        used_trials = used_trials[:int(len(used_trials) * (1 - self._cull_rate))]

        #do we have an odd number of chromos?
        if len(used_trials) % 2 == 1:
            self._pool.append(used_trials.pop())
            
        #crossover the chromos
        cross_pool = []
        while len(used_trials) > 1:
            trial1 = used_trials.pop()
            trial2 = used_trials.pop()
            
            if trial1.score < trial2.score:
                trial1, trial2 = trial2, trial1
            cross_pool.append(self._crossover(trial1, trial2))
            
        #mutate the chromos
        for trial in cross_pool:
            self._mutate(trial)
            
        #copy the cross_pool to the pool
        self._pool = cross_pool.copy()
        
        #backfill the pool
        self._backfill_pool()
        self._used_pool = []
        
    def _crossover(self, trial1:trial_module.Trial, trial2:trial_module.Trial):
        #walk through the genes of the chromos and randomly select one
        #from either chromo1 or chromo2
        
        new_trial = self._trial_creator()

        for hp in new_trial.keys():
            if random.random() < self._dominant_crossover_prob:
                new_trial[hp] = trial1.hyperparameters.values[hp]
            else:
                new_trial[hp] = trial2.hyperparameters.values[hp]
                
        return new_trial
    
    def _mutate(self, trial):
        for hp in trial.keys():
            if random.random() < self._mutate_prob:
                trial[hp] = self._oracle.hyperparameters[hp].random_sample()

    def _backfill_pool(self):
        #if the pool is less than the stable target size
        #backfill with random trials
        if(len(self._pool) < self._stable_size):
            while len(self._pool) < self._stable_size:
                self._pool.append(self._trial_creator())
        #else we don't want to go over the initial size
        #so backfill with a percentage of the current pool size
        elif len(self._pool) < self._init_size:
            new_len = int(len(self._pool) * (1 + self._backfill_rate))
            while len(self._pool) < new_len:
                self._pool.append(self._trial_creator())
            



class GeneticSearchOracle(oracle_module.Oracle):
    def __init__(
        self,
        objective=None,
        max_trials=100,
        seed=None,
        hyperparameters=None,
        allow_new_entries=True,
        tune_new_entries=True,
        max_retries_per_trial=0,
        max_consecutive_failed_trials=3,
        initial_population_size=20,
        stable_population_size=20,
        cull_rate=0.2,
        mutate_prob=0.1,
        backfill_rate=0.1,
        dominant_crossover_prob=0.5,
    ):
        super().__init__(
            objective=objective,
            max_trials=max_trials,
            seed=seed,
            hyperparameters=hyperparameters,
            allow_new_entries=allow_new_entries,
            tune_new_entries=tune_new_entries,
            max_retries_per_trial=max_retries_per_trial,
            max_consecutive_failed_trials=max_consecutive_failed_trials,
        )
        
        self._pool = GeneticPool(
            oracle = self,
            trial_creator=self._random_values,
            objective=self.objective,
            init_size=initial_population_size,
            stable_size=stable_population_size,
            cull_rate=cull_rate,
            mutate_prob=mutate_prob,
            backfill_rate=backfill_rate,
            dominant_crossover_prob=dominant_crossover_prob,
        )
        
    def populate_space(self, trial_id):
        values = self._pool.get_trial(trial_id)
        if values is None:
            return {"status": trial_module.TrialStatus.STOPPED, "values": None}
        return {"status": trial_module.TrialStatus.RUNNING, "values": values}
    
class GeneticSearch(tuner_module.Tuner):
    def __init__(
        self,
        hypermodel,
        objective,
        max_trials,
        seed=None,
        hyperparameters=None,
        allow_new_entries=True,
        tune_new_entries=True,
        max_retries_per_trial=0,
        max_consecutive_failed_trials=3,
        initial_population_size=20,
        stable_population_size=20,
        cull_rate=0.2,
        mutate_prob=0.1,
        backfill_rate=0.1,
        dominant_crossover_prob=0.5,
        **kwargs
    ):
        oracle = GeneticSearchOracle(
            objective=objective,
            max_trials=max_trials,
            seed=seed,
            hyperparameters=hyperparameters,
            allow_new_entries=allow_new_entries,
            tune_new_entries=tune_new_entries,
            max_retries_per_trial=max_retries_per_trial,
            max_consecutive_failed_trials=max_consecutive_failed_trials,
            initial_population_size=initial_population_size,
            stable_population_size=stable_population_size,
            cull_rate=cull_rate,
            mutate_prob=mutate_prob,
            backfill_rate=backfill_rate,
            dominant_crossover_prob=dominant_crossover_prob,
        )
        super(GeneticSearch, self).__init__(
            hypermodel=hypermodel,
            oracle=oracle,
            **kwargs
        )
    