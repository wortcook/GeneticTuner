from keras_tuner.api_export import keras_tuner_export
from keras_tuner.engine import oracle as oracle_module
from keras_tuner.engine import trial as trial_module
from keras_tuner.engine import tuner as tuner_module

class GeneticPool:
    def __init__(
        self,
        init_size,
        stable_size,
        cull_rate,
        mutate_prob,
        dominant_crossover_prob,
        chromo_creator,
    ):
        self._init_size = init_size
        self._stable_size = stable_size
        self._cull_rate = cull_rate
        self._mutate_prob = mutate_prob
        self._dominant_crossover_prob = dominant_crossover_prob
        self._chromo_creator = chromo_creator
        
        #populate pool from chromo_creator
        self._pool = [self._chromo_creator() for _ in range(self._init_size)]
        self._used_pool = []
        
    def get_chromo(self):
        #if the pool is empty, repopulate it
        if len(self._pool) == 0:
            self._repopulate_pool()
        else:
            chromo = self._pool.pop(self._pool_index)
            self._used_pool.append(chromo)
            return chromo
        
    def _repopulate_pool(self):
        #sort the used pool by fitness
        self._used_pool.sort(key=lambda x: x.score, reverse=(self.objective.direction == "max"))
        
        #remove the worst chromos based on cull rate
        self._used_pool = self._used_pool[:int(len(self._used_pool) * (1 - self._cull_rate))]

        #do we have an odd number of chromos?
        if len(self._used_pool) % 2 == 1:
            #add the worst chromo back in
            self._pool.append(self._used_pool.pop())        

        
        
        

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
        
    def populate_space(self, trial_id):
        
        values = None
        
        if len(self.start_order) == 0:
            # Use all default values for the first trial.
            self._ordered_ids.insert(trial_id)
            hps = self.get_space()
            values = {
                hp.name: hp.default
                for hp in self.get_space().space
                if hps.is_active(hp)
            }
        
    