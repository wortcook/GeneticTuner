from keras_tuner.api_export import keras_tuner_export
from keras_tuner.engine import oracle as oracle_module
from keras_tuner.engine import trial as trial_module
from keras_tuner.engine import tuner as tuner_module

class GeneticPool:
    def __init__(
        self,
        init_size,
        stable_size,
        mutate_prob,
        dominant_crossover_prob,
    ):
        self.pool = []
        
        
        

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
        trials_per_generation=10,
        initial_population_size=20,
        stable_population_size=20,
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
        
    