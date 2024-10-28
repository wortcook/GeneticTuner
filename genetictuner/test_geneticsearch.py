import random
import unittest

import keras
import numpy as np

from keras_tuner.engine import hypermodel
from keras_tuner.engine import trial as trial_module


import geneticsearch

class TestGeneticSearch(unittest.TestCase):
    def test_geneticsearch_one_trial(self):
        
        def build_model(hp):
            model = keras.Sequential()
            model.add(keras.layers.InputLayer(shape=(4,)))
            model.add(keras.layers.Dense(hp.Choice('dense1',[2,4,6,8])))
            model.add(keras.layers.Dense(hp.Choice('dense2',[2,4,6,8])))
            model.add(keras.layers.Dense(1))
            model.compile(optimizer='adam', loss='mse')
            return model
        
        objective = 'loss'
        tuner = geneticsearch.GeneticSearch(
            hypermodel=build_model,
            objective=objective,
            max_trials=50,
            seed=random.randint(1, 100),
            hyperparameters=None,
            allow_new_entries=True,
            tune_new_entries=True,
            max_retries_per_trial=0,
            max_consecutive_failed_trials=3,
            initial_population_size=10,
            stable_population_size=5,
            cull_rate=0.4,
            mutate_prob=0.1,
            backfill_rate=0.1,
            dominant_crossover_prob=0.5,
        )
        
        
        
        x = np.array(
            [[0,0,0,0],
             [0,0,0,1],
             [0,0,1,0],
             [0,0,1,1],
             [0,1,0,0],
             [0,1,0,1],
             [0,1,1,0],
             [0,1,1,1],
             [1,0,0,0],
             [1,0,0,1],
             [1,0,1,0],
             [1,0,1,1],
             [1,1,0,0],
             [1,1,0,1],
             [1,1,1,0],
             [1,1,1,1]]
        )
        
        y = np.array([[0],[1],[1],[0],[1],[0],[0],[1],[1],[0],[0],[1],[0],[1],[1],[0]])
        
        
        tuner.search(x=x, y=y)
        
        tuner.results_summary()

if __name__ == '__main__':
    unittest.main()