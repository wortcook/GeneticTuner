import random
import unittest

import keras
import numpy as np


import geneticsearch

class TestGeneticSearch(unittest.TestCase):
    def test_geneticsearch_one_trial(self):
        
        def build_model(hp):

            #build array from 2 to 100
            dense_range = [i for i in range(2, 100)]

            model = keras.Sequential()
            model.add(keras.layers.InputLayer(shape=(4,)))
            model.add(keras.layers.Dense(hp.Choice('dense1',dense_range)))
            model.add(keras.layers.Dense(hp.Choice('dense2',dense_range)))
            model.add(keras.layers.Dense(1))
            model.compile(optimizer='adam', loss='mse')
            return model
        
        objective = 'loss'
        tuner = geneticsearch.GeneticSearch(
            hypermodel=build_model,
            objective=objective,
            max_trials=200,
            seed=random.randint(1, 100),
            hyperparameters=None,
            allow_new_entries=True,
            tune_new_entries=True,
            max_retries_per_trial=0,
            max_consecutive_failed_trials=3,
            initial_population_size=20,
            stable_population_size=8,
            cull_rate=0.33333,
            mutate_prob=0.02,
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
        
        
        tuner.search(x=x, y=y, epochs=1)
        
        tuner.results_summary()

if __name__ == '__main__':
    unittest.main()