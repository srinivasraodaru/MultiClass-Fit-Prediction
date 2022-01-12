#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np


dataset = pd.read_json('../data/renttherunway_final_data.json.gz', lines=True)
dataset = dataset.dropna()


train_data, validation_data, test_data = np.split(dataset.sample(frac=1, random_state=42), 
                                                  [int(.7*len(dataset)), int(.85*len(dataset))])


