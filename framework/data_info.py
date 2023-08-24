# get number of labels

import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
import os
import glob
import collections

#data = pd.read_csv("/Users/Hannah1/My Drive/Enigma/m_thesis/info/titles_with_labels.csv", sep='\t', lineterminator='\n')
data = pd.read_csv("/Users/Hannah1/Downloads/articles_with_labels.csv", sep='\t', lineterminator='\n')
data.fillna('', inplace=True)
data.drop(columns = ['Unnamed: 0'], inplace=True)

print(data['agg_label'].value_counts())

# titles
#0    54363
#1    45573
#-1    42634

# articles
#0    44636
#1    60508
#-1    37426

