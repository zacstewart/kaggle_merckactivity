import numpy as np
from datasets import *
from sklearn.neighbors import *

if __name__ == '__main__':
  for dataset in range(1, 16):
    print "Loading file..."
    training_matrix, training_header = readFile("data/TrainingSet/ACT%(dataset)i_competition_training.csv" % {'dataset': dataset})
    print type(training_matrix)
    np.random.shuffle(training_matrix)

    construct, cv = training_matrix[:80,:], training_matrix[80:,:]

    construct_ids = construct[:, 0]
    construct_y   = construct[:, 1].astype(float)
    construct_X   = construct[:, 2:].astype(int)
    cv_ids = cv[:, 0]
    cv_y   = cv[:, 1].astype(float)
    cv_X   = cv[:, 2:].astype(int)

    print 'Training...'
    model = KNeighborsRegressor()
    model.fit(construct_X, construct_y)
    print 'Score: ' + str(model.score(cv_X, cv_y))
