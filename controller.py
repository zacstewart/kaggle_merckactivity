import numpy as np
from datasets import *
from sklearn.neighbors import *
from sklearn.linear_model import *
from sklearn.svm import SVR
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.gaussian_process import GaussianProcess
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

if __name__ == '__main__':
  suffix = ''
  submission = np.array(['MOLECULE','Prediction'], dtype=str)
  for dataset in range(1, 2):
    construct_ids, construct_X, construct_y, \
        cv_ids, cv_X, cv_y = \
        loadTrainingSet('data/TrainingSet/ACT%(dataset)i_competition_training%(suffix)s.csv' % {'dataset': dataset, 'suffix': suffix})
    test_ids, test_X = \
        loadTestSet('data/TestSet/ACT%(dataset)i_competition_test%(suffix)s.csv' % {'dataset': dataset, 'suffix': suffix})
    models = [
      Ridge(alpha=.5),
      LinearRegression(),
      SVR(),
      NearestCentroid(),
      GaussianProcess(),
      DecisionTreeRegressor(),
      RandomForestRegressor(),
      GradientBoostingRegressor()
    ]

    super_training_matrix = np.zeros((len(cv_y), len(models) + 1))
    super_training_matrix[:,0] = cv_y
    super_test_matrix = np.zeros((len(test_X), len(models)))

    for (i, model) in enumerate(models):
      print 'Training %(model)s...' % {'model': str(model).split('(')[0]}
      model.fit(construct_X, construct_y)
      print 'Predicting CV...'
      preds = model.predict(cv_X).flatten()
      print 'Adding predictions...'
      super_training_matrix[:, i+1] = preds
      print 'Predicting test set...'
      test_preds = model.predict(test_X).flatten()
      super_test_matrix[:, i] = test_preds

    construct_frac = int(.8 * len(super_training_matrix))
    super_construct, super_cv = super_training_matrix[:construct_frac,:], super_training_matrix[construct_frac:,:]

    super_construct_y = super_construct[:, 0]
    super_construct_X = super_construct[:, 1:]
    super_cv_y = super_cv[:, 0]
    super_cv_X = super_cv[:, 1:]

    best_score = 0.
    best_model = None
    for (i, model) in enumerate(models):
      print 'Super training %(model)s...' % {'model': str(model).split('(')[0]}
      model.fit(super_construct_X, super_construct_y)
      score = model.score(super_cv_X, super_cv_y)
      print 'Score: ' + str(score)
      if score > best_score:
        best_score = score
        best_model = model

    print 'Best model: ' + str(best_model)
    super_test_preds = best_model.predict(super_test_matrix)
    set_predictions = np.array((test_ids, test_preds)).transpose()
    submission = np.vstack((submission, set_predictions))
    np.savetxt('submission.csv', submission, delimiter=',', fmt='%s')
