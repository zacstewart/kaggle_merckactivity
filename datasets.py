import csv
import numpy as np
import os
import pickle as pkl
import string

def readFile(filename, delimiter=','):
  '''Returns a data matrix and a header list'''
  f = open(filename, 'rb')
  header = f.readline().strip().split(delimiter)
  f.close()
  matrix = np.loadtxt(filename, dtype=str, skiprows=1, delimiter=delimiter)
  return (matrix, header)

def loadTestSet(filename):
  cache_dir = os.path.join(os.getcwd(), 'data', 'cache', string.replace(filename.split('/')[-1], '.csv', ''))
  test_matrix, test_header = readFile(filename)
  test_ids = test_matrix[:, 0]
  test_X = test_matrix[:, 1:].astype(int)
  return (test_ids, test_X)

def loadTrainingSet(filename):
  cache_dir = os.path.join(os.getcwd(), 'data', 'cache', string.replace(filename.split('/')[-1], '.csv', ''))
  if os.path.isdir(cache_dir):
    print 'Loading pickled caches...'
    f = open(os.path.join(cache_dir, 'construct_ids.pkl'), 'rb')
    construct_ids = pkl.load(f)
    f.close()
    f = open(os.path.join(cache_dir, 'construct_X.pkl'), 'rb')
    construct_X = pkl.load(f)
    f.close()
    f = open(os.path.join(cache_dir, 'construct_y.pkl'), 'rb')
    construct_y = pkl.load(f)
    f.close()
    f = open(os.path.join(cache_dir, 'cv_ids.pkl'), 'rb')
    cv_ids = pkl.load(f)
    f.close()
    f = open(os.path.join(cache_dir, 'cv_X.pkl'), 'rb')
    cv_X = pkl.load(f)
    f.close()
    f = open(os.path.join(cache_dir, 'cv_y.pkl'), 'rb')
    cv_y = pkl.load(f)
    f.close()
  else:
    print 'Loading file...'
    training_matrix, training_header = readFile(filename)

    print 'Splitting CV...'
    np.random.shuffle(training_matrix)
    construct_frac = int(.8 * len(training_matrix))
    construct, cv = training_matrix[construct_frac:,:], training_matrix[:construct_frac,:]

    print 'Separating X/y, casting types...'
    construct_ids = construct[:, 0]
    construct_y   = construct[:, 1].astype(float)
    construct_X   = construct[:, 2:].astype(int)
    cv_ids = cv[:, 0]
    cv_y   = cv[:, 1].astype(float)
    cv_X   = cv[:, 2:].astype(int)

    print 'Dumping parts to Pickle...'
    os.makedirs(cache_dir)
    f = open(os.path.join(cache_dir, 'construct_ids.pkl'), 'wb')
    pkl.dump(construct_ids, f)
    f.close()
    f = open(os.path.join(cache_dir, 'construct_X.pkl'), 'wb')
    pkl.dump(construct_X, f)
    f.close()
    f = open(os.path.join(cache_dir, 'construct_y.pkl'), 'wb')
    pkl.dump(construct_y, f)
    f.close()
    f = open(os.path.join(cache_dir, 'cv_ids.pkl'), 'wb')
    pkl.dump(cv_ids, f)
    f.close()
    f = open(os.path.join(cache_dir, 'cv_X.pkl'), 'wb')
    pkl.dump(cv_X, f)
    f.close()
    f = open(os.path.join(cache_dir, 'cv_y.pkl'), 'wb')
    pkl.dump(cv_y, f)
    f.close()
  return (construct_ids, construct_X, construct_y, cv_ids, cv_X, cv_y)
