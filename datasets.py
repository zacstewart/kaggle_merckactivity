import csv
import numpy as np

def readFile(filename, delimiter=','):
  '''Returns a data matrix and a header list'''
  f = open(filename, 'rb')
  header = f.readline().strip().split(delimiter)
  f.close()
  matrix = np.loadtxt(filename, dtype=str, skiprows=1, delimiter=delimiter)
  return (matrix, header)
