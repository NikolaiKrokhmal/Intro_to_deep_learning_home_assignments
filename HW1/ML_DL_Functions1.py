import numpy
import numpy as np
def ID1():
    '''
        Write your personal ID here.
    '''
    # Insert your ID here
    return 320717184
def ID2():
    '''
        Only If you were allowed to work in a pair will you fill this section and place the personal id of your partner otherwise leave it zeros.
    '''
    # Insert your ID here
    return 000000000

def LeastSquares(X,y):
  '''
    Calculates the Least squares solution to the problem
    X*theta=y using the least squares method
    :param X: input matrix
    :param y: input vector
    :return: theta = (Xt*X)^(-1) * Xt * y 
  '''
  xTrans = numpy.transpose(X)
  multi = numpy.matmul(xTrans, X)
  inverted = numpy.linalg.inv(multi)
  theta = np.matmul(inverted, xTrans)
  theta = np.matmul(theta, y)
  return theta

def classification_accuracy(model,X,s):
  '''
    calculate the accuracy for the classification problem
    :param model: the classification model class
    :param X: input matrix
    :param s: input ground truth label
    :return: accuracy of the model
  '''
  predictions = model.predict(X)
  numElements = len(predictions)
  corrCount = 0
  for indx in range(numElements):
    if predictions[indx] == s[indx]:
      corrCount += 1
  accuracy = 100 * corrCount/numElements
  print(accuracy)
  return accuracy

def linear_regression_coeff_submission():
  '''
    copy the values from your notebook into a list in here. make sure the values
    seperated by commas
    :return: list of coefficiants for the linear regression problem.  
  '''
  return [[-0.03994061,  0.0455526,  -0.03936638,  0.03427444, -0.01961861, -0.01942904,
  0.05019078, -0.00753476,  0.02212707, -0.03843958,  0.02503965,  0.03756372,
  0.08520227,  0.15800111,  0.76939406,  0.03189247,  0.03234572,  0.01879143,
  0.00849401,  0.00174708,  0.05549413,  0.03989288,  0.00147247, -0.03236676,
 -0.0108228,   0.00901534, -0.00746466, -0.03030227]]

def linear_regression_intrcpt_submission():
  '''
    copy the intercept value from your notebook into here.
    :return: the intercept value.  
  '''
  return -2.0969551560001232e-16

def classification_coeff_submission():
  '''
    copy the values from your notebook into a list in here. make sure the values
    seperated by commas
    :return: list of coefficiants for the classification problem.  
  '''
  return [[-0.32873109, -0.18290037,  0.32112097,  0.01002806, -0.19957014, -0.6989196,
  -0.03445131,  0.08678432,  0.0342301,   0.2478134,  -0.77990261,  0.01896418,
   0.07066216,  1.01556249,  2.6758546,  -0.49167322, -0.10891771, -0.05349483,
  -0.05299106, -0.06942903,  0.01004557,  0.07598789, -0.07576343, -0.26217972,
  -0.43756114, -0.0491568,   0.13888414,  0.38170375]]

def classification_intrcpt_submission():
  '''
    copy the intercept value from your notebook into here.
    :return: the intercept value.  
  '''
  return [0.24507608]

def classification_classes_submission():
  '''
    copy the classes values from your notebook into a list in here. make sure the values
    seperated by commas
    :return: list of classes for the classification problem.  
  '''
  return [0, 1]


