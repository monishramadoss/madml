import numpy as np
import madml.regularization as reg


def onehot(labels):
    y = np.zeros([labels.size, np.max(labels) + 1])
    y[range(labels.size), labels] = 1.
    return y

def softmax(X):
    eX = np.exp((X.T - np.max(X, axis=1)).T)
    return (eX.T / eX.sum(axis=1)).T



def accuracy(y_true, y_pred):
    return np.mean(y_pred == y_true)

def regularization(model, reg_type='l2', lam=1e-3):
    reg_types = dict(l1=reg.l1_reg, l2=reg.l2_reg)
    return np.sum([ reg_types[reg_type](model[k], lam)  for k in model.keys()  if k.startswith('W') ])


def cross_entropy(model, y_pred, y_train, lam=1e-3):
    m = y_pred.shape[0]
    prob = softmax(y_pred)
    log_like = -np.log(prob[range(m), y_train])
    data_loss = np.sum(log_like) / m
    reg_loss = regularization(model, reg_type='l2', lam=lam)
    return data_loss + reg_loss

def d_cross_entropy(y_pred, y_train):
    m = y_pred.shape[0]
    grad_y = softmax(y_pred)
    grad_y[range(m), y_train] -= 1.
    grad_y /= m
    return grad_y


def hinge_loss(model, y_pred, y_train, lam=1e-3, delta=1):
    m = y_pred.shape[0]
    margins = (y_pred.T - y_pred[range(m), y_train]).T + delta
    margins[margins < 0] = 0
    margins[range(m), y_train] = 0
    data_loss = np.sum(margins) / m
    reg_loss = regularization(model, reg_type='l2', lam=lam)
    return data_loss + reg_loss


def dhinge_loss(y_pred, y_train, margin=1):
    m = y_pred.shape[0]
    margins = (y_pred.T - y_pred[range(m), y_train]).T + 1.
    margins[range(m), y_train] = 0
    grad_y = (margins > 0).astype(float)
    grad_y[range(m), y_train] = -np.sum(grad_y, axis=1)
    grad_y /= m
    return grad_y


def mean_squared_loss(model, y_pred, y_train, lam=1e-3):
    m = y_pred.shape[0]
    data_loss = 0.5 * np.sum((onehot(y_train) - y_pred)**2) / m
    reg_loss = regularization(model, reg_type='l2', lam=lam)
    return data_loss + reg_loss


def d_mean_squared_loss(y_pred, y_train):
    m = y_pred.shape[0]
    grad_y = y_pred - onehot(y_train)
    grad_y /= m
    return grad_y


def l2_regression(model, y_pred, y_train, lam=1e-3):
    m = y_pred.shape[0]
    data_loss = 0.5 * np.sum((y_train - y_pred)**2) / m
    reg_loss = regularization(model, reg_type='l2', lam=lam)
    return data_loss + reg_loss


def d_l2_regression(y_pred, y_train):
    m = y_pred.shape[0]
    grad_y = y_pred - y_train.reshape(-1, 1)
    grad_y /= m
    return grad_y


def l1_regression(model, y_pred, y_train, lam=1e-3):
    m = y_pred.shape[0]
    data_loss = np.sum(np.abs(y_train - y_pred)) / m
    reg_loss = regularization(model, reg_type='l2', lam=lam)
    return data_loss + reg_loss


def d_l1_regression(y_pred, y_train):
    m = y_pred.shape[0]
    grad_y = np.sign(y_pred - y_train.reshape(-1, 1))
    grad_y /= m
    return grad_y