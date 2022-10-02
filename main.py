import random

import plotly
import plotly.graph_objs as go
import plotly.express as px
from more_itertools import powerset
from plotly.subplots import make_subplots
import numpy as np

from visualisation import Visualization

N = 1000
x = np.linspace(0, 1, N)
z = 20 * np.sin(2 * np.pi * 3 * x) + 100 * np.exp(x)
error = 10 * np.random.randn(N)
t = z + error

tr = 0.8
val = 0.1
ind_prm = np.random.permutation(np.arange(N))
train_ind = ind_prm[:int(tr * N)]
valid_ind = ind_prm[int(tr * N):int((val + tr) * N)]
test_ind = ind_prm[int((val + tr) * N):]
x_train, t_train, x_valid, t_valid, x_test, t_test = x[train_ind], t[train_ind], x[valid_ind], t[valid_ind], x[
    test_ind], t[test_ind]

functions = [lambda x: np.sin(x), lambda x: np.cos(x), lambda x: np.exp(x), lambda x: x, lambda x: x ** 2,
             lambda x: x ** 3, lambda x: x ** 4, lambda x: x ** 5, lambda x: x ** 6, lambda x: x ** 7, lambda x: x ** 8,
             lambda x: x ** 9, lambda x: x ** 10, lambda x: x ** 20, lambda x: x ** 30, lambda x: x ** 40,
             lambda x: x ** 50, lambda x: x ** 60, lambda x: x ** 70, lambda x: x ** 80, lambda x: x ** 90,
             lambda x: x ** 100]
indexes = [i for i in range(len(functions))]
sets = [i for i in list(powerset(indexes)) if len(i) in [2, 3, 4, 5]]
func_names = ["sin(x)", "cos(x)", "exp(x)", "x", "x^2", "x^3", "x^4", "x^5",
              "x^6", "x^7", "x^8", "x^9", "x^10", "x^20",
              "x^30", "x^40", "x^50", "x^60", "x^70", "x^80", "x^90", "x^100"]
lamb = [0., 0.01, 0.001, 0.1, 0.5, 1., 5, 10, 50, 100]
chosen_sets = random.sample(sets, 30)
chosen_lamb = [random.sample(lamb, 5) for i in range(len(chosen_sets))]


def matrix_F(x, ind):
    F = np.ones((1, len(x)))
    for i in ind:
        F = np.append(F, [functions[i](x)], axis=0)
    return F.T


def learn(F, t, l):
    I = np.eye(F.shape[1])
    I[0][0] = 0
    return ((np.linalg.pinv(F.T.dot(F) + l * I)).dot(F.T)).dot(t)


def error(W, t, x, ind):
    F = matrix_F(x, ind)
    return (1 / 2) * sum((W.dot(F.T)) - t) ** 2


def toFixed(numObj, digits=0):
    return f"{numObj:.{digits}f}"


def get_name(functions_names, set, W):
    str = f"y = {toFixed(W[0], 2)}"
    for j in range(1, len(set) + 1):
        if W[j] > 0:
            str += f" + {toFixed(W[j], 2)}{functions_names[set[j - 1]]}"
        else:
            str += f" - {toFixed(abs(W[j]), 2)}{functions_names[set[j - 1]]}"
    return str


def get_names(functions_names, min_errors_sets, l_lst):
    names = []
    for i in range(10):
        W = learn(matrix_F(x_train, min_errors_sets[i]), t_train, l_lst[i])
        str = f"y = {toFixed(W[0], 2)}"
        for j in range(1, len(min_errors_sets[i]) + 1):
            if W[j] > 0:
                str += f" + {toFixed(W[j], 2)}{functions_names[min_errors_sets[i][j - 1]]}"
            else:
                str += f" - {toFixed(abs(W[j]), 2)}{functions_names[min_errors_sets[i][j - 1]]}"
        names.append(str)
    return np.array(names)


def create_graphics(x, z, t, title, test_error, la, best_func, name=None,
                    path2save=None):
    weights = learn(matrix_F(x_train, best_func), t_train, la)
    F = matrix_F(x, best_func)
    y = F.dot(weights)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, name='y(x, w)'))
    fig.add_trace(go.Scatter(x=x, y=z, name='z(x)'))
    fig.add_trace(go.Scatter(x=x, y=t, mode="markers", name='t(x)'))
    fig.update_layout(title=f"{title}, test error = {test_error}, {get_name(func_names, best_func, weights)}")
    fig.show()
    fig.write_html(f"{path2save}/{name}.html")


error_valid = []
error_test = []
for i in chosen_sets:
    F = matrix_F(x_train, i)
    for l in chosen_lamb[chosen_sets.index(i)]:
        W = learn(F, t_train, l)
        error_valid.append(error(W, t_valid, x_valid, i))
        error_test.append(error(W, t_test, x_test, i))
min_error_valid = np.array(sorted(error_valid)[:10])
min_errors_index = [error_valid.index(i) for i in min_error_valid]
l_lst = np.array([chosen_lamb[i // 5][i % 5] for i in min_errors_index])
min_errors_sets = [chosen_sets[i // 5] for i in min_errors_index]
min_error_test = np.array([error_test[i] for i in min_errors_index])

create_graphics(x, z, t, 'BEST MODEL', min_error_test[0], l_lst[0], min_errors_sets[0], name="ML_HW3_best_model",
                path2save="C:/Users/26067/PycharmProjects/ML_HW3")

visualisation = Visualization()
visualisation.models_error_scatter_plot(min_error_valid, min_error_test,
                                        get_names(func_names, min_errors_sets, l_lst), l_lst,
                                        'title', show=True, save=True,
                                        name="ML_HW3_10_models",
                                        path2save="C:/Users/26067/PycharmProjects/ML_HW3")
