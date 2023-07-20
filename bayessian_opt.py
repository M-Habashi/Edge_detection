from bayes_opt import BayesianOptimization
from bayessian_opt_utils import *


def my_function(loss_wt_1, loss_wt_2, gamma, lr):
    f = train_dynamic(loss_wt_1, loss_wt_2, gamma, lr)
    return f  # the library maximizes the obj function while we want to minimize


# Define the search space
pbounds = {'loss_wt_1': (1, 3),
           'loss_wt_2': (2, 10),
           'gamma': (1, 4),
           'lr': (0.0001, 0.01)
           }

# Create an instance of the optimizer
optimizer = BayesianOptimization(
    f=my_function, pbounds=pbounds,
)

# Perform the optimization
optimizer.maximize(init_points=5, n_iter=100)

# Get the optimized parameters
best_params = optimizer.max['params']
best_loss_wt_1 = best_params['loss_wt_1']
best_loss_wt_2 = best_params['loss_wt_2']
best_gamma = best_params['gamma']
best_lr = best_params['lr']

# Get the minimum value
min_value = optimizer.max['target']

print("Optimized Parameters:")
print(f"loss_wt_1: {best_loss_wt_1}, loss_wt_2: {best_loss_wt_2}, gamma: {best_gamma}, lr: {best_lr}")
print("Minimum Value:", min_value)

print("debug")
