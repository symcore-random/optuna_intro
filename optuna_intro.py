# %% initialization
from sklearn.metrics import mean_squared_error
from functools import partial
import numpy as np
import optuna

###############################################################
###############################################################
### In this exercise, we will optimize the linear function. ###
###############################################################
###############################################################


##########################################################
### defining the optimization function for fine tuning ###
##########################################################

# This example function mimics a model of e.g. deep learning,
# or catboost. The parameters are like the "learning_rate"
# (for deep learning or catboost) or "max_depth" (for catboost).
# As input for the model, we have an array 'x'.
def model(x, parameters):
    a = parameters["a"]
    b = parameters["b"]
    result = a * x + b

    return result


def optimize(
    trial,
    input_data,
    true_output_data,
    parameters: dict = {"a": 2, "b": 1},
    tune_parameters: dict = {},
):
    for tune_parameter, [range_type, start, end] in tune_parameters.items():
        if range_type == "uniform":
            parameter_trial = trial.suggest_uniform(tune_parameter, start, end)
        elif range_type == "int":
            parameter_trial = trial.suggest_int(tune_parameter, start, end)
        parameters[tune_parameter] = parameter_trial

    # model prediction, given input data and the model parameters
    prediction = model(input_data, parameters)
    # calculate the mean squared error between the prediction and
    # the true output data
    mse = mean_squared_error(true_output_data, prediction)

    return mse


#%%
#########################################
### running the optimization function ###
#########################################

# define the parameters which are to be fine tuned, their range
# types (in this case 'uniform') and their start and end range values.
# For more detail, see inside the 'optimize' function.
tune_parameters = {"b": ["uniform", 0.5, 4]}
# input data
X = np.array([1, 2, 3, 4, 5, 6])
# true output data
y = np.array([3.1, 4.7, 7.2, 9.02, 11.1, 13.8])

# we define a 'partial' function here, so that optuna concentrates
# only on the 'trial' parameter.
optimization_function = partial(
    optimize,
    input_data=X,
    true_output_data=y,
    tune_parameters=tune_parameters,
)
# create study which will minimize (depending on the returned value
# by 'optimize', you will have to define direction="maximize"). In our
# case, the study will minimize the function 'mean_squared_error'.
study = optuna.create_study(direction="minimize")
# optimization in practice (we define 100 trials)
study.optimize(optimization_function, n_trials=100)


# %%
############################
### show best parameters ###
############################

# obtain the parameters for the best trial
trial = study.best_trial
print(trial.value, trial.params)
# visualize the trial in a figure
optuna.visualization.plot_optimization_history(study)


######################################################
### I believe there is much more to optuna than    ###
### what we have explored in this simple exercise, ###
### however this is sufficient for starters.       ###
######################################################
