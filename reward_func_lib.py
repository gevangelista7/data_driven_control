import numpy as np


def quadratic_err(state, action, ref):
    err = state[0]
    return err**2


def neg_abs_err_rwd(state, action, ref):
    err = state[0]
    return - np.abs(err)


def optim_exp_dev_rwd(state, action, ref):
    err = state[0]
    rwd = np.exp(-err**2/0.05)
    rwd -= 0.75 * action**2
    return rwd


def exp_discontinuous_rwd(state, action, ref):
    err = state[0]
    if abs(err) < .025:
        return 1
    else:
        return 0.5*np.exp(-err**2/0.5)


def exponential_deviation_rwd(state, action, ref):
    err = state[0]
    return np.exp(-err**2/0.05)


def deviation_steps_rwd(state, action, ref):
    err = state[0]
    if abs(err) < .025*ref:
        return 1
    elif abs(err) < .1*ref:
        return .75
    elif abs(err) < .25*ref:
        return .5
    else:
        return .1


def generate_const_ref_func(r):
    def const_ref(t):
        return r

    return const_ref


def multi_step_ref_func(t):
    if t < .3:
        return 5
    elif t < .6:
        return 9
    else:
        return 11



