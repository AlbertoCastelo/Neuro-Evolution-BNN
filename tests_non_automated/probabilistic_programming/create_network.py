import theano
import numpy as np
import pymc3 as pm


def construct_nn(x, y, config):
    '''
    Follows Twiecki post: https://twiecki.io/blog/2016/06/01/bayesian-deep-learning/
    '''
    n_hidden = 3

    # Initialize random weights between each layer
    w_1_init = np.random.randn(config.n_input, n_hidden).astype(theano.config.floatX)
    b_1_init = np.random.randn(n_hidden).astype(theano.config.floatX)

    w_2_init = np.random.randn(n_hidden, config.n_output).astype(theano.config.floatX)
    b_2_init = np.random.randn(config.n_output).astype(theano.config.floatX)
    # init_out = np.random.randn(n_hidden).astype(floatX)

    with pm.Model() as neural_network:
        # Weights from input to hidden layer
        weights_in_1 = pm.Normal('w_in_1', 0, sd=1,
                                 shape=(config.n_input, n_hidden),
                                 testval=w_1_init)

        # Bias from input to hidden layer
        bias_1 = pm.Normal('b_in_1', 0, sd=1,
                              shape=n_hidden,
                              testval=b_1_init)

        # Weights from 1st to 2nd layer
        weights_1_2 = pm.Normal('w_1_2', 0, sd=1,
                                shape=(n_hidden, config.n_output),
                                testval=w_2_init)

        # Bias from 1st to 2nd layer
        bias_2 = pm.Normal('b_in_2', 0, sd=1,
                              shape=config.n_output,
                              testval=b_2_init)

        # # Weights from hidden layer to output
        # weights_2_out = pm.Normal('w_2_out', 0, sd=1,
        #                           shape=(n_hidden,),
        #                           testval=init_out)

        # Build neural-network using tanh activation function
        x_1 = pm.math.dot(x, weights_in_1) + bias_1
        act_1 = pm.math.tanh(x_1)

        x_2 = pm.math.dot(act_1, weights_1_2) + bias_2
        act_2 = pm.math.tanh(x_2)

        # Regression -> Normal likelihood
        out = pm.Normal('out', act_2, observed=y,
                        total_size=x.shape[0]  # IMPORTANT for minibatches
                        )
    return neural_network


# Trick: Turn inputs and outputs into shared variables.
# It's still the same thing, but we can later change the values of the shared variable
# (to switch in the test-data later) and pymc3 will just use the new data.
# Kind-of like a pointer we can redirect.
# For more info, see: http://deeplearning.net/software/theano/library/compile/shared.html


def construct_2_hidden_nn(x, y, config):
    '''
    Follows Twiecki post: https://twiecki.io/blog/2016/06/01/bayesian-deep-learning/
    '''
    n_hidden = 3

    # Initialize random weights between each layer
    w_1_init = np.random.randn(config.n_input, n_hidden).astype(theano.config.floatX)
    b_1_init = np.random.randn(n_hidden).astype(theano.config.floatX)

    w_2_init = np.random.randn(n_hidden, n_hidden).astype(theano.config.floatX)
    b_2_init = np.random.randn(n_hidden).astype(theano.config.floatX)

    w_3_init = np.random.randn(n_hidden, config.n_output).astype(theano.config.floatX)
    b_3_init = np.random.randn(config.n_output).astype(theano.config.floatX)

    with pm.Model() as neural_network:
        # Weights from input to hidden layer
        weights_in_1 = pm.Normal('w_in_1', 0, sd=1,
                                 shape=(config.n_input, n_hidden),
                                 testval=w_1_init)

        # Bias from input to hidden layer
        bias_1 = pm.Normal('b_in_1', 0, sd=1,
                              shape=n_hidden,
                              testval=b_1_init)

        # Weights from 1st to 2nd layer
        weights_1_2 = pm.Normal('w_1_2', 0, sd=1,
                                shape=(n_hidden, n_hidden),
                                testval=w_2_init)

        # Bias from 1st to 2nd layer
        bias_2 = pm.Normal('b_in_2', 0, sd=1,
                              shape=n_hidden,
                              testval=b_2_init)

        weights_2_3 = pm.Normal('w_2_3', 0, sd=1,
                                shape=(n_hidden, config.n_output),
                                testval=w_3_init)

        # Bias from 1st to 2nd layer
        bias_3 = pm.Normal('b_3', 0, sd=1,
                           shape=config.n_output,
                           testval=b_3_init)

        # # Weights from hidden layer to output
        # weights_2_out = pm.Normal('w_2_out', 0, sd=1,
        #                           shape=(n_hidden,),
        #                           testval=init_out)

        # Build neural-network using tanh activation function
        x_1 = pm.math.dot(x, weights_in_1) + bias_1
        act_1 = pm.math.tanh(x_1)

        x_2 = pm.math.dot(act_1, weights_1_2) + bias_2
        act_2 = pm.math.tanh(x_2)

        x_3 = pm.math.dot(act_2, weights_2_3) + bias_3
        act_3 = pm.math.tanh(x_3)

        # Regression -> Normal likelihood
        out = pm.Normal('out', act_3, observed=y,
                        total_size=x.shape[0]  # IMPORTANT for minibatches
                        )
    return neural_network
