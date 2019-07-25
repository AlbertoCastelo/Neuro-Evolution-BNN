import theano

from neat.evaluation import get_dataset
from tests_non_automated.probabilistic_programming.create_network import construct_nn
from tests.config_files.config_files import create_configuration
import pymc3 as pm


def main():
    config = create_configuration(filename='/siso.json')
    dataset = get_dataset(config.dataset_name, testing=False)

    # %%
    x_train = dataset.x
    y_train = dataset.y
    x = theano.shared(x_train)
    y = theano.shared(y_train)
    nn = construct_nn(x=x, y=y, config=config)

    # ADVI
    with nn:
        inference = pm.ADVI()
        approx = pm.fit(n=50000, method=inference)
    trace = approx.sample(draws=5000)

    # with nn:
    #     inference = pm.NUTS()
    #     trace = pm.sample(2000, tune=1000, cores=4, inference=inference)
    print(pm.summary(trace))

    x.set_value(x_train)
    y.set_value(y_train)

    with nn:
        ppc = pm.sample_ppc(trace, samples=500, progressbar=False)


if __name__ == '__main__':
    main()