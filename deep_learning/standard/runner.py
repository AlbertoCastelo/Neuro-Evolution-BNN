from deep_learning.standard.evaluate_standard_dl import EvaluateStandardDL
from neat.evaluation.utils import get_dataset
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from neat.neat_logger import logger
lr = 0.01
weight_decay = 0.0005
batch_size = 50000


class StandardDLRunner:
    def __init__(self, config, n_epochs=1000):
        self.config = config
        self.n_epochs = n_epochs
        self.evaluator = None

    def run(self):
        dataset = get_dataset(dataset=self.config.dataset, train_percentage=self.config.train_percentage,
                              random_state=self.config.dataset_random_state, noise=self.config.noise,
                              label_noise=self.config.label_noise)

        is_cuda = False

        self.evaluator = EvaluateStandardDL(dataset=dataset,
                                            batch_size=batch_size,
                                            lr=lr,
                                            weight_decay=weight_decay,
                                            n_epochs=self.n_epochs,
                                            n_neurons_per_layer=10,
                                            n_hidden_layers=1,
                                            is_cuda=is_cuda)
        self.evaluator.run()
        # self.evaluator.save_network(network_filename)

        # Show Evaluation metrics
        x, y_true, y_pred = self.evaluator.evaluate()
        x = x.numpy()
        y_true = y_true.numpy()
        y_pred = y_pred.numpy()

        # plot results
        y_pred = np.argmax(y_pred, 1)

        logger.info('Evaluate on Validation Test')
        logger.info('Confusion Matrix:')
        logger.info(confusion_matrix(y_true, y_pred))

        logger.info(f'Accuracy: {accuracy_score(y_true, y_pred)*100} %')