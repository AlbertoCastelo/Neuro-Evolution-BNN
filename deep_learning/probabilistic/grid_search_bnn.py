import torch
import os
from config_files.configuration_utils import create_configuration
from deep_learning.probabilistic.evaluate_prababilistic_dl import EvaluateProbabilisticDL
from experiments.reporting.report_repository import ReportRepository
from experiments.slack_client import SlackNotifier
from neat.evaluation.utils import get_dataset
from neat.neat_logger import get_neat_logger
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score


LOGS_PATH = f'{os.getcwd()}/'
logger = get_neat_logger(path=LOGS_PATH)

is_cuda = False
lr = 0.01
weight_decay = 0.0005
n_epochs = 1000
batch_size = 50000

report_repository = ReportRepository.create(project='neuro-evolution', logs_path=LOGS_PATH)
notifier = SlackNotifier.create(channel='batch-jobs')

dataset_name = 'iris'
config = create_configuration(filename=f'/{dataset_name}.json')
config.noise = 0.0
config.train_percentage = 0.75
config.n_samples = 100
# network_filename = f'network-probabilistic-classification.pt'

if is_cuda:
    use_cuda = torch.cuda.is_available()
    torch.cuda.set_device(0)

n_hidden_layers_values = [1, 2, 3]
n_neurons_per_layer_values = list(range(2, 21))

# n_hidden_layers_values = [1]
# n_neurons_per_layer_values = list(range(2, 3))
# n_epochs = 1

noises = [0.0, 0.5, 1.0, 2.0, 5.0]

for noise in noises:
    best_loss = 10000
    best_network_state = None
    config.noise = noise
    dataset = get_dataset(dataset=config.dataset,
                          train_percentage=config.train_percentage,
                          random_state=config.dataset_random_state,
                          noise=config.noise)

    for n_hidden_layers in n_hidden_layers_values:
        for n_neurons_per_layer in n_neurons_per_layer_values:
            # TODO: MAYBE critiria for best validation network is accuracy instead of loss
            evaluator = EvaluateProbabilisticDL(dataset=dataset,
                                                batch_size=batch_size,
                                                n_samples=config.n_samples,
                                                lr=lr,
                                                weight_decay=weight_decay,
                                                n_epochs=n_epochs,
                                                n_neurons_per_layer=n_neurons_per_layer,
                                                n_hidden_layers=n_hidden_layers,
                                                is_cuda=is_cuda,
                                                beta=config.beta)
            evaluator.run()

            if evaluator.best_loss_val < best_loss:
                best_loss = evaluator.best_loss_val
                params = {'n_hidden_layers': n_hidden_layers,
                          'n_neurons_per_layer': n_neurons_per_layer}
                best_network_state = evaluator.network.state_dict()

    evaluator = EvaluateProbabilisticDL(dataset=dataset,
                                        batch_size=batch_size,
                                        n_samples=config.n_samples,
                                        lr=lr,
                                        weight_decay=weight_decay,
                                        n_epochs=n_epochs,
                                        n_neurons_per_layer=params['n_neurons_per_layer'],
                                        n_hidden_layers=params['n_hidden_layers'],
                                        is_cuda=is_cuda,
                                        beta=config.beta)
    evaluator._initialize()
    evaluator.best_network_state = best_network_state
    _, y_true, y_pred = evaluator.evaluate()
    y_true = y_true.numpy()
    y_pred = torch.argmax(y_pred, dim=1).numpy()

    accuracy = accuracy_score(y_true, y_pred)*100
    f1 = f1_score(y_true, y_pred, average='weighted')
    confusion_m = confusion_matrix(y_true, y_pred)
    print('Confusion Matrix:')
    print(confusion_m)

    print(f'Accuracy: {accuracy} %')

    notifier.send(f'NEURAL ARCHITECTURE SEARCH. \n'
                  f'Noise: {config.noise}. \n'
                  f'Best lost found: {best_loss} with params: {str(params)}. \n'
                  f'Accuracy: {accuracy} %. \n'
                  f'F1: {f1}. \n'
                  f'Confusion-matrix: {confusion_matrix}')
