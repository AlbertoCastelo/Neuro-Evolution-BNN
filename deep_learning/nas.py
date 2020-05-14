import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from deep_learning.probabilistic.evaluate_probabilistic_dl import EvaluateProbabilisticDL
from deep_learning.report import BackpropReport
from neat.evaluation.utils import get_dataset

ALGORITHM_VERSION = 'nas'


def neural_architecture_search(n_hidden_layers_values, n_neurons_per_layer_values, correlation_id,
                               config, batch_size, lr, weight_decay, n_epochs, notifier, report_repository,
                               n_repetitions=1, is_cuda=False, EvaluateDL=EvaluateProbabilisticDL):
    # TODO: this search could be paralellized

    best_loss = 10000
    best_network = None
    dataset = get_dataset(dataset=config.dataset,
                          train_percentage=config.train_percentage,
                          random_state=config.dataset_random_state,
                          noise=config.noise,
                          label_noise=config.label_noise)

    backprop_report = BackpropReport.create(report_repository=report_repository,
                                            algorithm_version=ALGORITHM_VERSION,
                                            dataset=config.dataset,
                                            correlation_id=correlation_id)
    print(f'algorithm_version: {ALGORITHM_VERSION}\n'
          f'correlation_id: {correlation_id}\n'
          f'execution_id: {backprop_report.report.execution_id}')

    for n_hidden_layers in n_hidden_layers_values:
        for n_neurons_per_layer in n_neurons_per_layer_values:
            # TODO: MAYBE critiria for best validation network is accuracy instead of loss
            evaluator = EvaluateDL(dataset=dataset,
                                   batch_size=batch_size,
                                   n_samples=config.n_samples,
                                   lr=lr,
                                   weight_decay=weight_decay,
                                   n_epochs=n_epochs,
                                   n_neurons_per_layer=n_neurons_per_layer,
                                   n_hidden_layers=n_hidden_layers,
                                   is_cuda=is_cuda,
                                   beta=config.beta,
                                   n_repetitions=n_repetitions)
            evaluator.run()

            if evaluator.best_loss_val < best_loss:
                best_loss = evaluator.best_loss_val
                params = {'n_hidden_layers': n_hidden_layers,
                          'n_neurons_per_layer': n_neurons_per_layer}
                best_network = evaluator.network
    evaluator = EvaluateDL(dataset=dataset,
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
    evaluator.best_network = best_network
    _, y_true, y_pred = evaluator.evaluate()
    y_true = y_true.numpy()
    y_pred = torch.argmax(y_pred, dim=1).numpy()
    accuracy = accuracy_score(y_true, y_pred) * 100
    f1 = f1_score(y_true, y_pred, average='weighted')
    confusion_m = confusion_matrix(y_true, y_pred)

    backprop_report.report_best_network(best_network, params, accuracy, f1)
    backprop_report.set_config(config)
    backprop_report.persist_report()
    # backprop_report.persist_logs()

    print('Confusion Matrix:')
    print(confusion_m)
    print(f'Accuracy: {accuracy} %')
    notifier.send(f'NEURAL ARCHITECTURE SEARCH: {EvaluateDL.__name__} \n'
                  f'Dataset: {config.dataset}. \n'
                  f'Noise: {config.noise}. \n'
                  f'Label Noise: {config.label_noise}. \n'
                  f'Best lost found: {best_loss} with params: {str(params)}. \n'
                  f'Accuracy: {accuracy} %. \n'
                  f'F1: {f1}. \n'
                  f'Confusion-matrix: {confusion_m}')
