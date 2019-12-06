from multiprocessing import Manager
import os
import torch

from experiments.multiprocessing_utils import Worker
from neat.evaluation.evaluation_engine import Task
from neat.genome import Genome
from neat.loss.vi_loss import get_loss
from neat.neat_logger import get_neat_logger
from tests.config_files.config_files import create_configuration

config = create_configuration(filename='/mnist_binary.json')
LOGS_PATH = f'{os.getcwd()}/'
logger = get_neat_logger(path=LOGS_PATH)

# N_SAMPLES = 10
N_PROCESSES = 16
N_GENOMES = 5

genomes = []
for i in range(N_GENOMES):
    genome = Genome(key=i)
    genome.create_random_genome()
    genomes.append(genome)


manager = Manager()
task_queue = manager.Queue()
exit_queue = manager.Queue()
exception_queue = manager.Queue()
results_queue = manager.Queue()


workers = []
for i in range(N_PROCESSES):
    worker = Worker(task_queue=task_queue,
                    exit_queue=exit_queue,
                    exception_queue=exception_queue,
                    results_queue=results_queue)
    worker.start()
    workers.append(worker)

for genome in genomes:
    task_queue.put(Task(genome=genome.copy(), dataset=None,
                        x=torch.zeros((2, 784)).float(),
                        y=torch.zeros(2).long(),
                        # x=self.dataset.x.clone().detach(),
                        # y=self.dataset.y.clone().detach(),
                        loss=get_loss('classification'),
                        beta_type='other',
                        problem_type='classification',
                        batch_size=100000,
                        n_samples=20,
                        is_gpu=False))

while not task_queue.empty():
    print('reading results from workers')
    print(task_queue.qsize())
    # for worker in workers:
    #     print(f'Is alive: {worker.is_alive()}')
    # print('reading results from workers')
    # print(task_queue.qsize())
    if not exception_queue.empty():
        exception = exception_queue.get()
        raise exception
    results = results_queue.get()
    print(results)

print('finished')
