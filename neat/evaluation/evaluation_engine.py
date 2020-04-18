from multiprocessing import cpu_count, Pool
from experiments.logger import logger
from neat.configuration import get_configuration
from neat.dataset.custom_dataloader import get_data_loader
from neat.evaluation.evaluate_parallel import evaluate_genome_task, process_initialization, \
    evaluate_genome_task_jupyneat
from neat.evaluation.evaluate_simple import evaluate_genome
from neat.evaluation.utils import get_dataset
from neat.loss.vi_loss import get_loss, get_beta
from neat.utils import timeit
IS_TESTING = False


class EvaluationStochasticEngine:
    def __init__(self, testing=False, batch_size=None):
        self.config = get_configuration()
        self.testing = testing
        self.batch_size = batch_size if batch_size is not None else self.config.batch_size
        self.parallel_evaluation = self.config.parallel_evaluation
        self.is_gpu = self.config.is_gpu

        self.dataset = None
        self.data_loader = None
        self.loss = None

        if self.parallel_evaluation:
            self.n_processes = self._get_n_processes()
            self.pool = Pool(processes=self.n_processes,
                             initializer=process_initialization,
                             initargs=(self.config.dataset, self.config.train_percentage, testing,
                                       self.config.dataset_random_state))

    def _get_n_processes(self):
        if self.config.n_processes is not None:
            return int(self.config.n_processes)
        return min(cpu_count() // 2, 8)


    @timeit
    def evaluate(self, population: dict):
        '''
        population: is a Dict{Int, Genome}
        '''
        # TODO: make n_samples increase with generation number
        n_samples = self.config.n_samples
        if self.parallel_evaluation:
            tasks = []
            for genome in population.values():
                logger.debug(f'Genome {genome.key}: {genome.get_graph()}')
                x = (genome, get_loss(problem_type=self.config.problem_type),
                     self.config.beta_type, self.config.problem_type,
                     IS_TESTING,
                     self.batch_size, n_samples, self.is_gpu)
                tasks.append(x)

            # TODO: fix logging when using multiprocessing. Easy fix is to disable
            fitnesses = list(self.pool.imap(evaluate_genome_task, tasks, chunksize=len(population)//self.n_processes))

            for i, genome in enumerate(population.values()):
                genome.fitness = fitnesses[i]

        else:
            self.dataset = self._get_dataset()
            # self.data_loader = self._get_dataloader()
            self.loss = self._get_loss()
            for genome in population.values():
                logger.debug(f'Genome {genome.key}: {genome.get_graph()}')
                genome.fitness = - evaluate_genome(genome=genome,
                                                   problem_type=self.config.problem_type,
                                                   dataset=self.dataset,
                                                   loss=self.loss,
                                                   is_testing=IS_TESTING,
                                                   beta_type=self.config.beta_type,
                                                   batch_size=self.batch_size,
                                                   n_samples=n_samples,
                                                   is_gpu=self.is_gpu)

        return population

    def close(self):
        if self.parallel_evaluation:
            print('closing pool')
            self.pool.close()
            self.pool.join()

    def _get_dataset(self):
        if self.dataset is None:
            self.dataset = get_dataset(self.config.dataset, testing=self.testing,
                                       train_percentage=self.config.train_percentage,
                                       random_state=self.config.dataset_random_state)
            # self.dataset.generate_data()
        return self.dataset

    def _get_dataloader(self):
        if self.data_loader is None:
            self.data_loader = get_data_loader(dataset=self.dataset, batch_size=self.batch_size)
        return self.data_loader

    def _get_loss(self):
        if self.loss is None:
            self.loss = get_loss(problem_type=self.config.problem_type)
        return self.loss
