import threading
import time

import jsons
import os
from experiments.file_utils import write_json_file_from_dict, read_json_file_to_dict
from experiments.logger import logger
from experiments.slack_client import Notifier
from neat.configuration import get_configuration
from neat.evaluation.evaluation_engine_jupyneat import EvaluationStochasticEngineJupyneat
from neat.genome import Genome
from neat.reports import EvolutionReport
from neat.evaluation.evaluation_engine import EvaluationStochasticEngine
from neat.utils import timeit

JULIA_BASE_PATH = os.getenv("JULIA_BASE_PATH")


class JupyNeatFSEvaluationEngine:
    @staticmethod
    def create(report: EvolutionReport, notifier: Notifier):
        return JupyNeatFSEvaluationEngine(report, notifier, EvaluationStochasticEngineJupyneat())

    def __init__(self, report: EvolutionReport, notifier: Notifier, evaluation_engine: EvaluationStochasticEngine):
        self.report = report
        self.notifier = notifier
        self.evaluation_engine = evaluation_engine
        self.configuration = get_configuration()

    @timeit
    def run(self):
        end_condition = 'normal'
        logger.info('Started evolutionary process')
        # try:
        '''launch Julia Evolutionary service'''
        self._launch_evolutionary_service()

        '''launch Python Evaluation service'''
        for generation in range(0, self.configuration.n_generations + 1):
            self._run_generation(generation)
        self.evaluation_engine.close()
        # except Exception as e:
        #     end_condition = 'exception'
        #     logger.exception(str(e))
        #     self.notifier.send(str(e))
        # finally:
        #     self.report.generate_final_report(end_condition=end_condition) \
        #         .persist_report()
        #     self.report.persist_logs()
        #     self.notifier.send(str(self.report.get_best_individual()))
        logger.info('Finished evolutionary process')

    @timeit
    def _run_generation(self, generation):
        logger.info(f'Genaration {generation}')
        # read
        # population = self._read_population(generation=generation)
        population = self._read_population_dict(generation=generation)

        population = self.evaluation_engine.evaluate(population=population)
        self._write_fitness(population=population, generation=generation)
        # report
        self.report.report_new_generation(generation=generation,
                                          population=population,
                                          species=None)

    def _launch_evolutionary_service(self):
        self._save_configuration()
        # run julia script
        self._start_julia_service()

    def _save_configuration(self):
        file_dir = self._get_configuration_directory()
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        config_dict = jsons.dump(self.configuration)
        write_json_file_from_dict(data=config_dict, filename=f'{file_dir}/configuration.json')

    @timeit
    def _read_population(self, generation):
        file_dir = self._get_configuration_directory()
        filename = f'{file_dir}/generation_{generation}_population.json'
        _wait_for_file_to_be_available(filename=filename, timeout=60)
        genomes = read_json_file_to_dict(filename=filename)
        population = {}
        for genome_dict in genomes:
            genome = Genome.create_from_julia_dict(genome_dict)
            population[genome.key] = genome
        return population

    @timeit
    def _read_population_dict(self, generation):
        file_dir = self._get_configuration_directory()
        filename = f'{file_dir}/generation_{generation}_population.json'
        _wait_for_file_to_be_available(filename=filename, timeout=60)
        genomes = read_json_file_to_dict(filename=filename)
        population = {}
        for genome_dict in genomes:
            genome_key = genome_dict['key']
            population[genome_key] = genome_dict
        return population

    def _get_execution_path(self):
        """dataset=$(configuration.dataset)/experiment=$(configuration.experiment)/
        execution=$(configuration.execution_id)
        """

        return f'dataset={self.configuration.dataset}/experiment={self.configuration.experiment}/' \
            f'execution={self.configuration.execution}'

    def _start_julia_service(self):
        logger.info("Launching Julia Thread")
        x = threading.Thread(target=launch_julia_neat, args=(self._get_configuration_directory(),),
                             daemon=True)
        x.start()

    def _get_configuration_directory(self):
        return f'./{self._get_execution_path()}'

    @timeit
    def _write_fitness(self, population, generation):
        file_dir = self._get_configuration_directory()
        filename = f'{file_dir}/generation_{generation}_fitness.json'

        # fitnesses = [genome.fitness for genome in population.values()]
        fitnesses = [genome['fitness'] for genome in population.values()]

        fitness_dict = dict(zip(list(population.keys()), fitnesses))

        write_json_file_from_dict(data=fitness_dict, filename=filename)
        write_json_file_from_dict(data={}, filename=f'{filename}_signal')

@timeit
def _wait_for_file_to_be_available(filename, timeout=60):
    filename_signal = f'{filename}_signal'
    start_time = time.time()
    population_not_available = True
    while population_not_available:
        if os.path.exists(filename_signal):
            return True
        elapsed_time = round(time.time() - start_time, 3)
        if elapsed_time > timeout:
            raise TimeoutError(f'Elapsed time waiting for fiel {filename_signal}')


def launch_julia_neat(path):
    cmd = f'''julia {JULIA_BASE_PATH}/entrypoint.jl "{path}/"'''
    print(cmd)
    os.system(cmd)
