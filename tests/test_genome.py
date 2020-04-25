import tempfile
from unittest import TestCase
import os

from config_files.configuration_utils import create_configuration
from neat.genome import Genome


class TestGenome(TestCase):
    def setUp(self) -> None:
        self.config = create_configuration('/classification-miso.json')
        self.path = tempfile.mkdtemp()

    def test_write_and_read_geneme(self):
        filename = self.path + '/genome_test.json'
        genome = Genome(key=0)
        genome.create_random_genome()

        genome.save_genome(filename)

        genome_read = Genome.create_from_file(filename)

        self.assertEqual(len(genome.__dict__), len(genome_read.__dict__))
