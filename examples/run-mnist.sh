#!/bin/bash

export AWS_S3_HOST=http://localhost:9000
export AWS_ACCESS_KEY_ID=minioKey
export AWS_SECRET_ACCESS_KEY=minioSecret
export AWS_REGION=eu-west-1
export SLACK_API_TOKEN=xoxp-803548304551-788909703698-803912405606-3537f75bda859d01dafc0bffcb200ade
export JULIA_BASE_PATH=/home/alberto/Desktop/repos/master_thesis/Neat-Julieta


###################################################################
# MNIST-DOWNSAMPLED
DATASET=mnist_downsampled
noise=0.0
label_noise=0.0
initial_nodes_sample=32
architecture_mutation_power=10
is_initial_fully_connected=0

POP_SIZE=50
N_GENERATIONS=150

N_REPETITIONS=1
function run_bneat {
#  echo $1 $2 $3
  correlation_id=$1
  fix_std=$2
  n_samples=$3

#  do
#  for repetition in 1 2 3 4 5
  for repetition in $(seq 1 $N_REPETITIONS)
  do
    pipenv run python neat/run_example.py run \
        --dataset_name=$DATASET \
        --algorithm_version="bayes-neat"\
        --correlation_id=$correlation_id \
        --config_parameters="{'pop_size': $POP_SIZE,
                              'beta': $beta,
                              'fix_std': $fix_std,
                              'n_generations': $N_GENERATIONS,
                              'initial_nodes_sample': $initial_nodes_sample,
                              'architecture_mutation_power': $architecture_mutation_power,
                              'is_initial_fully_connected': $is_initial_fully_connected,
                              'noise': $noise,
                              'label_noise': $label_noise
        }"
  done
#  done
}

# test
#correlation_id='test_6'$DATASET
#fix_std=1
#n_samples=1
#run_bneat $correlation_id $fix_std $n_samples

#experiment_name='_ft_2_'
PREFIX='testing'
N_EXTERNAL_REPETITIONS=1
for rep in $(seq 1 $N_EXTERNAL_REPETITIONS)
  do
  # RUN Neat
#  correlation_id='neat_ft_22_'$PREFIX'_'$DATASET
#  fix_std=1
#  n_samples=1
#  beta=0.0
#  #N_GENERATIONS=100
#  run_bneat $correlation_id $fix_std $n_samples

  # RUN Bayesian-Neat
  correlation_id='bayesian_neat_ft_22_'$PREFIX'_'$DATASET
  fix_std=0
  n_samples=50
  beta=0.0001
  run_bneat $correlation_id $fix_std $n_samples

done
