#!/bin/bash
clear

export AWS_S3_HOST=http://localhost:9000
export AWS_ACCESS_KEY_ID=minioKey
export AWS_SECRET_ACCESS_KEY=minioSecret
export AWS_REGION=eu-west-1
export SLACK_API_TOKEN=xoxp-803548304551-788909703698-803912405606-3537f75bda859d01dafc0bffcb200ade
export JULIA_BASE_PATH=/home/alberto/Desktop/repos/master_thesis/Neat-Julieta

###################################################################
# CLASSIFICATION-MISO
#DATASET=classification-miso
#n_input=2
#n_output=2
#POP_SIZE=50
#PARALLEL_EVALUATION=1
#n_processes=15
#IS_DISCRETE=0
#initial_nodes_sample=2
#n_species=5
#architecture_mutation_power=1
#train_percentage=0.75

###################################################################
# IRIS
#DATASET=iris
#n_input=4
#n_output=3
#POP_SIZE=50
#PARALLEL_EVALUATION=1
#n_processes=15
#IS_DISCRETE=0
#initial_nodes_sample=4
#n_species=5
#architecture_mutation_power=1
#is_initial_fully_connected=1
#train_percentage=0.75

###################################################################
# TITANIC
#DATASET=titanic
#n_input=6
#n_output=2
#POP_SIZE=50
#PARALLEL_EVALUATION=1
#n_processes=15
#IS_DISCRETE=0
#initial_nodes_sample=6
#n_species=5
#architecture_mutation_power=1
#is_initial_fully_connected=1
#train_percentage=0.75

###################################################################
# BREAST CANCER
#DATASET=breast_cancer
#n_input=30
#n_output=2
#POP_SIZE=50
#PARALLEL_EVALUATION=1
#n_processes=15
#IS_DISCRETE=0
#initial_nodes_sample=6
#n_species=5
#architecture_mutation_power=1
#is_initial_fully_connected=1
#train_percentage=0.75


###################################################################
# MNIST-DOWNSAMPLED
DATASET=mnist_downsampled
n_input=64
n_output=10
POP_SIZE=50
PARALLEL_EVALUATION=1
n_processes=12
IS_DISCRETE=0
initial_nodes_sample=32
n_species=5
architecture_mutation_power=10
train_percentage=0.75
is_initial_fully_connected=0

###################################################################
## COMMON
mutation_type="random_mutation"

is_fine_tuning=1
epochs_fine_tuning=2000

bias_mean_max_value=10.0
bias_mean_min_value=-10.0
bias_std_max_value=2.0
bias_std_min_value=0.000001

weight_mean_max_value=10.0
weight_mean_min_value=-10.0
weight_std_max_value=2.0
weight_std_min_value=0.000001

node_delete_prob=0.0
connection_delete_prob=0.0
node_add_prob=1.0
connection_add_prob=1.0

n_initial_hidden_neurons=0

mutate_power=0.5
mutate_rate=0.8
compatibility_threshold=3.0


#N_GENERATIONS=155
#generation_fix_architecture=150
noise=0.0
label_noise=0.0
N_GENERATIONS=150
generation_fix_architecture=500
POP_SIZE=50

N_REPETITIONS=1
function run_bneat {
#  echo $1 $2 $3
  correlation_id=$1
  fix_std=$2
  n_samples=$3
#  echo $correlation_id, $fix_std, $n_samples
#  for noise in 10.0 7.5 5.0 2.5 0.0
#  for noise in 1.0 0.0
#  for train_percentage in 0.05 0.1 0.15 0.2 0.3 0.4 0.5 0.7 0.9
#  for train_percentage in 0.05 0.1 0.15 0.2
#  for label_noise in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8
  for label_noise in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8

  do
#  for repetition in 1 2 3 4 5
  for repetition in $(seq 1 $N_REPETITIONS)
  do
    pipenv run python neat/run_example.py run \
        --dataset_name=$DATASET \
        --algorithm_version="bayes-neat"\
        --correlation_id=$correlation_id \
        --config_parameters="{'pop_size': $POP_SIZE,
                              'is_fine_tuning': $is_fine_tuning,
                              'epochs_fine_tuning': $epochs_fine_tuning,
                              'beta': $beta,
                              'fix_std': $fix_std,
                              'n_generations': $N_GENERATIONS,
                              'generation_fix_architecture': $generation_fix_architecture,
                              'parallel_evaluation': $PARALLEL_EVALUATION,
                              'n_processes': $n_processes,
                              'mutation_type': $mutation_type,
                              'is_discrete': $IS_DISCRETE,
                              'train_percentage': $train_percentage,
                              'bias_mean_max_value': $bias_mean_max_value,
                              'bias_mean_min_value': $bias_mean_min_value,
                              'bias_std_max_value': $bias_std_max_value,
                              'bias_std_min_value': $bias_std_min_value,

                              'weight_mean_max_value': $weight_mean_max_value,
                              'weight_mean_min_value': $weight_mean_min_value,
                              'weight_std_max_value': $weight_std_max_value,
                              'weight_std_min_value': $weight_std_min_value,

                              'node_delete_prob': $node_delete_prob,
                              'node_add_prob': $node_add_prob,
                              'connection_delete_prob': $connection_delete_prob,
                              'connection_add_prob': $connection_add_prob,
                              'mutate_power': $mutate_power,
                              'compatibility_threshold': $compatibility_threshold,
                              'n_initial_hidden_neurons': $n_initial_hidden_neurons,
                              'initial_nodes_sample': $initial_nodes_sample,
                              'n_input': $n_input,
                              'n_output': $n_output,
                              'n_samples': $n_samples,
                              'n_species': $n_species,
                              'architecture_mutation_power': $architecture_mutation_power,
                              'is_initial_fully_connected': $is_initial_fully_connected,
                              'noise': $noise,
                              'label_noise': $label_noise
        }"
  done
  done
}

# test
#correlation_id='test_6'$DATASET
#fix_std=1
#n_samples=1
#run_bneat $correlation_id $fix_std $n_samples

#experiment_name='_ft_2_'
PREFIX=''
N_EXTERNAL_REPETITIONS=5
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
#  beta=0.005
  #N_GENERATIONS=150
  run_bneat $correlation_id $fix_std $n_samples

done
