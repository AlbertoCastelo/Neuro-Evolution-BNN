#!/bin/bash
clear

export AWS_S3_HOST=http://localhost:9000
export AWS_ACCESS_KEY_ID=minioKey
export AWS_SECRET_ACCESS_KEY=minioSecret
export AWS_REGION=eu-west-1
export SLACK_API_TOKEN=xoxp-803548304551-788909703698-803912405606-3537f75bda859d01dafc0bffcb200ade
export JULIA_BASE_PATH=/home/alberto/Desktop/repos/master_thesis/Neat-Julieta

###################################################################
# IRIS
DATASET=iris
n_input=4
n_output=3
POP_SIZE=50
PARALLEL_EVALUATION=1
n_processes=15
IS_DISCRETE=0
initial_nodes_sample=4
n_species=5
architecture_mutation_power=1
train_percentage=0.75


node_delete_prob=0.0
connection_delete_prob=0.0
node_add_prob=1.0
connection_add_prob=1.0

n_initial_hidden_neurons=0
#initial_nodes_sample=256

mutate_power=0.5
mutate_rate=0.8
compatibility_threshold=3.0

#fix_std=1
bias_std_max_value=0.00000001
weight_std_max_value=0.00000001
N_GENERATIONS=300

N_REPETITIONS=20
function run_bneat {
#  echo $1 $2 $3
  correlation_id=$1
  fix_std=$2
  n_samples=$3
#  echo $correlation_id, $fix_std, $n_samples
  for noise in 0.0 0.5 1.0 1.5 2.0
  do
#  for repetition in 1 2 3 4 5
  for repetition in $(seq 1 $N_REPETITIONS)
  do
    pipenv run python neat/run_example.py run \
        --dataset_name=$DATASET \
        --algorithm_version="bayes-neat"\
        --correlation_id=$correlation_id \
        --config_parameters="{'pop_size': $POP_SIZE,
                              'fix_std': $fix_std,
                              'n_generations': $N_GENERATIONS,
                              'parallel_evaluation': $PARALLEL_EVALUATION,
                              'n_processes': $n_processes,
                              'is_discrete': $IS_DISCRETE,
                              'train_percentage': $train_percentage,
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
                              'noise': $noise
        }"
  done
  done
}

# test
#correlation_id='test_6'$DATASET
#fix_std=1
#n_samples=1
#run_bneat $correlation_id $fix_std $n_samples

# RUN Neat
correlation_id='neat_5_'$DATASET
fix_std=1
n_samples=1
run_bneat $correlation_id $fix_std $n_samples

# RUN Bayesian-Neat
correlation_id='bayesian_neat_5_'$DATASET
fix_std=0
n_samples=50
run_bneat $correlation_id $fix_std $n_samples

