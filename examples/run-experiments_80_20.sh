#!/bin/bash
clear

export AWS_S3_HOST=http://localhost:9000
export AWS_ACCESS_KEY_ID=minioKey
export AWS_SECRET_ACCESS_KEY=minioSecret
export AWS_REGION=eu-west-1
export SLACK_API_TOKEN=xoxp-803548304551-788909703698-803912405606-3537f75bda859d01dafc0bffcb200ade
export JULIA_BASE_PATH=/home/alberto/Desktop/repos/master_thesis/Neat-Julieta

###################################################################
# Regression-Siso
#DATASET=regression-siso
#n_input=1
#n_output=1
#POP_SIZE=50
#PARALLEL_EVALUATION=1
#n_processes=15
#IS_DISCRETE=0
#initial_nodes_sample=1
#n_species=5
#architecture_mutation_power=1
#train_percentage=0.5
###################################################################
#DATASET=classification-miso
#n_input=2
#n_output=2
#n_processes=12
#initial_nodes_sample=2
#n_species=5
#architecture_mutation_power=1
##correlation_id='fix_std'
#correlation_id='free_std'
#fix_std=0
###################################################################
#DATASET=mnist_downsampled
#n_input=256
#n_output=2
#n_processes=10
#initial_nodes_sample=10
#fix_std=1
#n_species=10
#architecture_mutation_power=1
#correlation_id='test'
###################################################################
#DATASET=cancer
#n_input=1024
#n_output=2
#n_processes=10
#initial_nodes_sample=1024
#fix_std=1


#correlation_id="fix_std3"
#correlation_id="free_std"
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
#train_percentage=0.75

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

###################################################################
## MNIST DOWNSAMPLED
#DATASET=mnist_downsampled
#n_input=256
#n_output=4
#POP_SIZE=50
#PARALLEL_EVALUATION=1
#n_processes=12
#IS_DISCRETE=0
#initial_nodes_sample=5
#n_species=5
#architecture_mutation_power=2
#train_percentage=0.5


# REGRESSION MISO
#DATASET=regression-miso
#n_input=2
#n_output=1
#POP_SIZE=50
#PARALLEL_EVALUATION=1
#n_processes=12
#IS_DISCRETE=0
#initial_nodes_sample=2
#n_species=5
#architecture_mutation_power=1
#train_percentage=0.5

#correlation_id='neat_1'
#fix_std=1
#n_samples=1
#correlation_id='bayesian_neat_1'
#fix_std=0
#n_samples=50

#N_GENERATIONS_BASE=100
#POP_SIZE=70
#PARALLEL_EVALUATION=1
#n_processes=12
#train_percentage=0.5
#n_samples=10

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

##### MNIST DOWNSAMPLED
#DATASET=mnist_downsampled
#N_GENERATIONS_BASE=100
#POP_SIZE=20
#PARALLEL_EVALUATION=1
#n_processes=10
#IS_DISCRETE=1
#node_delete_prob=0.0
#connection_delete_prob=0.0
#node_add_prob=0.5
#connection_add_prob=0.8
#n_output=10
#n_initial_hidden_neurons=0
#initial_nodes_sample=50
#
#mutate_power=0.5
#mutate_rate=0.8
#compatibility_threshold=2.5
#
#fix_std=false
#bias_std_max_value=0.00000001
#weight_std_max_value=0.00000001
#N_GENERATIONS=300

#for n_output in 4 6 8 10
#do
#  for node_add_prob in 0.5
#  do
#    for connection_add_prob in 0.6
#    do
#for train_percentage in 0.4 0.6 0.8
#for compatibility_threshold in 2.0 2.5 3.0
# for train_percentage in 0.2 0.4 0.6 0.8
#do
#  N_GENERATIONS=$((n_output * 100))
#  echo $N_GENERATIONS

N_REPETITIONS=20
function run_bneat {
#  echo $1 $2 $3
  correlation_id=$1
  fix_std=$2
  n_samples=$3
#  echo $correlation_id, $fix_std, $n_samples
#  for train_percentage in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
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
                              'architecture_mutation_power': $architecture_mutation_power
        }"
  done
}

# test
#correlation_id='test_6'$DATASET
#fix_std=1
#n_samples=1
#run_bneat $correlation_id $fix_std $n_samples

# RUN Neat
correlation_id='neat_3_'$DATASET
fix_std=1
n_samples=1
run_bneat $correlation_id $fix_std $n_samples

# RUN Bayesian-Neat
correlation_id='bayesian_neat_3_'$DATASET
fix_std=0
n_samples=50
run_bneat $correlation_id $fix_std $n_samples

