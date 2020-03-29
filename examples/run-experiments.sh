#!/bin/bash
clear

export AWS_S3_HOST=http://localhost:9000
export AWS_ACCESS_KEY_ID=minioKey
export AWS_SECRET_ACCESS_KEY=minioSecret
export AWS_REGION=eu-west-1
export SLACK_API_TOKEN=xoxp-803548304551-788909703698-803912405606-3537f75bda859d01dafc0bffcb200ade
export JULIA_BASE_PATH=/home/alberto/Desktop/repos/master_thesis/Neat-Julieta


#DATASET=classification-miso
#DATASET=mnist_downsampled
#n_input=256
#n_output=4

correlation_id="fix_std"

DATASET=titanic
n_input=6
n_output=2

N_GENERATIONS_BASE=100
POP_SIZE=50
PARALLEL_EVALUATION=1
n_processes=12
train_percentage=0.9
IS_DISCRETE=0
n_samples=1

node_delete_prob=0.0
connection_delete_prob=0.0
node_add_prob=1.0
connection_add_prob=1.0

n_initial_hidden_neurons=0
initial_nodes_sample=256

mutate_power=0.5
mutate_rate=0.8
compatibility_threshold=2.5

fix_std=false
bias_std_max_value=0.00000001
weight_std_max_value=0.00000001
N_GENERATIONS=200

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
for train_percentage in 0.2 0.4 0.6 0.8
do
#  N_GENERATIONS=$((n_output * 100))
#  echo $N_GENERATIONS
  for repetition in 1 2 3
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
                              'n_samples': $n_samples
        }"
  done
done

#done
