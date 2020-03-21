#!/bin/bash
export AWS_S3_HOST=http://localhost:9000
export AWS_ACCESS_KEY_ID=minioKey
export AWS_SECRET_ACCESS_KEY=minioSecret
export AWS_REGION=eu-west-1
export SLACK_API_TOKEN=
export JULIA_BASE_PATH=/home/alberto/Desktop/repos/master_thesis/Neat-Julieta


#DATASET=classification-miso
DATASET=mnist_binary
DATASET=mnist_downsampled


N_GENERATIONS=200
POP_SIZE=150
PARALLEL_EVALUATION=1
IS_DISCRETE=1
node_add_prob=0.5
connection_add_prob=0.4


#for node_add_prob in 0.4 0.5 0.7
#do
#  for connection_add_prob in 0.4 0.5 0.7
#  do
for n_output in 4 6 8 10
do
  for repetition in 1 2
  do
	  pipenv run python neat/run_example.py run \
	      --dataset_name=$DATASET \
	      --algorithm_version="bayes-neat"\
        --correlation_id="test" \
        --config_parameters="{'pop_size': $POP_SIZE,
                              'n_generations': $N_GENERATIONS,
                              'parallel_evaluation': $PARALLEL_EVALUATION,
                              'is_discrete': $IS_DISCRETE,
                              'node_add_prob': $node_add_prob,
                              'connection_add_prob': $connection_add_prob,
                              'n_output': $n_output
        }"
  done
done
