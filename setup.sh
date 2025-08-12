#!/bin/bash

set -e 

# install lm-evaluation-harness
if [ ! -d "lm-evaluation-harness" ]; then
  git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
  cd lm-evaluation-harness
  pip install -e .
  cd ..
fi

pip install protobuf sentencepiece ## dependency

# copy wmdp_rephrased tasks to lm-evaluation-harness
if [ ! -d "lm-evaluation-harness/lm_eval/tasks/wmdp_rephrased" ]; then
  echo "Copying wmdp_rephrased tasks to lm-evaluation-harness..."
  mkdir -p lm-evaluation-harness/lm_eval/tasks/wmdp_rephrased
  cp -r ./evals/custom_tasks/wmdp_lm_eval_tasks/* lm-evaluation-harness/lm_eval/tasks/wmdp_rephrased
fi