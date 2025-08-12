# prompt_attacks_paper

## Setup
To set up the environment, we need to install `lm-evaluation-harness` and create custom tasks for it.
You can use the shell script we provide in the repo `setup.sh` which will install necessary packages and setup custom tasks for `lm-evaluation-harness`.

Custom tasks (e.g. `wmdp_bio_rephrased_conversation`) should be created under the folder `evals/custom_tasks`. You can find the general structure from [here](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/wmdp). Here's an example:
```yaml
# wmdp_bio_rephrased_conversation.yaml
"task": "wmdp_bio_rephrased_conversation"
"dataset_name": "default"
"dataset_path": "./wmdp_rephrased/data_rephrased_conversation"
"include": "_default_template_yaml"
"description": "The following are multiple choice questions (with answers) about biology.\n\n"
```

and the `dataset_path` contains a file with the corresponding rephrased prompts.

‼️ WARNING: You should create custom tasks before running the `setup.sh` script.

For more information on how to create custom tasks, please refer to [this](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/new_task_guide.md).

## Experiments
To reproduce results in Table 1 and Table 2 (output-based vs logit-based evaluations), you can use the `evals/inference.py` file, e.g.
```bash
echo "Evaluating $model on $task..."
python3 evals/inference.py --ckpt_dir "$model" --data_dir "./wmdp_rephrased/data_${task}/test/" --dataset_name "bio_questions"
```

You can use the HuggingFace checkpoints for the model (e.g. `cais/Zephyr_RMU`). 

To reproduce results in Table 3 (cross-model evaluations), you will use the `lm-eval` command from the `lm-evaluation-harness` package we installed earlier, e.g.
```bash
echo "Evaluating $model..."
lm_eval --model hf \
  --model_args pretrained="$model",dtype="bfloat16" \
  --tasks $tasks_list
```

We provide example codes in `run.sh`. 