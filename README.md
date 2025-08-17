# Prompt Attacks Reveal Superficial Knowledge Removal in Unlearning Methods

This repository contains the code and data for our paper "Prompt Attacks Reveal Superficial Knowledge Removal in Unlearning Methods", accepted at the SoLaR (Society, Language, and Reasoning) workshop at COLM 2025.

## Abstract
In this work, we demonstrate that certain machine unlearning methods may fail
under straightforward prompt attacks. We systematically evaluate eight unlearning
techniques across three model families using output-based, logit-based, and probe
analysis to assess the extent to which supposedly unlearned knowledge can be
retrieved. While methods like RMU and TAR exhibit robust unlearning, ELM
remains vulnerable to specific prompt attacks (e.g., prepending Hindi filler text
to the original prompt recovers 57.3% accuracy). Our logit analysis further indi-
cates that unlearned models are unlikely to hide knowledge through changes in
answer formatting, given the strong correlation between output and logit accuracy.
These findings challenge prevailing assumptions about unlearning effectiveness
and highlight the need for evaluation frameworks that can reliably distinguish be-
tween genuine knowledge removal and superficial output suppression. To facilitate
further research, we publicly release our evaluation framework to easily evaluate
prompting techniques to retrieve unlearned knowledge.

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

### Probes

To reproduce figures for the probes, you will need to do the following:

1. Get access to the WMDP Bio forget corpus: [form to request access to the dataset](https://huggingface.co/datasets/cais/wmdp-bio-forget-corpus)
2. In `probe/train_probes_on_unlearned_models.ipynb` update the filepath to the WMDP bio dataset that you want to probe with along with the model that you want to probe. (This should be on the third cell)
3. Run the rest of the script and produce the output CSV file
4. Run `plots/probe_visualization.py` on the CSV output to produce the graph.