import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# Adjusted font sizes for better readability
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12
})

# Read CSV from external file
df = pd.read_csv('models.csv')

# Adjust accuracy
df['adjusted_accuracy'] = df['accuracy'] - 0.25
df['adjusted_accuracy'] = df['adjusted_accuracy'].clip(lower=0)

# Define model families
model_families = [
    {'name': 'Zephyr', 'base': 'Zephyr_7B_Beta', 'elm': 'Zephyr-7B-ELM', 'color': '#0072B2'},
    {'name': 'Mistral', 'base': 'Mistral-7B-v0.1', 'elm': 'Mistral-7B-ELM', 'color': '#D55E00'},
    {'name': 'Llama3-8B', 'base': 'Llama3-8B', 'elm': 'Llama3-8B-ELM', 'color': '#009E73'},
    {'name': 'Llama3-8B-Instruct', 'base': 'Llama3-8B-Instruct', 'elm': 'Llama3-8B-Instruct-ELM', 'color': '#CC79A7'},
]

task_styles = {
    'wmdp_bio': {'marker': 's', 'label': 'Base prompt', 'size': 300, 'highlight': False},
    'tinyMMLU': {'marker': '*', 'label': 'tinyMMLU', 'size': 375, 'highlight': False},
    'wmdp_bio_rephrased_english_filler': {'marker': 'o', 'label': 'Filler text', 'size': 200, 'highlight': False},
    'wmdp_bio_rephrased_hindi_filler': {'marker': 'o', 'label': 'Filler text', 'size': 200, 'highlight': True},
    'wmdp_bio_rephrased_latin_filler': {'marker': 'o', 'label': 'Filler text', 'size': 200, 'highlight': False},
    'wmdp_bio_rephrased_conversation': {'marker': '^', 'label': 'Rephrased as conversation', 'size': 200, 'highlight': False},
    'wmdp_bio_rephrased_poem': {'marker': 'v', 'label': 'Rephrased as poem', 'size': 200, 'highlight': False},
    'wmdp_bio_rephrased_replace_with_variables': {'marker': 'P', 'label': 'Replaced with variables', 'size': 200, 'highlight': False},
    'wmdp_bio_rephrased_technical_terms_removed_1': {'marker': 'X', 'label': 'Technical terms removed', 'size': 200, 'highlight': False},
    'wmdp_bio_rephrased_translated_farsi': {'marker': 'd', 'label': 'Translated', 'size': 200, 'highlight': False},
    'wmdp_bio_rephrased_translated_german': {'marker': 'd', 'label': 'Translated', 'size': 200, 'highlight': False},
    'wmdp_bio_rephrased_translated_korean': {'marker': 'd', 'label': 'Translated', 'size': 200, 'highlight': False},
}
for style in task_styles.values():
    style['size'] = style['size'] / 3

# Define filled tasks
filled_tasks = {
    'wmdp_bio',
    'tinyMMLU',
    'wmdp_bio_rephrased_english_filler',
    'wmdp_bio_rephrased_hindi_filler'
}

# Create figure with gridspec: task legend row above two main plots
fig = plt.figure(figsize=(9, 5))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 3], width_ratios=[1.2, 1], hspace=0.08, wspace=0.17, right=0.99)

# Create subplots
task_legend_ax = fig.add_subplot(gs[0, :])  # Task legend spans both columns
ax1 = fig.add_subplot(gs[1, 0])  # Left plot
ax2 = fig.add_subplot(gs[1, 1], sharey=ax1)  # Right plot shares y-axis with left

# Hide the task legend axis frame
task_legend_ax.axis('off')

# ===== FIRST SUBPLOT: Model Families =====
label_pad = 2

# Diagonal reference line
ax1.plot([-0.1, 1], [-0.1, 1], 'k--', alpha=0.3)


# Plot points for first subplot
for family in model_families:
    base_data = df[df['model'] == family['base']]
    elm_data = df[df['model'] == family['elm']]
    merged = pd.merge(base_data, elm_data, on='task', suffixes=('_base', '_elm'))

    for _, row in merged.iterrows():
        task = row['task']
        if task not in task_styles:
            continue

        style = task_styles[task]
        is_filled = task in filled_tasks
        is_highlight = style.get('highlight', False)

        facecolor = family['color'] if is_filled else 'none'
        edgecolor = 'black' if is_highlight else family['color']
        linewidth = 2 if is_highlight else 1

        ax1.scatter(
            row['adjusted_accuracy_base'],
            row['adjusted_accuracy_elm'],
            facecolor=facecolor,
            edgecolor=edgecolor,
            marker=style['marker'],
            s=style.get('size', 100),
            linewidth=linewidth,
            alpha=0.9,
            zorder=5 if is_highlight else 4
        )

        # Add Hindi filler annotations
        if task == 'wmdp_bio_rephrased_hindi_filler':
            if family['name'] == 'Zephyr':
                ax1.text(
                    row['adjusted_accuracy_base'] + 0.006,
                    row['adjusted_accuracy_elm'] + 0.01,
                    'Hindi filler',
                    fontsize=11,
                    alpha=0.85
                )
            elif family['name'] == 'Llama3-8B-Instruct':
                ax1.text(
                    row['adjusted_accuracy_base'] + 0.004,
                    row['adjusted_accuracy_elm'] - 0.033,
                    'Hindi filler',
                    fontsize=11,
                    alpha=0.85
                )
            else:
                ax1.text(
                    row['adjusted_accuracy_base'] + 0.011,
                    row['adjusted_accuracy_elm'] + 0.003,
                    'Hindi filler',
                    fontsize=11,
                    alpha=0.85
                )

# Set up first subplot
ax1.set_xlabel(r'Base Model Accuracy')
ax1.set_ylabel(r'Unlearned Model Accuracy', labelpad=label_pad)
ax1.set_xlim(-0.01, 0.6)
ax1.set_ylim(-0.01, 0.5)
ax1.grid(True, linestyle='--', alpha=0.3)

# Model family legend for first subplot
model_legend = [
    Line2D([0], [0], marker='o', color=family['color'], label=family['name'],
           linestyle='', markersize=8) for family in model_families
]

ax1.legend(
    handles=model_legend,
    title='Models',
    loc='upper left',
    columnspacing=0.5,
    handletextpad=0.3,
    handlelength=1.2,
    borderaxespad=0.2,
)

# ===== SECOND SUBPLOT: Unlearning Methods =====

# Read unlearning methods CSV
df_unlearning = pd.read_csv('unlearning_method_comparison.csv')

# Define method name mapping
method_name_map = {
    'tar': 'TAR',
    'graddiff': 'GradDiff',
    'repnoise': 'RepNoise',
    'elm': 'ELM',
    'rmu-lat': 'RMU+LAT',
    'rmu': 'RMU',
    'pbj': 'PBJ',
    'rr': 'RR'
}

# Extract base model data
base_model_data = df_unlearning[df_unlearning['model'] == 'Llama3-8B-Instruct']

# Process unlearning models data
unlearning_models = []
for method_key, method_name in method_name_map.items():
    model_name = f"LLM-GAT__llama-3-8b-instruct-{method_key}-checkpoint-8"
    if model_name in df_unlearning['model'].values:
        unlearning_models.append({
            'name': method_name,
            'model': model_name,
            'color': None  # Will be assigned later
        })

# Wong's colorblind-friendly palette
colors = [
    '#882255',  # Burgundy
    '#56B4E9',  # Sky Blue
    '#E69F00',  # Orange
    '#009E73',  # Teal
    '#332288',  # Indigo
    '#AA7700',  # Dark Gold
    '#555555',  # Dark Gray
    '#CC79A7',  # Violet
]

# Assign colors to models
for i, model in enumerate(unlearning_models):
    model['color'] = colors[i % len(colors)]

# Update task styles for second plot
task_styles_unlearning = {
    'wmdp_bio': {'marker': 's', 'label': 'Base prompt', 'size': 150, 'highlight': False},
    'tinyMMLU': {'marker': '*', 'label': 'tinyMMLU', 'size': 200, 'highlight': False},
    'wmdp_bio_rephrased_english_filler': {'marker': 'o', 'label': 'Filler text', 'size': 150, 'highlight': False},
    'wmdp_bio_rephrased_hindi_filler': {'marker': 'o', 'label': 'Filler text', 'size': 150, 'highlight': False},
    'wmdp_bio_rephrased_latin_filler': {'marker': 'o', 'label': 'Filler text', 'size': 150, 'highlight': False},
    'wmdp_bio_rephrased_conversation': {'marker': '^', 'label': 'Rephrased as conversation', 'size': 150, 'highlight': False},
    'wmdp_bio_rephrased_poem': {'marker': 'v', 'label': 'Rephrased as poem', 'size': 150, 'highlight': False},
    'wmdp_bio_rephrased_replace_with_variables': {'marker': 'P', 'label': 'Replaced with variables', 'size': 150, 'highlight': False},
    'wmdp_bio_rephrased_technical_terms_removed_1': {'marker': 'X', 'label': 'Technical terms removed', 'size': 150, 'highlight': False},
    'wmdp_bio_rephrased_translated_farsi': {'marker': 'd', 'label': 'Translated', 'size': 150, 'highlight': False},
    'wmdp_bio_rephrased_translated_german': {'marker': 'd', 'label': 'Translated', 'size': 150, 'highlight': False},
    'wmdp_bio_rephrased_translated_korean': {'marker': 'd', 'label': 'Translated', 'size': 150, 'highlight': False},
}
for style in task_styles_unlearning.values():
    style['size'] = style['size'] / 3

# Define filled tasks for second plot
filled_tasks_unlearning = {
    'wmdp_bio',
    'tinyMMLU',
}

# Set up second subplot
ax2.set_xlabel('Base Model Accuracy')
ax2.set_xlim(-0.01, 0.5)
ax2.set_xticks(np.arange(0, 0.6, 0.1))  # Explicitly set x-axis ticks every 0.1
ax2.grid(True, linestyle='--', alpha=0.3)

# Add diagonal reference line
ax2.plot([-0.01, 1], [-0.01, 1], 'k--', alpha=0.3)

adjustment_factor = 0.25

# Plot points for second subplot
for model in unlearning_models:
    model_data = df_unlearning[df_unlearning['model'] == model['model']]

    for _, model_row in model_data.iterrows():
        task = model_row['task']
        if task not in task_styles_unlearning:
            continue

        # Find corresponding base model accuracy for this task
        base_row = base_model_data[base_model_data['task'] == task]
        if len(base_row) == 0:
            continue

        base_accuracy = base_row['accuracy'].values[0] - adjustment_factor
        model_accuracy = model_row['accuracy'] - adjustment_factor

        # Skip if adjusted accuracy is negative
        if base_accuracy <= 0 or model_accuracy <= 0:
            continue

        style = task_styles_unlearning[task]
        is_filled = task in filled_tasks_unlearning
        is_highlight = style.get('highlight', False)

        facecolor = model['color'] if is_filled else 'none'
        edgecolor = 'black' if is_highlight else model['color']
        linewidth = 1

        # Add scatter point
        ax2.scatter(
            base_accuracy,
            model_accuracy,
            facecolor=facecolor,
            edgecolor=edgecolor,
            marker=style['marker'],
            s=style['size'],
            linewidth=linewidth,
            alpha=0.9,
            zorder=7 if is_highlight else 6
        )

# Create model legend for second subplot
model_legend_handles = [
    Line2D([0], [0], marker='o', color=model['color'], label=model['name'],
           linestyle='', markersize=8) for model in unlearning_models
]

ax2.legend(
    handles=model_legend_handles,
    title='Unlearning Methods',
    loc='upper left',
    ncol=1,
    frameon=True,
    columnspacing=0.5,
    handletextpad=0.3,
    handlelength=1.2,
    borderaxespad=0.2,
)

# ===== SHARED TASK LEGEND =====
# Create task legend above both plots
ordered_tasks = [
    'tinyMMLU',
    'wmdp_bio',
    'wmdp_bio_rephrased_english_filler',
    'wmdp_bio_rephrased_hindi_filler',
    'wmdp_bio_rephrased_latin_filler'
]
seen_labels = set()
task_legend = []

# Helper to add a legend entry from a task key
def add_task_legend_entry(task_key, label_override=None):
    style = task_styles[task_key]
    label = label_override if label_override else style['label']
    if label in seen_labels:
        return
    seen_labels.add(label)
    is_highlight = style.get('highlight', False)
    facecolor = 'white' if is_highlight else '#999999'
    edgecolor = 'black' if is_highlight else '#999999'
    linewidth = 2.5 if is_highlight else 1.5

    task_legend.append(
        Line2D(
            [0], [0],
            marker=style['marker'],
            markerfacecolor=facecolor if not is_highlight else 'white',
            markeredgecolor=edgecolor,
            markeredgewidth=linewidth,
            linestyle='',
            markersize=10,
            label=label
        )
    )

# Add legend items in the requested order
add_task_legend_entry('tinyMMLU')  # MMLU
add_task_legend_entry('wmdp_bio')  # Base prompt
add_task_legend_entry('wmdp_bio_rephrased_hindi_filler', label_override='Knowledge retrieval')  # Highlighted
add_task_legend_entry('wmdp_bio_rephrased_english_filler')  # Filler text

# Add remaining task types not already seen
for task_key, style in task_styles.items():
    label = 'Knowledge retrieval' if task_key == 'wmdp_bio_rephrased_hindi_filler' else style['label']
    if label not in seen_labels:
        task_legend.append(
            Line2D(
                [0], [0],
                marker=style['marker'],
                markerfacecolor='none',
                markeredgecolor='#999999',
                markeredgewidth=1.5,
                linestyle='',
                markersize=10,
                label=label
            )
        )
        seen_labels.add(label)

# Add shared task legend in the top row
task_legend_ax.legend(
    handles=task_legend,
    title='Tasks',
    loc='center right',
    # bbox_to_anchor=(-0.04, 0.5),  # (x, y) - adjust x value to shift left/right
    ncol=4,
    frameon=True,
    columnspacing=0.5,
    handletextpad=0.3,
    handlelength=1.2,
    borderaxespad=0.2
)

plt.savefig('retrieval_and_methods.pdf', bbox_inches='tight', dpi=300)
plt.close()

print('Visualization saved')