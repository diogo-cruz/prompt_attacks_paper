"""
Probe Accuracy Visualization Tool

This module provides functionality to visualize probe accuracy results across different
unlearning methods and model configurations. It reads CSV files containing accuracy
data and generates grouped line plots for comparison.

The script expects CSV files with columns: head_name, accuracy, correct_count, total_count
where head_name follows the pattern 'wmdp_head_N' with N being the layer number.
"""

import os
from typing import List, Dict, Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def configure_plot_style() -> None:
    """Configure matplotlib plotting style for publication-quality figures."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['cmr10', 'Computer Modern Serif', 'DejaVu Serif', 'Times New Roman'],
        'mathtext.fontset': 'cm',  # Computer Modern for math
        'font.size': 16,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 12
    })


def process_accuracy_data(csv_file: str) -> Optional[pd.DataFrame]:
    """
    Process probe accuracy CSV data for plotting.
    
    Args:
        csv_file: Path to CSV file containing accuracy data
        
    Returns:
        Processed DataFrame with layer and accuracy columns, or None if file not found
        
    Expected CSV format:
        - head_name: Format 'wmdp_head_N' where N is layer number
        - accuracy: Accuracy value (0-1)
        - correct_count: Number of correct predictions
        - total_count: Total number of predictions
    """
    if not os.path.exists(csv_file):
        print(f"Warning: File {csv_file} not found, skipping...")
        return None
    
    try:
        df = pd.read_csv(csv_file)
        
        # Extract layer numbers from head names (e.g., 'wmdp_head_5' -> 5)
        df['original_layer'] = df['head_name'].apply(lambda x: int(x.split('_')[-1]))
        
        # Reverse layer ordering (higher original layer = lower plot layer)
        max_layer = df['original_layer'].max()
        df['layer'] = max_layer - df['original_layer']
        
        # Sort by layer for proper line plotting
        df = df.sort_values('layer')
        
        return df[['layer', 'accuracy']]
        
    except Exception as e:
        print(f"Error processing {csv_file}: {e}")
        return None


def plot_probe_accuracy_grouped(
    data_config: List[Dict[str, str]], 
    title: str = "Probe Accuracy",
    output_file: str = "probe_accuracy_grouped.pdf",
    figsize: Tuple[int, int] = (12, 8),
    random_chance: float = 0.25
) -> plt.Figure:
    """
    Create a grouped line plot of probe accuracy across layers for different models.
    
    Args:
        data_config: List of dictionaries with configuration for each line:
            - csv_file: Path to CSV file
            - label: Legend label for the line
            - color: Line color (hex or named color)
            - linestyle: Line style ('-', '--', '-.', ':')
        title: Plot title (currently commented out in styling)
        output_file: Output filename for saving the plot
        figsize: Figure size as (width, height)
        random_chance: Y-value for horizontal reference line
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    plotted_count = 0
    
    for config in data_config:
        csv_file = config['csv_file']
        label = config['label']
        color = config['color']
        linestyle = config['linestyle']
        
        # Process the data
        df = process_accuracy_data(csv_file)
        if df is None:
            continue
            
        # Plot the accuracy line
        ax.plot(df['layer'], df['accuracy'], 
                label=label, 
                color=color,
                linestyle=linestyle,
                linewidth=2.5,
                alpha=0.8)
        
        plotted_count += 1
    
    if plotted_count == 0:
        print("Warning: No data files found to plot")
        return fig
    
    # Add reference line for random chance performance
    ax.axhline(y=random_chance, color='#9467bd', linestyle='-', 
               alpha=0.7, label=f'Random chance ({random_chance:.2f})')
    
    # Apply styling
    ax.set_xlabel('Layer', fontsize=30)
    ax.set_ylabel('Accuracy', fontsize=30)
    # ax.set_title(title, fontsize=30)  # Commented out as in original
    ax.legend(fontsize=25)
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure
    fig.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Plot saved as {output_file}")
    
    return fig


def main() -> None:
    """
    Main function to generate probe accuracy comparison plots.
    
    This function configures the plot style and generates visualizations comparing
    different unlearning methods (ELM, RMU) with their respective variants.
    """
    # Configure plotting style
    configure_plot_style()
    
    # Update these paths to the correct location of the csv files
    zephyr_base_path = "probe_accuracy_summary_zephyr_base.csv"
    zephyr_elm_path = "probe_accuracy_summary_baulab_elm-zephyr-7b-beta.csv"
    zephyr_elm_hindi_filler_path = "probe_accuracy_hindi_baolab_elm_zephyr_7b_beta.csv"
    zephyr_rmu_path = "probe_accuracy_summary_zephyr_rmu.csv"
    zephyr_rmu_hindi_filler_path = "probe_accuracy_hindi_filler_zephyr_rmu.csv"

    
    data_config = [
        # Base model
        {
            'csv_file': zephyr_base_path,
            'label': 'Base model',
            'color': '#1f77b4',
            'linestyle': '-'
        },
        
        # ELM unlearned models (red color family)
        {
            'csv_file': zephyr_elm_path,
            'label': 'ELM',
            'color': '#d62728',
            'linestyle': '-'  # solid line
        },
        {
            'csv_file': zephyr_elm_hindi_filler_path,
            'label': 'ELM, Hindi filler',
            'color': '#d62728',
            'linestyle': '--'  # dashed line
        },
        
        # RMU unlearned models (green color family)
        {
            'csv_file': zephyr_rmu_path,
            'label': 'RMU',
            'color': '#2ca02c',
            'linestyle': '-'  # solid line
        },
        {
            'csv_file': zephyr_rmu_hindi_filler_path,
            'label': 'RMU, Hindi filler',
            'color': '#2ca02c',
            'linestyle': '--'  # dashed line
        }
    ]
    
    # Generate the plot
    title = "Probe Accuracy for Unlearning Methods on Zephyr 7b"
    fig = plot_probe_accuracy_grouped(
        data_config=data_config,
        title=title,
        output_file='probe_accuracy_grouped.pdf'
    )
    
    # Display the plot
    plt.show()


if __name__ == "__main__":
    main()