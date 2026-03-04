import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from scipy.stats import linregress

def visualize_errors(csv_file_path):
    """
    Loads data from a CSV, processes it, and generates presentable plots to visualize
    translation and rotation errors against total area and area per number of areas.
    """
    if not os.path.exists(csv_file_path):
        print(f"Error: CSV file not found at {csv_file_path}")
        return

    # Load the CSV data
    df = pd.read_csv(csv_file_path)

    # Filter out average rows to only use raw data points
    df = df[~df['SchemeID'].astype(str).str.contains('Avg', case=False, na=False)]

    # Define schema mapping based on user-provided data
    schema_map = {
        'S1': {'total_area': 1187.176, 'area_per_num_areas': 395.7253333},
        'S2': {'total_area': 250.812, 'area_per_num_areas': 62.703},
        'S3': {'total_area': 363.353, 'area_per_num_areas': 72.6706},
        'S5': {'total_area': 1462.25898736715316, 'area_per_num_areas': 365.5647468417883}
    }

    # Add 'total_area' and 'area_per_num_areas' columns to the DataFrame
    df['total_area'] = df['SchemeID'].map(lambda x: schema_map.get(x, {}).get('total_area'))
    df['area_per_num_areas'] = df['SchemeID'].map(lambda x: schema_map.get(x, {}).get('area_per_num_areas'))
    df.dropna(subset=['total_area', 'area_per_num_areas'], inplace=True)

    # Convert errors to numeric just in case
    df['transl_err'] = pd.to_numeric(df['transl_err'], errors='coerce')
    df['rot_err'] = pd.to_numeric(df['rot_err'], errors='coerce')
    df.dropna(subset=['transl_err', 'rot_err'], inplace=True)

    # Styling for presentable graphs
    sns.set_theme(style="whitegrid", context="talk")
    palette = sns.color_palette("husl", len(df['SchemeID'].unique()))

    def create_presentable_plot(x_col, xlabel, x_filename):
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        fig.suptitle(f'Registration Errors vs {xlabel}', fontsize=20, weight='bold', y=1.05)

        for ax, y_col, ylabel in zip(axes, ['transl_err', 'rot_err'], ['Translational Error (mm)', 'Rotational Error (degrees)']):
            # Scatter plot of all data points
            sns.scatterplot(
                data=df, 
                x=x_col, 
                y=y_col,
                hue='SchemeID',
                s=120, 
                alpha=0.7, 
                edgecolor='w',
                palette=palette,
                ax=ax
            )

            # Calculate and plot linear regression with Seaborn
            sns.regplot(
                data=df, 
                x=x_col, 
                y=y_col, 
                scatter=False, 
                color='black', 
                line_kws={'linestyle': '--', 'linewidth': 2, 'alpha': 0.7}, 
                ax=ax
            )

            # Get means to plot as large X marks and for annotations
            means = df.groupby([x_col, 'SchemeID'])[y_col].mean().reset_index()
            ax.scatter(means[x_col], means[y_col], color='red', marker='X', s=200, label='Mean Error', zorder=10)
            
            # Annotate Scheme IDs on the means
            for i, row in means.iterrows():
                ax.annotate(
                    row['SchemeID'],
                    (row[x_col], row[y_col]),
                    textcoords="offset points", xytext=(0,15), ha='center',
                    fontsize=12, fontweight='bold', color='black',
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8)
                )

            # Perform regression and display stats box
            slope, intercept, r_value, p_value, std_err = linregress(df[x_col], df[y_col])
            stats_text = f"Trend:\n$y = {slope:.3e}x + {intercept:.2f}$\n$R^2 = {r_value**2:.2f}$\np-value = {p_value:.3f}"
            
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                     fontsize=12, va='top', bbox=dict(boxstyle="round", alpha=0.9, facecolor='white', edgecolor='gray'))

            ax.set_title(ylabel, fontsize=16, pad=10)
            ax.set_xlabel(xlabel, fontsize=14)
            ax.set_ylabel(ylabel, fontsize=14)
            if y_col == 'transl_err':
                ax.legend(title='Scheme ID', loc='best')
            else:
                ax.get_legend().remove() if ax.get_legend() else None

        plt.tight_layout()
        filename = f"errors_vs_{x_filename}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Generated presentable plot: {filename}")

    def create_variance_plot(x_col, xlabel, x_filename):
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        fig.suptitle(f'Error Variance vs {xlabel}', fontsize=20, weight='bold', y=1.05)

        # Calculate variance for each group
        var_df = df.groupby([x_col, 'SchemeID'])[['transl_err', 'rot_err']].var().reset_index()

        for ax, y_col, ylabel in zip(axes, ['transl_err', 'rot_err'], ['Translational Variance (mm²)', 'Rotational Variance (deg²)']):
            sns.scatterplot(
                data=var_df,
                x=x_col,
                y=y_col,
                hue='SchemeID',
                s=200,
                alpha=0.9,
                edgecolor='w',
                palette=palette,
                ax=ax
            )
            
            sns.regplot(
                data=var_df,
                x=x_col,
                y=y_col,
                scatter=False,
                color='black',
                line_kws={'linestyle': '--', 'linewidth': 2, 'alpha': 0.7},
                ax=ax
            )
            
            for i, row in var_df.iterrows():
                ax.annotate(
                    row['SchemeID'],
                    (row[x_col], row[y_col]),
                    textcoords="offset points", xytext=(0,15), ha='center',
                    fontsize=12, fontweight='bold', color='black',
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8)
                )

            slope, intercept, r_value, p_value, std_err = linregress(var_df[x_col], var_df[y_col])
            stats_text = f"Trend:\n$y = {slope:.3e}x + {intercept:.2f}$\n$R^2 = {r_value**2:.2f}$\np-value = {p_value:.3f}"
            
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                     fontsize=12, va='top', bbox=dict(boxstyle="round", alpha=0.9, facecolor='white', edgecolor='gray'))

            ax.set_title(ylabel, fontsize=16, pad=10)
            ax.set_xlabel(xlabel, fontsize=14)
            ax.set_ylabel(ylabel, fontsize=14)
            if y_col == 'transl_err':
                ax.legend(title='Scheme ID', loc='best')
            else:
                ax.get_legend().remove() if ax.get_legend() else None

        plt.tight_layout()
        filename = f"variance_vs_{x_filename}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Generated variance plot: {filename}")

    # Generate the refined plots
    create_presentable_plot('total_area', 'Total Area of ROI ($mm^2$)', 'total_area')
    create_presentable_plot('area_per_num_areas', 'Area Per Number of Areas ($mm^2$)', 'area_per_num_areas')
    
    # Generate the variance plots
    create_variance_plot('total_area', 'Total Area of ROI ($mm^2$)', 'total_area')
    create_variance_plot('area_per_num_areas', 'Area Per Number of Areas ($mm^2$)', 'area_per_num_areas')

    print("Cleanup complete. See the generated high-quality images.")

# Define the path to your CSV file
csv_path = "/home/mitchell/Downloads/Recollection.csv"

if __name__ == "__main__":
    visualize_errors(csv_path)
