import os
import pandas as pd


def consolidate_metrics():
    # Directory containing the CSV files
    directory = "./data/results"

    # Key metrics to extract
    all_metrics = [
        'MAP@1', 'MAP@10', 'MAP@3', 'MAP@5',
        'NDCG@1', 'NDCG@10', 'NDCG@3', 'NDCG@5',
        'P@1', 'P@10', 'P@3', 'P@5',
        'Recall@1', 'Recall@10', 'Recall@3', 'Recall@5'
    ]

    # List to hold all rows
    results = []

    # Iterate through all CSV files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            filepath = os.path.join(directory, filename)
            try:
                df = pd.read_csv(filepath)
                metrics = {metric: pd.to_numeric(df[metric].values[0], errors='coerce') for metric in all_metrics if metric in df.columns}
                metrics['Setup'] = filename  # Keep full filename
                results.append(metrics)
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    # Create final DataFrame
    results_df = pd.DataFrame(results)

    # Compute average across available metrics
    results_df['Average'] = results_df[all_metrics].mean(axis=1)

    # Reorder columns to make Setup first
    column_order = ['Setup'] + [m for m in all_metrics if m in results_df.columns] + ['Average']
    results_df = results_df[column_order]

    # Sort by performance
    results_df = results_df.sort_values(by='Average', ascending=False)

    # Export to Excel
    output_path = f"{directory}/compiled_rag_metrics.xlsx"
    results_df.to_excel(output_path, index=False)

    print(f"Metrics Excel file written to: {output_path}")
