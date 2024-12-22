import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime

class PerformanceAnalyzer:
    def __init__(self, results_dir='results', output_dir='performance_analysis'):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Store dataset metrics
        self.datasets = {}
        self.comparative_metrics = {}
        
    def load_dataset_metrics(self, dataset_name):
        """Load performance metrics for a dataset"""
        perf_file = self.results_dir / f"{dataset_name}_performance.csv"
        summary_file = self.results_dir / f"{dataset_name}_summary.csv"
        size_file = self.results_dir / f"{dataset_name}_size_distribution.csv"
        
        try:
            perf_df = pd.read_csv(perf_file)
            summary_df = pd.read_csv(summary_file)
            size_df = pd.read_csv(size_file)
            
            # Get dataset size (number of transactions)
            total_transactions = float(perf_df[perf_df['Metric'] == 'Total Transactions']['Time(seconds)'].iloc[0])
            
            self.datasets[dataset_name] = {
                'performance': perf_df,
                'summary': summary_df,
                'size_distribution': size_df,
                'total_transactions': total_transactions
            }
            
            # Calculate metrics
            total_itemsets = size_df['Count'].sum()
            avg_support = summary_df['Support'].mean()
            max_itemset_size = size_df['Size'].max()
            processing_time = float(perf_df[perf_df['Metric'] == 'Processing']['Time(seconds)'].iloc[0])
            total_time = float(perf_df[perf_df['Metric'] == 'Total']['Time(seconds)'].iloc[0])
            
            # Calculate normalized metrics
            self.comparative_metrics[dataset_name] = {
                'total_itemsets': total_itemsets,
                'avg_support': avg_support,
                'max_itemset_size': max_itemset_size,
                'processing_time': processing_time,
                'total_time': total_time,
                'total_transactions': total_transactions,
                # Normalized metrics
                'time_per_transaction': total_time / total_transactions,
                'time_per_itemset': processing_time / total_itemsets,
                'itemsets_per_transaction': total_itemsets / total_transactions,
                'processing_speed': total_itemsets / processing_time,
                'transaction_throughput': total_transactions / total_time
            }
            
        except FileNotFoundError as e:
            print(f"Error loading files for dataset {dataset_name}: {e}")
            return False
        return True

    def plot_timing_comparison(self):
        """Create comparative timing analysis plot"""
        timing_data = []
        for dataset, metrics in self.comparative_metrics.items():
            timing_data.append({
                'Dataset': dataset,
                'Processing Time': metrics['processing_time'],
                'Total Time': metrics['total_time']
            })
        
        timing_df = pd.DataFrame(timing_data)
        
        plt.figure(figsize=(12, 6))
        x = np.arange(len(timing_df))
        width = 0.35
        
        plt.bar(x - width/2, timing_df['Processing Time'], width, label='Processing Time')
        plt.bar(x + width/2, timing_df['Total Time'], width, label='Total Time')
        
        plt.xlabel('Dataset')
        plt.ylabel('Time (seconds)')
        plt.title('Processing vs Total Time Comparison')
        plt.xticks(x, timing_df['Dataset'])
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'timing_comparison.png')
        plt.close()

    def plot_efficiency_metrics(self):
        """Create efficiency metrics visualization"""
        efficiency_data = []
        for dataset, metrics in self.comparative_metrics.items():
            efficiency_data.append({
                'Dataset': dataset,
                'Itemsets per Second': metrics['total_itemsets'] / metrics['processing_time'],
                'Avg Support': metrics['avg_support']
            })
        
        eff_df = pd.DataFrame(efficiency_data)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Itemsets per second
        sns.barplot(data=eff_df, x='Dataset', y='Itemsets per Second', ax=ax1)
        ax1.set_title('Processing Efficiency')
        ax1.tick_params(axis='x', rotation=45)
        
        # Average support
        sns.barplot(data=eff_df, x='Dataset', y='Avg Support', ax=ax2)
        ax2.set_title('Average Support Values')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'efficiency_metrics.png')
        plt.close()

    def plot_size_vs_time(self):
        """Analyze relationship between itemset size and processing time"""
        plt.figure(figsize=(10, 6))
        
        for dataset, data in self.datasets.items():
            size_dist = data['size_distribution']
            total_time = self.comparative_metrics[dataset]['processing_time']
            
            # Calculate time per itemset size (estimated)
            size_dist['Est_Time'] = size_dist['Count'] * total_time / size_dist['Count'].sum()
            
            plt.plot(size_dist['Size'], size_dist['Est_Time'], 
                    marker='o', label=dataset)
        
        plt.xlabel('Itemset Size')
        plt.ylabel('Estimated Processing Time (s)')
        plt.title('Processing Time vs Itemset Size')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'size_vs_time.png')
        plt.close()

    def plot_normalized_metrics(self):
        """Create visualization of normalized performance metrics"""
        norm_data = []
        metrics_to_plot = [
            'time_per_transaction',
            'time_per_itemset',
            'itemsets_per_transaction',
            'processing_speed',
            'transaction_throughput'
        ]
        
        for dataset, metrics in self.comparative_metrics.items():
            for metric in metrics_to_plot:
                norm_data.append({
                    'Dataset': dataset,
                    'Metric': metric,
                    'Value': metrics[metric]
                })
        
        norm_df = pd.DataFrame(norm_data)
        
        # Create subplot for each normalized metric
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics_to_plot):
            if i < len(axes):
                metric_data = norm_df[norm_df['Metric'] == metric]
                sns.barplot(data=metric_data, x='Dataset', y='Value', ax=axes[i])
                axes[i].set_title(metric.replace('_', ' ').title())
                axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'normalized_metrics.png')
        plt.close()

    def generate_performance_report(self):
        """Generate a detailed performance report with normalized metrics"""
        report = ["Performance Analysis Report", "=" * 25 + "\n"]
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Overall statistics
        report.append("Dataset Statistics:")
        report.append("-" * 20)
        
        for dataset, metrics in self.comparative_metrics.items():
            report.append(f"\nDataset: {dataset}")
            report.append(f"  Dataset Size: {metrics['total_transactions']} transactions")
            report.append(f"  Total Itemsets: {metrics['total_itemsets']}")
            report.append(f"  Maximum Itemset Size: {metrics['max_itemset_size']}")
            
            report.append("\nRaw Performance Metrics:")
            report.append(f"  Processing Time: {metrics['processing_time']:.3f} seconds")
            report.append(f"  Total Time: {metrics['total_time']:.3f} seconds")
            
            report.append("\nNormalized Metrics:")
            report.append(f"  Time per Transaction: {metrics['time_per_transaction']:.6f} seconds")
            report.append(f"  Time per Itemset: {metrics['time_per_itemset']:.6f} seconds")
            report.append(f"  Itemsets per Transaction: {metrics['itemsets_per_transaction']:.2f}")
            report.append(f"  Processing Speed: {metrics['processing_speed']:.2f} itemsets/second")
            report.append(f"  Transaction Throughput: {metrics['transaction_throughput']:.2f} transactions/second")
            report.append("\n" + "-" * 50)
        
        # Write report to file
        with open(self.output_dir / 'performance_report.txt', 'w') as f:
            f.write('\n'.join(report))
            
    def analyze_all(self, dataset_names):
        """Perform complete performance analysis"""
        # Load all datasets
        for dataset in dataset_names:
            if not self.load_dataset_metrics(dataset):
                continue
        
        # Generate all analyses
        self.plot_timing_comparison()
        self.plot_efficiency_metrics()
        self.plot_size_vs_time()
        self.plot_normalized_metrics()  # Add normalized metrics plot
        self.generate_performance_report()
        
        print(f"Analysis complete. Results saved in {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Analyze performance metrics of Apriori algorithm runs')
    parser.add_argument('datasets', nargs='+', help='Names of datasets to analyze')
    parser.add_argument('--results-dir', default='results', help='Directory containing results files')
    parser.add_argument('--output-dir', default='performance_analysis', help='Directory for output files')
    
    args = parser.parse_args()
    
    analyzer = PerformanceAnalyzer(args.results_dir, args.output_dir)
    analyzer.analyze_all(args.datasets)

if __name__ == "__main__":
    main()