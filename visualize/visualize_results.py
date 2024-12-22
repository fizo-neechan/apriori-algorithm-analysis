import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from pathlib import Path
import argparse

class AprioriVisualizer:
    def __init__(self, dataset_name, results_dir='results'):
        self.dataset_name = dataset_name
        self.results_dir = Path(results_dir)
        
        # Load all CSV files
        self.summary_df = pd.read_csv(self.results_dir / f"{dataset_name}_summary.csv")
        self.size_dist_df = pd.read_csv(self.results_dir / f"{dataset_name}_size_distribution.csv")
        self.support_dist_df = pd.read_csv(self.results_dir / f"{dataset_name}_support_distribution.csv")
        self.performance_df = pd.read_csv(self.results_dir / f"{dataset_name}_performance.csv")
        
        # Create output directory for plots
        self.plots_dir = Path('plots')
        self.plots_dir.mkdir(exist_ok=True)
        
    def plot_size_distribution(self):
        """Create bar plot of itemset sizes distribution"""
        plt.figure(figsize=(10, 6))
        sns.barplot(data=self.size_dist_df, x='Size', y='Count')
        plt.title(f'Distribution of Frequent Itemset Sizes - {self.dataset_name}')
        plt.xlabel('Itemset Size')
        plt.ylabel('Number of Frequent Itemsets')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(self.plots_dir / f"{self.dataset_name}_size_distribution.png")
        plt.close()

    def plot_support_distribution(self):
        """Create box plot of support values by itemset size"""
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=self.support_dist_df, x='ItemsetSize', y='Support')
        plt.title(f'Support Distribution by Itemset Size - {self.dataset_name}')
        plt.xlabel('Itemset Size')
        plt.ylabel('Support Value')
        plt.tight_layout()
        plt.savefig(self.plots_dir / f"{self.dataset_name}_support_distribution.png")
        plt.close()

    def plot_performance_metrics(self):
        """Create bar plot of performance metrics"""
        # Filter only time-related metrics
        time_metrics = self.performance_df[self.performance_df['Metric'].isin(['Data Loading', 'Processing', 'Total'])]
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=time_metrics, x='Metric', y='Time(seconds)')
        plt.title(f'Performance Metrics - {self.dataset_name}')
        plt.xlabel('Metric')
        plt.ylabel('Time (seconds)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.plots_dir / f"{self.dataset_name}_performance.png")
        plt.close()

    def plot_item_network(self, min_size=2, max_size=3):
        """Create network graph of item relationships"""
        # Filter itemsets of desired sizes
        filtered_df = self.summary_df[
            (self.summary_df['Size'] >= min_size) & 
            (self.summary_df['Size'] <= max_size)
        ]

        # Create network graph
        G = nx.Graph()
        
        # Add edges for each itemset
        for _, row in filtered_df.iterrows():
            items = row['Items'].split(',')
            # Add edges between all pairs in the itemset
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    if G.has_edge(items[i], items[j]):
                        G[items[i]][items[j]]['weight'] += 1
                    else:
                        G.add_edge(items[i], items[j], weight=1)

        # Create network visualization
        plt.figure(figsize=(15, 15))
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Draw the network
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                             node_size=1000, alpha=0.6)
        nx.draw_networkx_edges(G, pos, alpha=0.2)
        nx.draw_networkx_labels(G, pos, font_size=8)
        
        plt.title(f'Item Relationship Network - {self.dataset_name}\n(Itemsets of size {min_size}-{max_size})')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(self.plots_dir / f"{self.dataset_name}_item_network.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_support_heatmap(self, top_n=20):
        """Create heatmap of support values for top frequent itemsets"""
        # Get top N itemsets by support
        top_itemsets = self.summary_df.nlargest(top_n, 'Support')
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pd.DataFrame({
            'Support': top_itemsets['Support'],
            'Size': top_itemsets['Size']
        }).set_index(top_itemsets['Items']).T, 
            cmap='YlOrRd', annot=True, fmt='.3f')
        
        plt.title(f'Top {top_n} Frequent Itemsets - {self.dataset_name}')
        plt.xlabel('Itemsets')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.plots_dir / f"{self.dataset_name}_support_heatmap.png")
        plt.close()

    def generate_all_plots(self):
        """Generate all visualizations"""
        print(f"Generating visualizations for {self.dataset_name}...")
        self.plot_size_distribution()
        self.plot_support_distribution()
        self.plot_performance_metrics()
        self.plot_item_network()
        self.plot_support_heatmap()
        print(f"Visualizations saved in {self.plots_dir}")

def main():
    parser = argparse.ArgumentParser(description='Generate visualizations for Apriori results')
    parser.add_argument('dataset_name', help='Name of the dataset (without file extensions)')
    parser.add_argument('--results-dir', default='results', help='Directory containing results CSV files')
    
    args = parser.parse_args()
    
    visualizer = AprioriVisualizer(args.dataset_name, args.results_dir)
    visualizer.generate_all_plots()

if __name__ == "__main__":
    main()