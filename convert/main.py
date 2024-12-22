import argparse
import csv
import os
from pathlib import Path

class DatasetConverter:
    def __init__(self, input_file, output_file=None):
        self.input_file = input_file
        self.output_file = output_file or f"{Path(input_file).stem}_processed.txt"
        
    def convert_chess(self):
        """
        Convert chess dataset format.
        Each row contains binary/categorical attributes that we'll convert to items.
        """
        chess_attributes = [
            'wk_file', 'wk_rank', 'wr_file', 'wr_rank', 
            'bk_file', 'bk_rank'
        ]
        
        with open(self.input_file, 'r') as infile, open(self.output_file, 'w') as outfile:
            for line in infile:
                items = []
                values = line.strip().split(',')
                
                # Convert each attribute-value pair to an item
                for attr, val in zip(chess_attributes, values):
                    if val:  # Skip empty values
                        items.append(f"{attr}_{val}")
                
                # Write space-separated items
                outfile.write(' '.join(items) + '\n')
                
        print(f"Converted chess dataset saved to {self.output_file}")

    def convert_connect(self):
        """
        Convert connect dataset format.
        Each row represents a game state with multiple attributes.
        """
        with open(self.input_file, 'r') as infile, open(self.output_file, 'w') as outfile:
            for line in infile:
                items = []
                values = line.strip().split()
                
                # Convert each value to an item with position
                for pos, val in enumerate(values, 1):
                    if val != '0':  # Skip empty positions
                        items.append(f"pos{pos}_{val}")
                
                # Write space-separated items
                outfile.write(' '.join(items) + '\n')
                
        print(f"Converted connect dataset saved to {self.output_file}")

    def convert_accident(self):
        """
        Convert accident dataset format.
        Each row contains multiple attributes about an accident.
        """
        with open(self.input_file, 'r') as infile, open(self.output_file, 'w') as outfile:
            for line in infile:
                items = []
                values = line.strip().split()
                
                # Convert each attribute to an item
                for val in values:
                    if val:  # Skip empty values
                        items.append(f"attr_{val}")
                
                # Write space-separated items
                outfile.write(' '.join(items) + '\n')
                
        print(f"Converted accident dataset saved to {self.output_file}")

    def create_sample(self, n_rows=1000):
        """
        Create a sample dataset with the first n_rows.
        """
        sample_file = f"{Path(self.output_file).stem}_sample.txt"
        
        with open(self.output_file, 'r') as infile, open(sample_file, 'w') as outfile:
            for i, line in enumerate(infile):
                if i >= n_rows:
                    break
                outfile.write(line)
                
        print(f"Sample dataset with {n_rows} rows saved to {sample_file}")

def main():
    parser = argparse.ArgumentParser(description='Convert datasets for Apriori implementation')
    parser.add_argument('input_file', help='Input dataset file')
    parser.add_argument('--type', choices=['chess', 'connect', 'accident'], 
                      required=True, help='Type of dataset')
    parser.add_argument('--output', help='Output file name (optional)')
    parser.add_argument('--sample', type=int, help='Create sample with N rows')
    
    args = parser.parse_args()
    
    converter = DatasetConverter(args.input_file, args.output)
    
    # Convert based on dataset type
    if args.type == 'chess':
        converter.convert_chess()
    elif args.type == 'connect':
        converter.convert_connect()
    elif args.type == 'accident':
        converter.convert_accident()
    
    # Create sample if requested
    if args.sample:
        converter.create_sample(args.sample)

if __name__ == "__main__":
    main()