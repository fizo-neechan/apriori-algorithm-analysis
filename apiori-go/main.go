package main

import (
    "fmt"
    "log"
    "os"
    "path/filepath"
    "strings"
    "time"
)

func getOutputBasename(filename string) string {
    if filename == "" {
        return "example_dataset"
    }
    // Remove file extension and directory path
    base := filepath.Base(filename)
    return strings.TrimSuffix(base, filepath.Ext(base))
}

func main() {
    startTime := time.Now()
    var dataLoadTime time.Duration
    var processingTime time.Duration
    var dataset Dataset
    
    // Check if a file is provided as argument
    if len(os.Args) > 1 {
        // Load dataset from file
        filename := os.Args[1]
        loadStart := time.Now()
        var err error
        dataset, err = LoadDataset(filename)
        if err != nil {
            log.Fatal(err)
        }
        dataLoadTime = time.Since(loadStart)
        
        fmt.Printf("Running Apriori on dataset from %s\n", filename)
        
        // Run Apriori with file data
        processStart := time.Now()
        miner := NewAprioriMiner(dataset, 0.4) // 40% minimum support
        miner.Mine()
        processingTime = time.Since(processStart)
        
        printResults(miner)
        
        // Calculate total time
        totalTime := time.Since(startTime)
        
        // Create timing metrics
        metrics := TimingMetrics{
            DataLoadTime:    dataLoadTime.Seconds(),
            ProcessingTime:  processingTime.Seconds(),
            TotalTime:      totalTime.Seconds(),
        }
        
        // Output results to CSV files
        if err := miner.OutputResults(getOutputBasename(filename), metrics); err != nil {
            log.Printf("Error writing results to CSV: %v", err)
        } else {
            fmt.Println("\nResults have been written to CSV files in the 'results' directory.")
            fmt.Printf("\nPerformance Metrics:\n")
            fmt.Printf("Data Loading Time: %.2f seconds\n", metrics.DataLoadTime)
            fmt.Printf("Processing Time: %.2f seconds\n", metrics.ProcessingTime)
            fmt.Printf("Total Time: %.2f seconds\n", metrics.TotalTime)
        }
        
    } else {
        // Use example dataset
        dataset = Dataset{
            {"bread", "milk"},
            {"bread", "diaper", "beer", "eggs"},
            {"milk", "diaper", "beer", "cola"},
            {"bread", "milk", "diaper", "beer"},
            {"bread", "milk", "diaper", "cola"},
        }
        
        fmt.Println("Running Apriori on example dataset")
        
        // Process example dataset
        processStart := time.Now()
        miner := NewAprioriMiner(dataset, 0.4)
        miner.Mine()
        processingTime = time.Since(processStart)
        
        printResults(miner)
        
        // Calculate total time
        totalTime := time.Since(startTime)
        
        // Create timing metrics
        metrics := TimingMetrics{
            DataLoadTime:    0, // No loading time for example dataset
            ProcessingTime:  processingTime.Seconds(),
            TotalTime:      totalTime.Seconds(),
        }
        
        // Output results to CSV files
        if err := miner.OutputResults("example_dataset", metrics); err != nil {
            log.Printf("Error writing results to CSV: %v", err)
        } else {
            fmt.Println("\nResults have been written to CSV files in the 'results' directory.")
            fmt.Printf("\nPerformance Metrics:\n")
            fmt.Printf("Processing Time: %.2f seconds\n", metrics.ProcessingTime)
            fmt.Printf("Total Time: %.2f seconds\n", metrics.TotalTime)
        }
    }
}

func printResults(miner *AprioriMiner) {
    fmt.Println("\nFrequent Itemsets:")
    for k, itemsets := range miner.frequentSets {
        fmt.Printf("\n%d-itemsets:\n", k)
        for _, itemset := range itemsets {
            items := sortedItems(itemset)
            fmt.Printf("  %v (Support: %.2f)\n", items, miner.calculateSupport(itemset))
        }
    }
}