package main

import (
	"bufio"
	"fmt"
	"os"
	"sort"
	"strings"
)

// ItemSet represents a set of items
type ItemSet map[string]bool

// Transaction represents a single transaction containing multiple items
type Transaction []string

// Dataset represents a collection of transactions
type Dataset []Transaction

// AprioriMiner implements the Apriori algorithm
type AprioriMiner struct {
	minSupport     float64
	dataset        Dataset
	frequentSets   map[int][]ItemSet
	transactionLen int
}

// NewAprioriMiner creates a new instance of AprioriMiner
func NewAprioriMiner(dataset Dataset, minSupport float64) *AprioriMiner {
	return &AprioriMiner{
		minSupport:     minSupport,
		dataset:        dataset,
		frequentSets:   make(map[int][]ItemSet),
		transactionLen: len(dataset),
	}
}

// generateCandidates generates candidate itemsets of size k+1 from frequent itemsets of size k
func (am *AprioriMiner) generateCandidates(frequentSets []ItemSet, size int) []ItemSet {
	candidates := make([]ItemSet, 0)
	
	for i := 0; i < len(frequentSets); i++ {
		items1 := sortedItems(frequentSets[i])
		for j := i + 1; j < len(frequentSets); j++ {
			items2 := sortedItems(frequentSets[j])
			
			// Check if first k-1 items are same
			canCombine := true
			for k := 0; k < size-1; k++ {
				if items1[k] != items2[k] {
					canCombine = false
					break
				}
			}
			
			if canCombine && items1[size-1] < items2[size-1] {
				// Create new candidate
				newSet := make(ItemSet)
				for k := 0; k < size-1; k++ {
					newSet[items1[k]] = true
				}
				newSet[items1[size-1]] = true
				newSet[items2[size-1]] = true
				
				// Add only if all subsets are frequent
				if am.isValidCandidate(newSet, frequentSets) {
					candidates = append(candidates, newSet)
				}
			}
		}
	}
	return candidates
}

// isValidCandidate checks if all subsets of candidate are frequent
func (am *AprioriMiner) isValidCandidate(candidate ItemSet, frequentSets []ItemSet) bool {
	items := sortedItems(candidate)
	
	// Generate all subsets of size k-1
	for i := range items {
		subset := make(ItemSet)
		for j, item := range items {
			if i != j {
				subset[item] = true
			}
		}
		
		// Check if subset exists in frequent itemsets
		found := false
		for _, freqSet := range frequentSets {
			if setsEqual(subset, freqSet) {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	return true
}

// calculateSupport calculates support for a candidate itemset
func (am *AprioriMiner) calculateSupport(candidate ItemSet) float64 {
	count := 0
	for _, transaction := range am.dataset {
		if isSubset(candidate, transaction) {
			count++
		}
	}
	return float64(count) / float64(am.transactionLen)
}

// Mine performs the Apriori algorithm
func (am *AprioriMiner) Mine() {
	// Generate frequent 1-itemsets
	candidates := am.generateInitialCandidates()
	k := 1
	
	for len(candidates) > 0 {
		frequent := make([]ItemSet, 0)
		
		// Calculate support for each candidate
		for _, candidate := range candidates {
			support := am.calculateSupport(candidate)
			if support >= am.minSupport {
				frequent = append(frequent, candidate)
			}
		}
		
		if len(frequent) > 0 {
			am.frequentSets[k] = frequent
			// Generate candidates for next iteration
			candidates = am.generateCandidates(frequent, k)
			k++
		} else {
			break
		}
	}
}

// generateInitialCandidates generates 1-itemsets from the dataset
func (am *AprioriMiner) generateInitialCandidates() []ItemSet {
	itemCounts := make(map[string]int)
	
	// Count occurrences of each item
	for _, transaction := range am.dataset {
		for _, item := range transaction {
			itemCounts[item]++
		}
	}
	
	// Generate candidates meeting minimum support
	candidates := make([]ItemSet, 0)
	for item, count := range itemCounts {
		support := float64(count) / float64(am.transactionLen)
		if support >= am.minSupport {
			itemset := make(ItemSet)
			itemset[item] = true
			candidates = append(candidates, itemset)
		}
	}
	
	return candidates
}

// Helper functions
func sortedItems(set ItemSet) []string {
	items := make([]string, 0, len(set))
	for item := range set {
		items = append(items, item)
	}
	sort.Strings(items)
	return items
}

func setsEqual(set1, set2 ItemSet) bool {
	if len(set1) != len(set2) {
		return false
	}
	for item := range set1 {
		if !set2[item] {
			return false
		}
	}
	return true
}

func isSubset(set ItemSet, transaction Transaction) bool {
	for item := range set {
		found := false
		for _, transItem := range transaction {
			if item == transItem {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	return true
}

// TimingMetrics stores timing information for the mining process
type TimingMetrics struct {
    DataLoadTime    float64
    ProcessingTime  float64
    TotalTime      float64
}

// OutputResults writes the mining results and timing metrics to CSV files
func (am *AprioriMiner) OutputResults(baseFilename string, metrics TimingMetrics) error {
    // Create a directory for the output if it doesn't exist
    err := os.MkdirAll("results", 0755)
    if err != nil {
        return fmt.Errorf("failed to create results directory: %v", err)
    }

    // Create summary file with all itemsets
    summaryFile, err := os.Create(fmt.Sprintf("results/%s_summary.csv", baseFilename))
    if err != nil {
        return fmt.Errorf("failed to create summary file: %v", err)
    }
    defer summaryFile.Close()

    // Write summary header
    summaryFile.WriteString("Size,Items,Support\n")

    // Write each itemset to the summary file
    for k, itemsets := range am.frequentSets {
        for _, itemset := range itemsets {
            items := strings.Join(sortedItems(itemset), ",")
            support := am.calculateSupport(itemset)
            summaryFile.WriteString(fmt.Sprintf("%d,\"%s\",%f\n", k, items, support))
        }
    }

    // Create size distribution file
    sizeFile, err := os.Create(fmt.Sprintf("results/%s_size_distribution.csv", baseFilename))
    if err != nil {
        return fmt.Errorf("failed to create size distribution file: %v", err)
    }
    defer sizeFile.Close()

    // Write size distribution header
    sizeFile.WriteString("Size,Count\n")

    // Write size distribution data
    for k, itemsets := range am.frequentSets {
        sizeFile.WriteString(fmt.Sprintf("%d,%d\n", k, len(itemsets)))
    }

    // Create support distribution file
    supportFile, err := os.Create(fmt.Sprintf("results/%s_support_distribution.csv", baseFilename))
    if err != nil {
        return fmt.Errorf("failed to create support distribution file: %v", err)
    }
    defer supportFile.Close()

    // Write support distribution header
    supportFile.WriteString("ItemsetSize,Items,Support\n")

    // Write support distribution data
    for k, itemsets := range am.frequentSets {
        for _, itemset := range itemsets {
            items := strings.Join(sortedItems(itemset), ",")
            support := am.calculateSupport(itemset)
            supportFile.WriteString(fmt.Sprintf("%d,\"%s\",%f\n", k, items, support))
        }
    }

    // Create performance metrics file
    perfFile, err := os.Create(fmt.Sprintf("results/%s_performance.csv", baseFilename))
    if err != nil {
        return fmt.Errorf("failed to create performance file: %v", err)
    }
    defer perfFile.Close()

    // Write performance metrics header
    perfFile.WriteString("Metric,Time(seconds)\n")
    
    // Write timing metrics
    perfFile.WriteString(fmt.Sprintf("Data Loading,%f\n", metrics.DataLoadTime))
    perfFile.WriteString(fmt.Sprintf("Processing,%f\n", metrics.ProcessingTime))
    perfFile.WriteString(fmt.Sprintf("Total,%f\n", metrics.TotalTime))
    
    // Write additional performance metrics
    perfFile.WriteString(fmt.Sprintf("Total Transactions,%d\n", am.transactionLen))
    perfFile.WriteString(fmt.Sprintf("Total Frequent Itemsets,%d\n", am.getTotalFrequentItemsets()))

    return nil
}

// getTotalFrequentItemsets returns the total number of frequent itemsets found
func (am *AprioriMiner) getTotalFrequentItemsets() int {
    total := 0
    for _, itemsets := range am.frequentSets {
        total += len(itemsets)
    }
    return total
}

// LoadDataset loads transactions from a file
func LoadDataset(filename string) (Dataset, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var dataset Dataset
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		items := strings.Fields(line)
		dataset = append(dataset, items)
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return dataset, nil
}