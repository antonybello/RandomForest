package ml.classifiers;

import java.util.ArrayList;
import java.util.Random;

import ml.classifiers.DecisionTreeClassifier;
import ml.data.DataSet;
import ml.data.Example;
import ml.utils.HashMapCounter;

/**
 * 
 * @author Antony Bello, Nick Reminder, Dima Smirnov
 * 
 * Random Forest implementation using feature bagging. Trains N decision
 * trees on random subsets of the data, and uses majority vote
 * to classify a given example.
 *
 */
public class RFFeatureBaggingClassifier implements Classifier {
	private ArrayList<DecisionTreeClassifier> trees;
	private int treesInTheForest = 20;
	private int depthLimit = 5;
	
	/**
	 * Trains our random forest by training each decision tree
	 * on random subsets of a dataset.
	 * 
	 * @param data data
	 */
	public void train(DataSet data) {
		trees = new ArrayList<DecisionTreeClassifier>();
		
		// Set the feature subset size to be a majority of the features in the dataset
		Random rand = new Random();
		int dataSetSize = data.getAllFeatureIndices().size();
		int featureSubsetSize = rand.nextInt(dataSetSize - dataSetSize / 2) + (dataSetSize / 2);
		
		for (int i = 0; i < this.treesInTheForest; i++) {
			DataSet dataSetWithFeatureSubset = data.createDatasetWithFeatureSubset(featureSubsetSize);
			DecisionTreeClassifier d = new DecisionTreeClassifier();
			d.setDepthLimit(this.depthLimit);
			d.train(dataSetWithFeatureSubset);
			trees.add(d);
		}
	}

	/**
	 * Classifies a given example using a majority vote from our 
	 * decision trees. 
	 * 
	 * @param example 
	 */
	public double classify(Example example) {
		HashMapCounter<Double> counter = new HashMapCounter<Double>();
		for (DecisionTreeClassifier d : this.trees) {
			double prediction = d.classify(example);
			counter.increment(prediction);
		}	
		return counter.sortedEntrySet().get(0).getKey();
	}
	
	public double confidence(Example example) {
		return 0;
	}

	public void setIterations(int newTrees) {
		this.treesInTheForest = newTrees;
	}
	
	public void setDepthLimit(int newDepth) {
		this.depthLimit = newDepth;
	}
	
}
