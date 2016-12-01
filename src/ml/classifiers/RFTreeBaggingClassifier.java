package ml.classifiers;

import java.util.ArrayList;
import ml.classifiers.DecisionTreeClassifier;
import ml.data.DataSet;
import ml.data.Example;
import ml.utils.HashMapCounterDouble;

/**
 * 
 * @author Antony Bello, Nick Reminder, Dima Smirnov
 * 
 * Random Forest implementation using bagging. Trains N decision
 * trees on random subsets of the data, and uses majority vote
 * to classify a given example.
 *
 */
public class RFTreeBaggingClassifier implements Classifier {
	private ArrayList<DecisionTreeClassifier> trees;
	private int numTreesInTheForest = 20;
	private int depthLimit = 5;

	/**
	 * Trains our random forest by training each decision tree
	 * on random subsets of a dataset.
	 * 
	 * @param data data
	 */
	public void train(DataSet data) {
		trees = new ArrayList<DecisionTreeClassifier>();
		for (int i = 0; i < this.numTreesInTheForest; i++) {
			DataSet subset = data.createDatasetWithRandomExampleSubset();
			DecisionTreeClassifier d = new DecisionTreeClassifier();
			d.setDepthLimit(depthLimit);
			d.train(subset);
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
		HashMapCounterDouble<Double> counter = new HashMapCounterDouble<Double>();
		for (DecisionTreeClassifier d : this.trees) {
			double prediction = d.classify(example);
			counter.increment(prediction, 1);
		}	
		return counter.sortedEntrySet().get(0).getKey();
	}
	
	public double confidence(Example example) {
		return 0;
	}

	public void setNumTrees(int numTrees) {
		this.numTreesInTheForest = numTrees;
	}
}
