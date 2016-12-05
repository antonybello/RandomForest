package ml.classifiers;

import java.util.ArrayList;
import ml.classifiers.DecisionTreeClassifier;
import ml.data.DataSet;
import ml.data.Example;
import ml.utils.HashMapCounter;

/**
 * 
 * @author Antony Bello, Nick Reminder, Dima Smirnov
 * 
 * Random Forest implementation using bagging. Trains N decision
 * trees on random subsets of the data, and uses majority vote
 * to classify a given example.
 *
 */
public class EnsembleTreeClassifier implements Classifier {
	private ArrayList<DecisionTreeClassifier> trees;
	private int numTrees = 20;
	private int depthLimit = 5;
	private boolean featureBagging = false;
	private boolean extraTrees = false;

	/**
	 * Trains our random forest by training each decision tree
	 * on random subsets of a dataset.
	 * 
	 * @param data data
	 */
	public void train(DataSet data) {
		this.trees = new ArrayList<DecisionTreeClassifier>();
		for (int i = 0; i < this.numTrees; i++) {
			DecisionTreeClassifier d = new DecisionTreeClassifier();
			d.setDepthLimit(this.depthLimit);
			d.setSplitRandomly(extraTrees);
			DataSet newData = featureBagging ? data.createDatasetWithFeatureBagging() : data.createDatasetWithBagging();
			d.train(newData);
			this.trees.add(d);
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
		for (DecisionTreeClassifier d : this.trees)
			counter.increment(d.classify(example));
		return counter.sortedEntrySet().get(0).getKey();
	}
	
	public double confidence(Example example) {
		return 0;
	}

	public void setNumTrees(int numTrees) {
		this.numTrees = numTrees;
	}
	
	public void setDepthLimit(int newDepth) {
		this.depthLimit = newDepth;
	}

	public void setFeatureBagging(boolean featureBagging) {
		this.featureBagging = featureBagging;
	}
	
	public void setExtraTrees(boolean extraTrees) {
		this.extraTrees = extraTrees;
	}
}
