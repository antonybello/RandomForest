package ml.data;
import ml.classifiers.*;

/**
 * Random Forest classifier experiments. 
 * 
 * @author Antony Bello, Dima Smirnov, Nicholas Reminder
 *
 */
public class Experimenter {
	public static void main(String[] args) {
		DataSetSplit splits;

		for (DataSet dataset : new DataSet[] {new DataSet("wines.train", DataSet.TEXTFILE), new DataSet("titanic-train.csv", DataSet.CSVFILE)}) {
			System.out.println("num trees  depth  test acc  train acc  tb test acc  tb train acc  fb test acc  fb train acc  et test acc  et train acc");
			for (int depth = 1; depth <= 5; depth++) {
				DecisionTreeClassifier d = new DecisionTreeClassifier();
				d.setDepthLimit(depth);
				splits = dataset.split(0.8);
				double[] accuracies = calcAccuracy(dataset, d, splits);
				
				for (int numTrees = 10; numTrees <= 50; numTrees += 10) {
					EnsembleTreeClassifier etc = new EnsembleTreeClassifier();
					etc.setDepthLimit(depth);
					etc.setNumTrees(numTrees);

					double[] tbAccuracies = calcAccuracy(dataset, etc, splits);
					etc = new EnsembleTreeClassifier();
					etc.setDepthLimit(depth);
					etc.setNumTrees(numTrees);
					etc.setFeatureBagging(true);
					double[] fbAccuracies = calcAccuracy(dataset, etc, splits);
					etc = new EnsembleTreeClassifier();
					etc.setDepthLimit(depth);
					etc.setNumTrees(numTrees);
					etc.setExtraTrees(true);
					double[] etAccuracies = calcAccuracy(dataset, etc, splits);
					
					System.out.format("%-11d%-7d%-10.4f%-11.4f%-13.4f%-14.4f%-13.4f%-14.4f%-13.4f%-14.4f%n", numTrees, depth, accuracies[0], accuracies[1], tbAccuracies[0], tbAccuracies[1], fbAccuracies[0], fbAccuracies[1], etAccuracies[0], etAccuracies[1]);
				}
			}
			System.out.println("-----------------------------");
		}
	}


	/**
	 * Splits a dataset, trains a classifier on the test data, and computes accuracies
	 * on both the training and test data.
	 * 
	 * @param data dataset being evaluated
	 * @param classifier classifier being used
	 * @param depth depth limit for classifier 
	 * @param split proportion of split for test/training data
	 * @return array with two doubles, the test data accuracy and training data accuracy
	 */
	private static double[] calcAccuracy(DataSet data, Classifier classifier, DataSetSplit splits) {
		classifier.train(splits.getTrain());
		double correctTest = 0, totalTest = 0;
		for (Example e : splits.getTest().getData()) {
			if (classifier.classify(e) == e.getLabel()) correctTest++;
			totalTest++;
		}
		double correctTrain = 0, totalTrain = 0;
		for (Example e : splits.getTrain().getData()) {
			if (classifier.classify(e) == e.getLabel()) correctTrain++;
			totalTrain++;
		}

		return new double[] {correctTest / totalTest, correctTrain / totalTrain};
	}
}