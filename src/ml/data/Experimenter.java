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
//		DataSet dataset = new DataSet("wines.train", DataSet.TEXTFILE);
		DataSet dataset = new DataSet("titanic-train.csv", DataSet.CSVFILE);

		DecisionTreeClassifier d = new DecisionTreeClassifier();
		d.setDepthLimit(5);
		System.out.println("Decision Tree");
		System.out.println("test acc train acc");
		DataSetSplit splits = dataset.split(0.8);
		double[] accuracies = calcAccuracy(dataset, d, splits);
		System.out.format("%-10.4f%.4f%n", accuracies[0], accuracies[1]);
		
		EnsembleTreeClassifier etc = new EnsembleTreeClassifier();
		System.out.println("Tree Bagging");
		System.out.println("test acc train acc");
		accuracies = calcAccuracy(dataset, etc, splits);
		System.out.format("%-10.4f%.4f%n", accuracies[0], accuracies[1]);	
		
		etc.setFeatureBagging(true);
		System.out.println("Feature Bagging");
		System.out.println("depth test acc train acc");
		accuracies = calcAccuracy(dataset, etc, splits);
		System.out.format("%-10.4f%.4f%n", accuracies[0], accuracies[1]);
		
		etc.setExtraTrees(true);
		System.out.println("ExtraTrees");
		System.out.println("depth test acc train acc");
		accuracies = calcAccuracy(dataset, etc, splits);
		System.out.format("%-10.4f%.4f%n", accuracies[0], accuracies[1]);
	}

	/**
	 * Computes n-fold validation for a given classifier
	 * @param cvs cross validation set
	 * @param classifier classifier to train and classify an example
	 */
	private static void computeFolds(CrossValidationSet cvs, Classifier classifier) {
		double trainAcc = 0, testAcc = 0;
		for (int s = 0; s < cvs.getNumSplits(); s++) {
			DataSet train = cvs.getValidationSet(s).getTrain();
			classifier.train(train);
			
			int correct = 0;
			for (Example e : cvs.getValidationSet(s).getTrain().getData()) {
				if (e.getLabel() == classifier.classify(e)) correct++;
			}
			trainAcc += correct * 1.0 / cvs.getValidationSet(s).getTrain().getData().size();

			correct = 0;
			for (Example e : cvs.getValidationSet(s).getTest().getData()) {
				if (e.getLabel() == classifier.classify(e)) correct++;
			}
			testAcc += correct * 1.0 / cvs.getValidationSet(s).getTest().getData().size();
		}
		System.out.format("%.4f %.4f%n", trainAcc / cvs.getNumSplits(), testAcc / cvs.getNumSplits());
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
	
	private static double calcAccuracyWithMajority(DataSet data, DecisionTreeClassifier classifier) {
		classifier.train(data);
		double majorityLabel = classifier.getMajorityLabel(data.getData()).majorityLabel;
		double correct = 0, total = 0;
		for (Example e : data.getData()) {
			if (classifier.classify(e) == majorityLabel) correct++;
			total++;
		}
		return correct / total; 
	}

}
