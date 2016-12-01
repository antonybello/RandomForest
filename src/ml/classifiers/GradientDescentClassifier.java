/**
 * GradientDescentClassifier.java
 * Dima Smirnov, Antony Bello
 * CS158 - Machine Learning 
 */

package ml.classifiers;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Set;
import java.util.Random;

import ml.data.DataSet;
import ml.data.Example;

/**
 * Gradient descent classifier allowing for two different loss functions and
 * three different regularization settings.
 * 
 * @author Dima Smirnov, Antony Bello
 *
 */
public class GradientDescentClassifier implements Classifier {
	// constants for the different surrogate loss functions
	public static final int EXPONENTIAL_LOSS = 0;
	public static final int HINGE_LOSS = 1;

	// constants for the different regularization parameters
	public static final int NO_REGULARIZATION = 0;
	public static final int L1_REGULARIZATION = 1;
	public static final int L2_REGULARIZATION = 2;

	// initial lambda and eta
	public static final double INITIAL_LAMBDA = 0.1;
	public static final double INITIAL_ETA = 0.1;

	protected int iterations = 10;

	protected int regularizationType;
	protected int lossType;
	protected double lambda;
	protected double eta;

	protected HashMap<Integer, Double> weights; // the feature weights
	protected double b = 0; // the intersect weight

	public GradientDescentClassifier() {
		this.lossType = HINGE_LOSS;
		this.regularizationType = NO_REGULARIZATION;
		this.lambda = INITIAL_LAMBDA;
		this.eta = INITIAL_ETA;
	}

	/**
	 * Get a weight vector over the set of features with each weight set to 0
	 * 
	 * @param features
	 *            the set of features to learn over
	 * @return
	 */
	protected HashMap<Integer, Double> getZeroWeights(Set<Integer> features) {
		HashMap<Integer, Double> temp = new HashMap<Integer, Double>();

		for (Integer f : features) {
			temp.put(f, 0.0);
		}

		return temp;
	}

	/**
	 * Initialize the weights and the intersect value
	 * 
	 * @param features
	 */
	protected void initializeWeights(Set<Integer> features) {
		weights = getZeroWeights(features);
		b = 0;
	}

	/**
	 * Set the number of iterations the classifier should run during training
	 * 
	 * @param iterations
	 */
	public void setIterations(int iterations) {
		this.iterations = iterations;
	}

	/**
	 * Trains the GD classifier by computing the weight vector and bias that 
	 * minimize a given surrogate loss function. Iterates over the examples,
	 * starting with a weight vector of 0, and, given each example's distance
	 * from the hyperplane, adjusts the weight vector in a direction that aims
	 * to minimize the loss.
	 * 
	 * @param data 
	 * 
	 */
	public void train(DataSet data) {

		initializeWeights(data.getAllFeatureIndices());
		ArrayList<Example> training = (ArrayList<Example>) data.getData().clone();

		double label, distancePrediction, loss;
		double oldWeight, featureValue, regularization, update, bReg;

		for (int it = 0; it < iterations; it++) {
			Collections.shuffle(training);

			for (Example e : training) {
				label = e.getLabel();

				// Calculate model's prediction for the example (y')
				distancePrediction = getDistanceFromHyperplane(e, weights, b);

				// Calculate the loss for our prediction (c)
				loss = getLossByType(label, distancePrediction);

				for (Integer featureIndex : e.getFeatureSet()) {
					oldWeight = weights.get(featureIndex);
					featureValue = e.getFeature(featureIndex);

					// Calculate the regularization for this weight
					regularization = getRegularizationByType(oldWeight);

					// Calculate the update to our weight
					update = (eta * ((label * featureValue * loss) - regularization));

					weights.put(featureIndex, oldWeight + update);
				}

				// Treat our update for b as a feature with value of 1
				bReg = getRegularizationByType(b);
				b += (eta * label * loss - bReg);
			}
		}
	}

	@Override
	public double classify(Example example) {
		return getPrediction(example);
	}

	@Override
	public double confidence(Example example) {
		return Math.abs(getDistanceFromHyperplane(example, weights, b));
	}

	/**
	 * Get the prediction from the current set of weights on this example
	 * 
	 * @param e
	 *            the example to predict
	 * @return
	 */
	protected double getPrediction(Example e) {
		return getPrediction(e, weights, b);
	}

	/**
	 * Get the prediction from the on this example from using weights w and
	 * inputB
	 * 
	 * @param e
	 *            example to predict
	 * @param w
	 *            the set of weights to use
	 * @param inputB
	 *            the b value to use
	 * @return the prediction
	 */
	protected static double getPrediction(Example e,
			HashMap<Integer, Double> w, double inputB) {
		double sum = getDistanceFromHyperplane(e, w, inputB);

		if (sum > 0) {
			return 1.0;
		} else if (sum < 0) {
			return -1.0;
		} else {
			return 0;
		}
	}

	protected static double getDistanceFromHyperplane(Example e,
			HashMap<Integer, Double> w, double inputB) {
		double sum = inputB;

		for (Integer featureIndex : w.keySet()) {
			sum += w.get(featureIndex) * e.getFeature(featureIndex);
		}

		return sum;
	}

	/**
	 * Based on what type of loss is set, calculate the loss (c) by the
	 * exponential or hinge loss formula
	 * 
	 * @param label
	 *            Example's label
	 * @param distancePrediction
	 *            Example's distance from hyperplane
	 * @return loss
	 */
	public double getLossByType(double label, double distancePrediction) {
		return this.lossType == EXPONENTIAL_LOSS ? Math.exp(-1 * label
				* distancePrediction) : Math.max(0,
				1 - (label * distancePrediction));
	}

	/**
	 * Based on what type of regularization is set, calculate the regularization
	 * for given weight or bias term
	 * 
	 * @param d
	 *            weight or bias
	 * @return L1, L2, or no regularization on parameter
	 */
	public double getRegularizationByType(double d) {
		double sign = Math.signum(d);
		switch (this.regularizationType) {
		case L1_REGULARIZATION:
			return this.lambda * sign;
		case L2_REGULARIZATION:
			return this.lambda * d;
		default:
			return 0.0;
		}
	}

	public void setLoss(int newLoss) {
		lossType = newLoss;
	}

	public void setRegularization(int newReg) {
		regularizationType = newReg;
	}

	public void setLambda(double newLamb) {
		lambda = newLamb;
	}

	public void setEta(double newEta) {
		eta = newEta;
	}

	public String toString() {
		StringBuffer buffer = new StringBuffer();

		ArrayList<Integer> temp = new ArrayList<Integer>(weights.keySet());
		Collections.sort(temp);

		for (Integer index : temp) {
			buffer.append(index + ":" + weights.get(index) + " ");
		}

		return buffer.substring(0, buffer.length() - 1);
	}
}
