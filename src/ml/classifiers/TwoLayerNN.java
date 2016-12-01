package ml.classifiers;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

import ml.data.DataSet;
import ml.data.Example;

public class TwoLayerNN implements Classifier {
	protected int numHiddenNodes;
	private double eta = 0.1; 
	private int numIterations = 200;
	private double[][] hiddenWeights;
	private double[] outputWeights;
	private DataSet data;
	private DataSet trainData;
	private DataSet testData;
	
	public TwoLayerNN(int numHiddenNodes) {
		this.numHiddenNodes = numHiddenNodes;
		this.outputWeights = new double[numHiddenNodes+1];
		Random r = new Random();
		for (int i = 0; i < this.outputWeights.length; i++)
			this.outputWeights[i] = -0.1 + 0.2 * r.nextDouble();
	}

	public void train(DataSet data) {
		this.data = data.getCopyWithBias();
		ArrayList<Example> examples = this.data.getData();
		this.hiddenWeights = new double[this.numHiddenNodes][this.data.getAllFeatureIndices().size()];
		Random r = new Random();
		for (int i = 0; i < this.hiddenWeights.length; i++)
			for (int j = 0; j < this.hiddenWeights[0].length; j++)
				this.hiddenWeights[i][j] = -0.1 + 0.2 * r.nextDouble();		
		
		for (int it = 0; it < this.numIterations; it++) {
			Collections.shuffle(examples);
			
			double error = 0;
			
			for (Example e : examples) {
				double[] h = new double[this.numHiddenNodes+1];
				h[this.numHiddenNodes] = 1; // hidden layer bias
				for (int i = 0; i < this.numHiddenNodes; i++) {
					h[i] = 0;
					for (int f : e.getFeatureSet())
						h[i] += e.getFeature(f) * this.hiddenWeights[i][f];
					h[i] = Math.tanh(h[i]);
				}
				
				double output = 0;
				for (int i = 0; i < this.numHiddenNodes+1; i++)
					output += h[i] * this.outputWeights[i];
				double vh = output;	
				output = Math.tanh(vh);
				
				error += Math.pow(output - e.getLabel(), 2) / examples.size();
				
				double Dout = Math.pow(1/Math.cosh(vh), 2) * (e.getLabel() - output);
				for (int i = 0; i < this.numHiddenNodes+1; i++) {
					this.outputWeights[i] += this.eta * h[i] * Dout;
				
					if (i == this.numHiddenNodes) break; // no edges from input to hidden layer bias
					
					double wx = 0;
					for (int f : e.getFeatureSet())
						wx += e.getFeature(f) * this.hiddenWeights[i][f];
				
					double Dk = Math.pow(1/Math.cosh(wx), 2) * this.outputWeights[i] * Dout;
					for (int f : this.data.getAllFeatureIndices())
						this.hiddenWeights[i][f] += this.eta * e.getFeature(f) * Dk;
				}
			}
			System.out.println("Iter: " + (it+1) + ", ss error: " + error);
			
			
		}
	}
	
	public void newTrain(DataSet train, DataSet test) {
		this.trainData = train.getCopyWithBias();
		this.testData = test.getCopyWithBias();
		ArrayList<Example> trainExamples = this.trainData.getData();
		ArrayList<Example> testExamples = this.testData.getData();
		
		this.hiddenWeights = new double[this.numHiddenNodes][this.testData.getAllFeatureIndices().size()];
		Random r = new Random();
		for (int i = 0; i < this.hiddenWeights.length; i++)
			for (int j = 0; j < this.hiddenWeights[0].length; j++)
				this.hiddenWeights[i][j] = -0.1 + 0.2 * r.nextDouble();		
		
		for (int it = 0; it < this.numIterations; it++) {
			Collections.shuffle(trainExamples);
			Collections.shuffle(testExamples);
			
			double trainError = 0, trainCorrect = 0, testCorrect = 0, totalTrain = 0, totalTest = 0;
			
			for (Example e : trainExamples) {
				double[] h = new double[this.numHiddenNodes+1];
				h[this.numHiddenNodes] = 1; // hidden layer bias
				for (int i = 0; i < this.numHiddenNodes; i++) {
					h[i] = 0;
					for (int f : e.getFeatureSet())
						h[i] += e.getFeature(f) * this.hiddenWeights[i][f];
					h[i] = Math.tanh(h[i]);
				}
				
				double output = 0;
				for (int i = 0; i < this.numHiddenNodes+1; i++)
					output += h[i] * this.outputWeights[i];
				double vh = output;
				output = Math.tanh(vh);
				
				trainError += Math.pow(output - e.getLabel(), 2) / trainExamples.size();
								
				double Dout = Math.pow(1/Math.cosh(vh), 2) * (e.getLabel() - output);
				for (int i = 0; i < this.numHiddenNodes+1; i++) {
					this.outputWeights[i] += this.eta * h[i] * Dout;
				
					if (i == this.numHiddenNodes) break; // no edges from input to hidden layer bias
					
					double wx = 0;
					for (int f : e.getFeatureSet())
						wx += e.getFeature(f) * this.hiddenWeights[i][f];
				
					double Dk = Math.pow(1/Math.cosh(wx), 2) * this.outputWeights[i] * Dout;
					for (int f : this.trainData.getAllFeatureIndices())
						this.hiddenWeights[i][f] += this.eta * e.getFeature(f) * Dk;
				}
			}
			
			for (Example e: trainExamples) {
				if (classify(e) == e.getLabel()) trainCorrect++;
				totalTrain++;
			}
			
			for (Example e: testExamples) {
				if (classify(e) == e.getLabel()) testCorrect++;
				totalTest++;
				
			}
			
			System.out.println( (it+1) + "," + trainError + "," + trainCorrect / totalTrain + "," + testCorrect/totalTest);
			
		}
	}

	public double classify(Example example) {
		return this.calculateOutput(example) > 0 ? 1 : -1;
	}

	public double confidence(Example example) {
		return Math.abs(this.calculateOutput(example));
	}
	
	private double calculateOutput(Example example) {
		// TODO: Uncomment this after experimenting
//		example = this.trainData.addBiasFeature(example);
		double[] h = new double[this.numHiddenNodes+1];
		h[this.numHiddenNodes] = 1; // hidden layer bias
		for (int i = 0; i < this.numHiddenNodes; i++) {
			h[i] = 0;
			for (int f : example.getFeatureSet())
				h[i] += example.getFeature(f) * this.hiddenWeights[i][f];
			h[i] = Math.tanh(h[i]);
		}
		
		double output = 0;
		for (int i = 0; i < this.numHiddenNodes+1; i++)
			output += h[i] * this.outputWeights[i];
		double vh = output;
		output = Math.tanh(vh);
		
		return output;
	}
	
	public void setEta(double newEta) {
		this.eta = newEta;
	}
	
	public void setIterations(int numIterations) {
		this.numIterations = numIterations;
	}

}
