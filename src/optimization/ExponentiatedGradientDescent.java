package optimization;

import inference.SequentialInference;
import data.Evaluator;
import feature.SequentialFeatures;

public class ExponentiatedGradientDescent {
	SequentialFeatures features;
	SequentialInference model;
	Evaluator eval;
	int[][] labels;
	int[] trainList, devList;
	double[] parameters, empiricalCounts, gradient, runningAccuracy;
	double[][] edgeScores; // pre-tag x current-tag
	double[][][] nodeScores; // sentence-id x sentence-length x current-tag
	double[] logNorms; 
	double lambda, objective, initialStepSize, stepSize;
	int numInstances, numFeatures, maxNumIterations, numStates, numTargetStates;
	
	public ExponentiatedGradientDescent(SequentialFeatures features,
			int[][] labels, int[] trainList, int[] devList,
			Evaluator eval, double lambda, double initialStepSize,
			int maxNumIterations) {
		this.features = features;
		this.labels = labels;
		this.trainList = trainList;
		this.devList = devList;
		this.eval = eval;
		this.lambda = lambda;
		this.initialStepSize = initialStepSize;
		this.maxNumIterations = maxNumIterations;
		initialize();
	}
	
	private void initialize() {
		numInstances = features.getNumInstances();
		numFeatures = features.getNumFeatures();
		numStates = features.getNumStates();
		numTargetStates = numStates - 2;
		model= new SequentialInference(1000, numStates);
		parameters = new double[numFeatures];
		gradient = new double[numFeatures];
		empiricalCounts = new double[numFeatures];
		edgeScores = new double[numStates][numStates];
		nodeScores = new double[numInstances][][];
		for (int i : trainList) {
			//computeHardCounts(i, empiricalCounts);
			nodeScores[i] =
				new double[features.getInstanceLength(i)][numTargetStates];
		}
		ArrayHelper.deepFill(parameters, 0.0);
		ArrayHelper.deepFill(empiricalCounts, 0.0);
		ArrayHelper.deepFill(edgeScores, 0.0);
		ArrayHelper.deepFill(nodeScores, 0.0);
		runningAccuracy = new double[3]; // Precision, Recall, F1
	}
}
