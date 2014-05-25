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
	double[] parameters, empiricalCounts, expectedCounts, runningAccuracy;
	double[][] edgeScores, edgeGradient; // pre-tag x current-tag
	double[][][] nodeScores, nodeGradient; // sentence-id x sentence-length x current-tag
	double[] logNorms; 
	double lambda, objective, initialStepSize;
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
		this.lambda = 1.0 / lambda;
		this.initialStepSize = initialStepSize;
		this.maxNumIterations = maxNumIterations;
		initialize();
	}
	
	private void initialize() {
		numInstances = features.numInstances;
		numFeatures = features.numAllFeatures;
		numStates = features.numStates;
		numTargetStates = features.numTargetStates;
		model= new SequentialInference(1000, numStates);
		parameters = new double[numFeatures];
		empiricalCounts = new double[numFeatures];
		expectedCounts = new double[numFeatures];
		edgeScores = new double[numStates][numStates];
		edgeGradient = new double[numStates][numStates];
		nodeScores = new double[numInstances][][];
		nodeGradient = new double[numInstances][][];
		for (int i : trainList) {
			int length = features.getInstanceLength(i);
			nodeScores[i] = new double[length][numTargetStates];
			nodeGradient[i] = new double[length][numTargetStates];
			ArrayHelper.deepFill(nodeScores[i], 0.0);
			ArrayHelper.deepFill(nodeGradient[i], 0.0);
		}
		ArrayHelper.deepFill(parameters, 0.0);
		ArrayHelper.deepFill(empiricalCounts, 0.0);
		ArrayHelper.deepFill(expectedCounts, 0.0);
		ArrayHelper.deepFill(edgeScores, 0.0);
		ArrayHelper.deepFill(edgeGradient, 0.0);
		for (int i : trainList) {
			OptimizationHelper.computeHardCounts(features, labels, i,
					empiricalCounts);
		}
		runningAccuracy = new double[3]; // Precision, Recall, F1
	}
	
	public void optimize() {
		double stepSize = initialStepSize;
		double prevObjective = Double.POSITIVE_INFINITY;
		for (int iteration = 0; iteration < maxNumIterations; iteration ++) {
			updateObjectiveAndGradient();
			// compute accuracy
			double precision = runningAccuracy[2] / runningAccuracy[1];
			double recall = runningAccuracy[2] / runningAccuracy[0];
			double f1 = (precision + recall > 0) ?
					(2 * precision * recall) / (precision + recall) : 0.0;
					
			System.out.println("ITER::\t" + iteration + "\tSTEP:\t" + stepSize +
					"\tOBJ::\t" + objective + "\tPREV::\t" + prevObjective +
					"\tPARA::\t" + ArrayHelper.l2NormSquared(parameters) +
					"\tPREC::\t" + precision + "\tREC::\t" + recall +
					"\tF1::\t" + f1);
			/*
			if (Math.abs((objective - prevObjective) / prevObjective) < 1e-5) {
				System.out.println("succeed!");
				break;
			}
			*/
			updateParameters(stepSize);
			
			prevObjective = objective;
			stepSize = Math.max(initialStepSize / (iteration + 1), 1e-5);
			/*
			if (iteration % 100 == 99) {
				testModel();
			}
			*/
		}
	}
	
	// gradient = u_{ir} + \lambda * f(x_i, r) x (E[f] - \tilde{f})
	private void updateObjectiveAndGradient() {
		ArrayHelper.deepFill(runningAccuracy, 0.0);
		ArrayHelper.deepFill(expectedCounts, 0.0);
		double totalEntropy = 0;
		// marginalize and compute entropy for each training instance
		for (int i : trainList) {
			int length = features.getInstanceLength(i);
			double[][] nodeMarginals = new double[length][numStates];
			double[][][] edgeMarginals =
					new double[length + 1][numStates][numStates];
			int[] decoded = new int[length];
			double logNorm = model.computeMarginals(nodeScores[i], edgeScores,
					nodeMarginals, edgeMarginals);
			double entropy = model.computeEntropy(nodeScores[i], edgeScores,
					edgeMarginals, logNorm);
			OptimizationHelper.computeSoftCounts(features, i, edgeMarginals,
					expectedCounts);
			totalEntropy += entropy;
			// compute accuracy on training instance
			model.posteriorDecoding(nodeMarginals, decoded);
			int[] result = eval.evaluate(labels[i], decoded);
			runningAccuracy[0] += result[0];
			runningAccuracy[1] += result[1];
			runningAccuracy[2] += result[2];
		}
		// update primal parameters \theta = Emp[f] - E_q[f]
		for (int i = 0; i < numFeatures; i++) {
			parameters[i] = empiricalCounts[i] - expectedCounts[i];
		}
		// compute objective
		objective = - totalEntropy +
				0.5 * lambda * ArrayHelper.l2NormSquared(parameters);
		// update dual gradient for each node and edge factor
		for (int i = 0; i < numStates; i++) {
			for (int j = 0; j < numStates; j++) {
				edgeGradient[i][j] = edgeScores[i][j] - lambda *
						features.computeEdgeScore(i, j, parameters);
			}
		}
		for (int i : trainList) {
			int length = features.getInstanceLength(i);
			for (int j = 0; j < length; j++) {
				for (int k = 0; k < numTargetStates; k++) {
					nodeGradient[i][j][k] = nodeScores[i][j][k] - lambda *
							features.computeNodeScore(i, j, k, parameters);
				}
			}
		}
	}
	
	private void updateParameters(double stepSize) {
		for (int i = 0; i < numStates; i++) {
			for (int j = 0; j < numStates; j++) {
				edgeScores[i][j] -= stepSize * edgeGradient[i][j];
			}
		}
		for (int i : trainList) {
			int length = features.getInstanceLength(i);
			for (int j = 0; j < length; j++) {
				for (int k = 0; k < numTargetStates; k++) {
					nodeScores[i][j][k] -= stepSize * nodeGradient[i][j][k];
				}
			}
		}
	}
}
