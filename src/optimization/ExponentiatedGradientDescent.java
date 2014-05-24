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
	double[] parameters, empiricalCounts, runningAccuracy;
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
		empiricalCounts = new double[numFeatures];
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
		ArrayHelper.deepFill(edgeScores, 0.0);
		ArrayHelper.deepFill(edgeGradient, 0.0);
		for (int i : trainList) {
			computeHardCounts(i, empiricalCounts);
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
					
			System.out.println("ITER::\t" + iteration + "\tOBJ::\t" +
					objective + "\tPREV::\t" + prevObjective +
					"\tPREC::\t" + precision + "\tREC::\t" + recall +
					"\tF1::\t" + f1);
			
			if (Math.abs(objective - prevObjective) / prevObjective < 1e-5) {
				System.out.println("succeed!");
				break;
			}
			//double currStepSize = this.backtrackingLineSearch(stepSize, 0.5,
			//		0.5, 20);
			//updateParameters(stepSize / Math.sqrt(iteration + 1));
			updateParameters(stepSize);
			
			prevObjective = objective;
			//stepSize = currStepSize * 1.5;
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
		double negEntropy = 0;
		for (int i = 0; i < numFeatures; i++) {
			parameters[i] = - empiricalCounts[i];
		}
		// marginalize and compute entropy for each training instance
		// update primal parameters \theta = E[f] - \tilde{f}
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
			computeSoftCounts(i, edgeMarginals, parameters);
			negEntropy -= entropy;
			// compute accuracy on training instance
			model.posteriorDecoding(nodeMarginals, decoded);
			int[] result = eval.evaluate(labels[i], decoded);
			runningAccuracy[0] += result[0];
			runningAccuracy[1] += result[1];
			runningAccuracy[2] += result[2];
		}
		// compute objective
		objective = negEntropy +
				0.5 * lambda * ArrayHelper.l2NormSquared(parameters);
		// update dual gradient for each node and edge factor
		for (int i = 0; i < numStates; i++) {
			for (int j = 0; j < numStates; j++) {
				edgeGradient[i][j] = edgeScores[i][j] + lambda *
						features.computeEdgeScore(i, j, parameters);
			}
		}
		for (int i : trainList) {
			int length = features.getInstanceLength(i);
			for (int j = 0; j < length; j++) {
				for (int k = 0; k < numTargetStates; k++) {
					nodeGradient[i][j][k] = nodeScores[i][j][k] + lambda *
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
	
	private void computeHardCounts(int instanceID, double[] counts) {
		int length = features.getInstanceLength(instanceID);
		for (int i = 0; i < length; i++) {
			int s = labels[instanceID][i];
			int sp = (i == 0) ? model.S0 : labels[instanceID][i - 1];
			features.addToCounts(instanceID, i, s, sp, counts, 1.0);
		}
	}
	
	private void computeSoftCounts(int instanceID, double[][][] edgeMarginals,
			double[] counts) {
		int length = features.getInstanceLength(instanceID);
		for (int s = 0; s < numTargetStates; s++) {
			features.addToCounts(instanceID, 0, s, model.S0, counts,
					edgeMarginals[0][s][model.S0]);
			features.addToCounts(instanceID, length, model.SN, s, counts,
					edgeMarginals[length][model.SN][s]);
		}
		for (int i = 1; i < length; i++) {
			for (int s = 0; s < numTargetStates; s++) {
				for (int sp = 0; sp < numTargetStates; sp++) {
					features.addToCounts(instanceID, i, s, sp, counts,
							edgeMarginals[i][s][sp]);
				}
			}
		}
	}
}
