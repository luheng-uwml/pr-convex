package optimization;

import java.util.Random;

import inference.SequentialInference;
import data.Evaluator;
import feature.SequentialFeatures;

public class OnlineExponentiatedGradientDescent {
	SequentialFeatures features;
	SequentialInference model;
	Evaluator eval;
	int[][] labels;
	int[] trainList, devList;
	double[] parameters, empiricalCounts, expectedCounts, runningAccuracy;
	double[][][] edgeScores, edgeGradient; // pre-tag x current-tag
	double[][][] nodeScores, nodeGradient; // sentence-id x sentence-length x current-tag
	double[][][] marginalsOld;
	double[] logNorm, entropy; 
	double lambda, objective, initialStepSize;
	int numInstances, numFeatures, maxNumIterations, numStates, numTargetStates;
	Random randomGen;
	
	public OnlineExponentiatedGradientDescent(SequentialFeatures features,
			int[][] labels, int[] trainList, int[] devList,
			Evaluator eval, double lambda, double initialStepSize,
			int maxNumIterations, int randomSeed) {
		this.features = features;
		this.labels = labels;
		this.trainList = trainList;
		this.devList = devList;
		this.eval = eval;
		this.lambda = 1.0 / lambda;
		this.initialStepSize = initialStepSize;
		this.maxNumIterations = maxNumIterations;
		this.randomGen = new Random(randomSeed);
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
		edgeScores = new double[numInstances][numStates][numStates];
		edgeGradient = new double[numInstances][numStates][numStates];
		nodeScores = new double[numInstances][][];
		nodeGradient = new double[numInstances][][];
		logNorm = new double[numInstances];
		entropy = new double[numInstances];
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
		ArrayHelper.deepFill(entropy, 0.0);
		ArrayHelper.deepFill(logNorm, 0.0);
		objective = 0;
		runningAccuracy = new double[3]; // Precision, Recall, F1
		// initialize objective
		marginalsOld = null;
		for (int i : trainList) {
			OptimizationHelper.computeHardCounts(features, labels, i,
					empiricalCounts);
		}
		ArrayHelper.deepCopy(empiricalCounts, parameters);
		for (int instanceID : trainList) {
			int length = features.getInstanceLength(instanceID);
			double[][][] edgeMarginals =
					new double[length + 1][numStates][numStates];
			logNorm[instanceID] = model.computeMarginals(nodeScores[instanceID],
					edgeScores[instanceID], null, edgeMarginals);
			entropy[instanceID] = model.computeEntropy(nodeScores[instanceID],
					edgeScores[instanceID], edgeMarginals, logNorm[instanceID]);
			OptimizationHelper.computeSoftCounts(features, instanceID,
						edgeMarginals, parameters, -1.0);
			objective -= entropy[instanceID];
		}
		objective += 0.5 * lambda * ArrayHelper.l2NormSquared(parameters);
		System.out.println("initial objective::\t" + objective);
	}
	
	public void optimize() {
		double stepSize = initialStepSize;
		double prevObjective = Double.POSITIVE_INFINITY;
		for (int iteration = 0; iteration < maxNumIterations; iteration ++) {
			for (int k = 0; k < trainList.length; k++) {
				int instanceID = trainList[randomGen.nextInt(trainList.length)];
				backupMarginals(instanceID);
				updateGradient(instanceID);
				updateDualParameters(instanceID, stepSize);
				updateObjective(instanceID);
			}
			System.out.println("ITER::\t" + iteration + "\tSTEP:\t" + stepSize +
					"\tOBJ::\t" + objective + "\tPREV::\t" + prevObjective +
					"\tPARA::\t" + ArrayHelper.l2NormSquared(parameters));
			
			prevObjective = objective;
			stepSize = Math.max(initialStepSize / (iteration + 1), 1e-5);
			/*
			if (Math.abs((objective - prevObjective) / prevObjective) < 1e-5) {
				System.out.println("succeed!");
				break;
			}
			*/
		}
	}
	
	private void backupMarginals(int instanceID) {
		int length = features.getInstanceLength(instanceID);
		marginalsOld = new double[length + 1][numStates][numStates];
		model.computeMarginals(nodeScores[instanceID], edgeScores[instanceID],
				null, marginalsOld);
	}
	
	private void updateDualParameters(int instanceID, double stepSize) {
		int length = features.getInstanceLength(instanceID);
		for (int i = 0; i < numStates; i++) {
			for (int j = 0; j < numStates; j++) {
				edgeScores[instanceID][i][j] -= stepSize *
						edgeGradient[instanceID][i][j];
			}
		}
		for (int i = 0; i < length; i++) {
			for (int j = 0; j < numTargetStates; j++) {
				nodeScores[instanceID][i][j] -= stepSize *
						nodeGradient[instanceID][i][j];
			}
		}
	}
	
	private void updateObjective(int instanceID) {
		int length = features.getInstanceLength(instanceID);
		double[][][] edgeMarginals =
				new double[length + 1][numStates][numStates];
		double entropyOld = entropy[instanceID]; 
		logNorm[instanceID] = model.computeMarginals(nodeScores[instanceID],
				edgeScores[instanceID], null, edgeMarginals);
		entropy[instanceID] = model.computeEntropy(nodeScores[instanceID],
				edgeScores[instanceID], edgeMarginals, logNorm[instanceID]);
		// update objective
		objective += entropyOld  - entropy[instanceID];
		objective -= 0.5 * lambda * ArrayHelper.l2NormSquared(parameters);
		OptimizationHelper.computeSoftCounts(features, instanceID,
					marginalsOld, parameters);
		OptimizationHelper.computeSoftCounts(features, instanceID,
					edgeMarginals, parameters, -1.0);
		objective += 0.5 * lambda * ArrayHelper.l2NormSquared(parameters);
	}
	
	private void updateGradient(int instanceID) {
		for (int i = 0; i < numStates; i++) {
			for (int j = 0; j < numStates; j++) {
				edgeGradient[instanceID][i][j] = edgeScores[instanceID][i][j] -
						lambda * features.computeEdgeScore(i, j, parameters);
			}
		}
		int length = features.getInstanceLength(instanceID);
		for (int i = 0; i < length; i++) {
			for (int j = 0; j < numTargetStates; j++) {
				nodeGradient[instanceID][i][j] =
					nodeScores[instanceID][i][j] - lambda *
					features.computeNodeScore(instanceID, i, j, parameters);
			}
		}
	}
}
