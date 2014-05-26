package optimization;

import java.util.Arrays;
import java.util.Random;

import inference.SequentialInference;
import data.Evaluator;
import feature.SequentialFeatures;

public class SemiSupervisedExponentiatedGradientDescent {
	SequentialFeatures features;
	SequentialInference model;
	Evaluator eval;
	int[][] labels;
	int[] trainList, devList, workList;
	double[] parameters, parametersGrad, empiricalCounts, expectedCounts,
			runningAccuracy, logNorm, entropy, trainRatio;
	double[][][] edgeScores, edgeGradient; // pre-tag x current-tag
	double[][][] nodeScores, nodeGradient; // sentence-id x sentence-length x current-tag
	double[][][] marginalsOld;
	double lambda, objective, initialStepSize;
	int numTrains, numInstances, numFeatures, maxNumIterations, numStates,
		numTargetStates;
	Random randomGen;
	
	public SemiSupervisedExponentiatedGradientDescent(SequentialFeatures features,
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
		parametersGrad = new double[numFeatures];
		empiricalCounts = new double[numFeatures];
		expectedCounts = new double[numFeatures];
		edgeScores = new double[numInstances][numStates][numStates];
		edgeGradient = new double[numInstances][numStates][numStates];
		nodeScores = new double[numInstances][][];
		nodeGradient = new double[numInstances][][];
		logNorm = new double[numInstances];
		entropy = new double[numInstances];
		
		this.workList = new int[trainList.length + devList.length];
		this.numTrains = trainList.length;
		for (int i = 0; i < trainList.length; i++) {
			workList[i] = trainList[i];
		}
		for (int i = 0; i < devList.length; i++) {
			workList[i + numTrains] = devList[i];
		}
		for (int i : workList) {
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
		 
		initializeTrainRatio();
		initializeObjective();
	}
	
	private void initializeTrainRatio() {
		trainRatio = new double[numFeatures];
		int[] trainFeatureCounts = new int[numFeatures];
		int[] devFeatureCounts = new int[numFeatures];
		Arrays.fill(trainFeatureCounts, 0);
		Arrays.fill(devFeatureCounts, 0);
		features.countFeatures(trainList, trainFeatureCounts);
		features.countFeatures(devList, devFeatureCounts);
		for (int i = 0; i < numFeatures; i++) {
			if (trainFeatureCounts[i] > 0) {
				trainRatio[i] = 1.0 * trainFeatureCounts[i] /
						(trainFeatureCounts[i] + devFeatureCounts[i]);
			} else {
				trainRatio[i] = 0.0;
			}
		}
		double avgRatio = 0, nnzRatio = 0;
		for (int i = 0; i < numFeatures; i++) {
			if (trainRatio[i] != 0) {
				avgRatio += trainRatio[i];
				nnzRatio += 1;
			}
		}
		System.out.println("Averaged non-zero train ratio::\t" +
				avgRatio / nnzRatio);
	}
	
	private void initializeObjective() {
		marginalsOld = null;
		for (int instanceID : trainList) {
			OptimizationHelper.computeHardCounts(features, labels, instanceID,
					empiricalCounts);
		}
		for (int instanceID : workList) {
			int length = features.getInstanceLength(instanceID);
			double[][][] edgeMarginals =
					new double[length + 1][numStates][numStates];
			logNorm[instanceID] = model.computeMarginals(nodeScores[instanceID],
					edgeScores[instanceID], null, edgeMarginals);
			entropy[instanceID] = model.computeEntropy(nodeScores[instanceID],
					edgeScores[instanceID], edgeMarginals, logNorm[instanceID]);
			OptimizationHelper.computeSoftCounts(features, instanceID,
						edgeMarginals, expectedCounts);
			objective -= entropy[instanceID];
		}
		updatePrimalParameters();
		objective += 0.5 * lambda * ArrayHelper.l2NormSquared(parameters);
		System.out.println("initial objective::\t" + objective);
	}
	
	public void optimize() {
		double stepSize = initialStepSize;
		double prevObjective = objective;
		for (int iteration = 0; iteration < maxNumIterations; iteration ++) {
			for (int k = 0; k < workList.length; k++) {
				int instanceID = workList[randomGen.nextInt(workList.length)];
				computeGradient(instanceID);
				update(instanceID, stepSize);
			}
			System.out.println("ITER::\t" + iteration +
					"\tSTEP:\t" + stepSize +
					"\tOBJ::\t" + objective +
					"\tPREV::\t" + prevObjective +
					"\tPARA::\t" + ArrayHelper.l2NormSquared(parameters));
			
			if (iteration % 10 == 9) {
				validate(trainList);
				validate(devList);
				computeAccuracy(devList);
				computePrimalObjective();
			}
			if (objective < prevObjective) {
				if (iteration < 100) {
					stepSize *= 1.02;
				} else {
					stepSize *= 1.0 * iteration / (iteration + 1);
				}
			} else {
				stepSize *= 0.5;
			}
			prevObjective = objective;
			// TODO: stopping criterion
		}
	}
	
	private void updatePrimalParameters() {
		for (int i = 0; i < numFeatures; i++) {
			parameters[i] = empiricalCounts[i] - trainRatio[i] *
					expectedCounts[i];
			parametersGrad[i] = parameters[i] * trainRatio[i];
		}
	}
	
	private void update(int instanceID, double stepSize) {
		int length = features.getInstanceLength(instanceID);
		double[][][] tMarginals = new double[length + 1][numStates][numStates];
		model.computeMarginals(nodeScores[instanceID], edgeScores[instanceID],
				null, tMarginals);
		objective += entropy[instanceID];
		objective -= 0.5 * lambda * ArrayHelper.l2NormSquared(parameters);
		OptimizationHelper.computeSoftCounts(features, instanceID, tMarginals,
				expectedCounts, -1.0);
		// perform a gradient update
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
		logNorm[instanceID] = model.computeMarginals(nodeScores[instanceID],
				edgeScores[instanceID], null, tMarginals);
		entropy[instanceID] = model.computeEntropy(nodeScores[instanceID],
				edgeScores[instanceID], tMarginals, logNorm[instanceID]);
		OptimizationHelper.computeSoftCounts(features, instanceID, tMarginals,
				expectedCounts, 1.0);
		updatePrimalParameters();
		objective -= entropy[instanceID];
		objective += 0.5 * lambda * ArrayHelper.l2NormSquared(parameters);
		/*
		System.out.println("ID::\t" + instanceID +
				"\tSTEP::\t" + currStep +
				"\tOBJ::\t" + objective + "->" + objectiveNew +
				"\tENT::\t" + entropy[instanceID] + "->" + entropyNew +
				"\tLOGNORM::\t" + logNorm[instanceID] + "->" + logNormNew +
				"\tPNORM::\t" + paraNormOld + "->" + paraNormNew);
		*/
	}
	
	private void computePrimalObjective() {
		double[] theta = new double[numFeatures];
		for (int i = 0; i < numFeatures; i++) {
			theta[i] = parameters[i] * lambda;
		}
		double primalObjective = 0.5 / lambda * ArrayHelper.l2NormSquared(theta);
		double[][] tEdgeScores = new double[numStates][numStates];
		features.computeEdgeScores(tEdgeScores, theta);
		for (int i : trainList) {
			int length = features.getInstanceLength(i);
			double[][] tNodeScores = new double[length][numTargetStates];
			double[][][] tMarginals =
					new double[length + 1][numStates][numStates];
			features.computeNodeScores(i, tNodeScores, theta);
			double tLogNorm = model.computeMarginals(tNodeScores, tEdgeScores,
					null, tMarginals);
			primalObjective -= model.computeLabelLikelihood(tNodeScores,
					tEdgeScores, tLogNorm, labels[i]);
		}
		System.out.println("primal objective::\t" + primalObjective);
	}
	
	private void computeGradient(int instanceID) {
		for (int i = 0; i < numStates; i++) {
			for (int j = 0; j < numStates; j++) {
				edgeGradient[instanceID][i][j] = edgeScores[instanceID][i][j] -
						lambda * features.computeEdgeScore(i, j, parametersGrad);
			}
		}
		int length = features.getInstanceLength(instanceID);
		for (int i = 0; i < length; i++) {
			for (int j = 0; j < numTargetStates; j++) {
				nodeGradient[instanceID][i][j] =
					nodeScores[instanceID][i][j] - lambda *
					features.computeNodeScore(instanceID, i, j, parametersGrad);
			}
		}
	}
	
	private void validate(int[] instList) {
		double[] runningAccuracy = new double[3];
		ArrayHelper.deepFill(runningAccuracy, 0.0);
		// compute objective and likelihood
		int numStates = features.numStates;
		for (int i : instList) {
			int length = features.getInstanceLength(i);
			double[][] nodeMarginals = new double[length][numStates];
			double[][][] edgeMarginals =
					new double[length + 1][numStates][numStates];
			int[] decoded = new int[length];
			model.computeMarginals(nodeScores[i], edgeScores[i], 
					nodeMarginals, edgeMarginals);
			model.posteriorDecoding(nodeMarginals, decoded);
			int[] result = eval.evaluate(labels[i], decoded);
			runningAccuracy[0] += result[0];
			runningAccuracy[1] += result[1];
			runningAccuracy[2] += result[2];
		}
		double precision = runningAccuracy[2] / runningAccuracy[1];
		double recall = runningAccuracy[2] / runningAccuracy[0];
		double f1 = (precision + recall > 0) ?
				(2 * precision * recall) / (precision + recall) : 0.0;
		System.out.println("\tPREC::\t" + precision + "\tREC::\t" + recall +
				"\tF1::\t" + f1);
	}
	

	public void computeAccuracy(int[] instList) {
		double[] theta = new double[numFeatures];
		for (int i = 0; i < numFeatures; i++) {
			theta[i] = parameters[i] * lambda;
		}
		OptimizationHelper.testModel(features, eval, labels, instList,
				theta);
	}
}
