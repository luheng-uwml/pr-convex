package optimization;

import java.util.Random;

import inference.SequentialInference;
import data.Evaluator;
import feature.SequentialFeatures;
import graph.GraphRegularizer;

public class RegularizedExponentiatedGradientDescent2 {
	SequentialFeatures features;
	SequentialInference model;
	GraphRegularizer graph;
	Evaluator eval;
	int[][] labels;
	int[] trainList, devList, workList;
	boolean[] isLabeled;
	double[] parameters, empiricalCounts, expectedCounts, logNorm, entropy;
	double[][] nodeCounts;
	double[][][] edgeScores, nodeScores, pEdgeScores, pNodeScores; 
	double lambda1, lambda2, objective, initialStepSize, totalGraphPenalty;
	int numTrains, numInstances, numFeatures, maxNumIterations, numStates,
		numTargetStates;
	Random randomGen;
	static final double stoppingCriterion = 1e-5;
	
	public RegularizedExponentiatedGradientDescent2(
			SequentialFeatures features, GraphRegularizer graph,
			int[][] labels, int[] trainList, int[] devList, Evaluator eval,
			double lambda1, double lambda2, double initialStepSize,
			int maxNumIterations, int randomSeed) {
		this.features = features;
		this.graph = graph;
		this.labels = labels;
		this.trainList = trainList;
		this.devList = devList;
		this.eval = eval;
		this.lambda1 = 1.0 / lambda1;
		this.lambda2 = lambda2;
		this.initialStepSize = initialStepSize;
		this.maxNumIterations = maxNumIterations;
		this.randomGen = new Random(randomSeed);
		initializeDataStructure();
		initializeObjective();
	}
	
	private void initializeDataStructure() {
		numInstances = features.numInstances;
		numFeatures = features.numAllFeatures;
		numStates = features.numStates;
		numTargetStates = features.numTargetStates;
		model= new SequentialInference(1000, numStates);
		parameters = new double[numFeatures];
		empiricalCounts = new double[numFeatures];
		expectedCounts = new double[numFeatures];
		nodeCounts = new double[numTargetStates][graph.numNodes];
		edgeScores = new double[numInstances][numStates][numStates];
		nodeScores = new double[numInstances][][];
		pEdgeScores = new double[numInstances][numStates][numStates];
		pNodeScores = new double[numInstances][][];
		logNorm = new double[numInstances];
		entropy = new double[numInstances];

		workList = new int[trainList.length + devList.length];
		isLabeled = new boolean[numInstances ];
		numTrains = trainList.length;
		for (int i = 0; i < trainList.length; i++) {
			workList[i] = trainList[i];
			isLabeled[i] = true;
		}
		for (int i = 0; i < devList.length; i++) {
			workList[i + numTrains] = devList[i];
			isLabeled[i + numTrains] = false;
		}
		for (int i : workList) {
			int length = features.getInstanceLength(i);
			nodeScores[i] = new double[length][numTargetStates];
			pNodeScores[i] = new double[length][numTargetStates];
			ArrayHelper.deepFill(nodeScores[i], 0.0);
			ArrayHelper.deepFill(pNodeScores[i], 0.0);
		}
		ArrayHelper.deepFill(parameters, 0.0);
		ArrayHelper.deepFill(empiricalCounts, 0.0);
		ArrayHelper.deepFill(expectedCounts, 0.0);
		ArrayHelper.deepFill(nodeCounts, 0.0);
		ArrayHelper.deepFill(edgeScores, 0.0);
		ArrayHelper.deepFill(pEdgeScores, 0.0);
		ArrayHelper.deepFill(entropy, 0.0);
		ArrayHelper.deepFill(logNorm, 0.0);
	}
	
	private void initializeObjective() {
		objective = 0;
		graph.setNodeCounts(workList);
		for (int instanceID : trainList) {
			OptimizationHelper.computeHardCounts(features, instanceID, labels,
					empiricalCounts);
			OptimizationHelper.computeHardCounts(graph, instanceID, labels,
					nodeCounts, 1.0);
		}
		for (int instanceID : workList) {
			int length = features.getInstanceLength(instanceID);
			double[][][] edgeMarginals =
					new double[length + 1][numStates][numStates];
			updateEntropy(instanceID, edgeMarginals);
			updateSoftCounts(instanceID, edgeMarginals, expectedCounts, 1.0);
			objective -= entropy[instanceID];
			if (!isLabeled[instanceID]) {
				// update for pseudo empirical counts
				updateSoftCounts(instanceID, edgeMarginals, empiricalCounts,
						1.0);
				OptimizationHelper.computeSoftCounts(graph, instanceID,
						edgeMarginals, nodeCounts, 1.0);
			}
		}
		updatePrimalParameters();
		totalGraphPenalty = graph.computeTotalPenalty(nodeCounts);
		objective += 0.5 * lambda1 * ArrayHelper.l2NormSquared(parameters);
		objective += 0.25 * lambda2 * totalGraphPenalty;
		System.out.println("initial objective::\t" + objective);
	}
	
	public void optimize() {
		double stepSize = initialStepSize;
		double prevObjective = objective;
		for (int iteration = 0; iteration < maxNumIterations; iteration ++) {
			for (int k = 0; k < workList.length; k++) {
				int instanceID = workList[randomGen.nextInt(workList.length)];
				update(instanceID, stepSize);
			}
			System.out.println("ITER::\t" + iteration +
					"\tSTEP:\t" + stepSize +
					"\tOBJ::\t" + objective +
					"\tPREV::\t" + prevObjective +
					"\tPARA::\t" + ArrayHelper.l2NormSquared(parameters) +
					"\tGRAPH::\t" + totalGraphPenalty);
			if (iteration % 5 == 4) {
				validate(trainList);
				validate(devList);
				computeAccuracy(devList);
				computePrimalObjective();
			}
			if (objective < prevObjective) {
				stepSize *= 1.02;
			} else {
				stepSize *= 0.5;
			}
			if (Math.abs((prevObjective - objective) / prevObjective) <
					stoppingCriterion) {
				break;
			}
			prevObjective = objective;
		}
		System.out.println("Optimization finished.");
		validate(trainList);
		validate(devList);
		computeAccuracy(devList);
		computePrimalObjective();
	}
	
	private void updatePrimalParameters() {
		for (int i = 0; i < numFeatures; i++) {
			parameters[i] = empiricalCounts[i] - expectedCounts[i];
		}
	}
	
	private void update(int instanceID, double stepSize) {
		int length = features.getInstanceLength(instanceID);
		double[][][] tMarginals = new double[length + 1][numStates][numStates];
		double[][] tEdgeGradient = new double[numStates][numStates],
					tNodeGradient = new double[length][numTargetStates];
		double tPenalty;
		
		// backup marginals and subtract from marginalized counts
		model.computeMarginals(nodeScores[instanceID], edgeScores[instanceID],
				null, tMarginals);
		updateSoftCounts(instanceID, tMarginals, expectedCounts, -1);
		if (!isLabeled[instanceID]) {
			model.computeMarginals(pNodeScores[instanceID],
					pEdgeScores[instanceID], null, tMarginals);
			updateSoftCounts(instanceID, tMarginals, empiricalCounts, -1);
			OptimizationHelper.computeSoftCounts(graph, instanceID, tMarginals,
					nodeCounts, +1);
		}
		
		// subtract from objective
		objective += entropy[instanceID];
		objective -= 0.5 * lambda1 * ArrayHelper.l2NormSquared(parameters);
		tPenalty = graph.computeTotalPenalty(instanceID, nodeCounts);
		totalGraphPenalty -= tPenalty;
		objective -= 0.25 * lambda2 * tPenalty;
		
		// compute gradient
		computeGradient(instanceID, tNodeGradient, tEdgeGradient);
		updateParameters(instanceID, nodeScores[instanceID],
				edgeScores[instanceID], tNodeGradient, tEdgeGradient, stepSize);
		updateEntropy(instanceID, tMarginals);
		updateSoftCounts(instanceID, tMarginals, expectedCounts, +1);
		if (!isLabeled[instanceID]) {
			computeGradientBar(instanceID, tNodeGradient, tEdgeGradient);
			updateParameters(instanceID, pNodeScores[instanceID],
					pEdgeScores[instanceID], tNodeGradient, tEdgeGradient,
					stepSize);
			model.computeMarginals(pNodeScores[instanceID],
					pEdgeScores[instanceID], null, tMarginals);
			updateSoftCounts(instanceID, tMarginals, empiricalCounts, +1);
			OptimizationHelper.computeSoftCounts(graph, instanceID, tMarginals,
					nodeCounts, +1);
		}
		// update counts and objective
		updatePrimalParameters();
		objective -= entropy[instanceID];
		objective += 0.5 * lambda1 * ArrayHelper.l2NormSquared(parameters);
		tPenalty = graph.computeTotalPenalty(instanceID, nodeCounts);
		totalGraphPenalty += tPenalty;
		objective += 0.25 * lambda2 * tPenalty;
	}
	
	private void updateEntropy(int instanceID, double[][][] edgeMarginals) {
		logNorm[instanceID] = model.computeMarginals(nodeScores[instanceID],
				edgeScores[instanceID], null, edgeMarginals);
		entropy[instanceID] = model.computeEntropy(nodeScores[instanceID],
				edgeScores[instanceID], edgeMarginals, logNorm[instanceID]);
	}
	
	private void updateSoftCounts(int instanceID, double[][][] edgeMarginals,
			double[] counts, double weight) {
		OptimizationHelper.computeSoftCounts(features, instanceID,
				edgeMarginals, counts, weight);
	}
	
	private void computePrimalObjective() {
		double[] theta = new double[numFeatures];
		for (int i = 0; i < numFeatures; i++) {
			theta[i] = parameters[i] * lambda1;
		}
		double primalObjective = 0.5 / lambda1 *
				ArrayHelper.l2NormSquared(theta);
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
	
	private void computeGradient(int instanceID, double[][] nodeGradient,
			double[][] edgeGradient) {
		int length = features.getInstanceLength(instanceID);
		if (isLabeled[instanceID]) {
			for (int i = 0; i < numStates; i++) {
				for (int j = 0; j < numStates; j++) {
					edgeGradient[i][j] = edgeScores[instanceID][i][j] -
						lambda1 * features.computeEdgeScore(i, j, parameters);
				}
			}
			for (int i = 0; i < length; i++) {
				for (int j = 0; j < numTargetStates; j++) {
					nodeGradient[i][j] = nodeScores[instanceID][i][j] -
						lambda1 * features.computeNodeScore(instanceID, i, j,
								parameters);
				}
			}
		}
	}
	
	private void computeGradientBar(int instanceID, double[][] nodeGradient,
			double[][] edgeGradient) {
		int length = features.getInstanceLength(instanceID);
		for (int i = 0; i < numStates; i++) {
			for (int j = 0; j < numStates; j++) {
				edgeGradient[i][j] =
					lambda1 * features.computeEdgeScore(i, j, parameters);
			}
		}
		for (int i = 0; i < length; i++) {
			for (int j = 0; j < numTargetStates; j++) {
				nodeGradient[i][j] =
					lambda1 * features.computeNodeScore(instanceID, i, j,
								parameters) -
					lambda2 * graph.computePenalty(instanceID, i,
								nodeCounts[j]);
			}
		}
	}
	
	private void updateParameters(int instanceID, double[][] nScores,
			double[][] eScores, double[][] nodeGradient,
			double[][] edgeGradient, double stepSize) {
		int length = features.getInstanceLength(instanceID);
		for (int i = 0; i < length; i++) {
			for (int j = 0; j < numTargetStates; j++) {
				nScores[i][j] -= stepSize * nodeGradient[i][j];
			}
		}
		for (int i = 0; i < numStates; i++) {
			for (int j = 0; j < numStates; j++) {
				eScores[i][j] -= stepSize * edgeGradient[i][j];
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
			theta[i] = parameters[i] * lambda1;
		}
		OptimizationHelper.testModel(features, eval, labels, instList, theta);
	}
}
