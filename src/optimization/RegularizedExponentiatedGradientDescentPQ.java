package optimization;

import java.util.Random;

import inference.SequentialInference;
import data.Evaluator;
import feature.SequentialFeatures;
import graph.GraphRegularizer;

/**
 * Semi-supervised version, with dummy variable
 * @author luheng
 *
 */
public class RegularizedExponentiatedGradientDescentPQ {
	SequentialFeatures features;
	SequentialInference model;
	GraphRegularizer graph;
	Evaluator eval;
	int[][] labels;
	int[] trainList, devList, workList;
	boolean[] isLabeled;
	double[] parameters, empiricalCounts, expectedCounts, logNorm, entropy;
	double[][] nodeCounts;
	double[][][] pEdges, pNodes, qEdges, qNodes; 
	double lambda1, lambda2, objective, initialStepSize, totalGraphPenalty,
		   unlabeledWeight;
	int numTrains, numInstances, numFeatures, maxNumIterations, numStates,
		numTargetStates;
	Random randomGen;
	static final double stoppingCriterion = 1e-5;
	
	public RegularizedExponentiatedGradientDescentPQ(
			SequentialFeatures features, GraphRegularizer graph,
			int[][] labels, int[] trainList, int[] devList, Evaluator eval,
			double lambda1, double lambda2, double unlabeledWeight,
			double initialStepSize, int maxNumIterations, int randomSeed) {
		this.features = features;
		this.graph = graph;
		this.labels = labels;
		this.trainList = trainList;
		this.devList = devList;
		this.eval = eval;
		this.lambda1 = 1.0 / lambda1;
		this.lambda2 = lambda2;
		this.unlabeledWeight = unlabeledWeight;
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
		pEdges = new double[numInstances][numStates][numStates];
		pNodes = new double[numInstances][][];
		qEdges = new double[numInstances][numStates][numStates];
		qNodes = new double[numInstances][][];
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
			pNodes[i] = new double[length][numTargetStates];
			qNodes[i] = new double[length][numTargetStates];
			ArrayHelper.deepFill(pNodes[i], 0.0);
			ArrayHelper.deepFill(qNodes[i], 0.0);
		}
		ArrayHelper.deepFill(parameters, 0.0);
		ArrayHelper.deepFill(empiricalCounts, 0.0);
		ArrayHelper.deepFill(expectedCounts, 0.0);
		ArrayHelper.deepFill(nodeCounts, 0.0);
		ArrayHelper.deepFill(pEdges, 0.0);
		ArrayHelper.deepFill(qEdges, 0.0);
		ArrayHelper.deepFill(entropy, 0.0);
		ArrayHelper.deepFill(logNorm, 0.0);
	}
	
	private void initializeObjective() {
		objective = 0;
		graph.setNodeCounts(workList);
		// compute empirical counts
		for (int instanceID : trainList) {
			OptimizationHelper.computeHardCounts(features, instanceID, labels,
					empiricalCounts);
			OptimizationHelper.computeHardCounts(graph, instanceID, labels,
					nodeCounts, 1.0);
		}
		// compute expected counts
		for (int instanceID : workList) {
			int length = features.getInstanceLength(instanceID);
			double[][][] tMarginals = new double[length+1][numStates][numStates];
			updateEntropy(instanceID, tMarginals);
			OptimizationHelper.computeSoftCounts(features, instanceID,
					tMarginals, expectedCounts,
					isLabeled[instanceID] ? 1.0 : unlabeledWeight);
			objective -= entropy[instanceID];
			// update for pseudo empirical counts
			if (!isLabeled[instanceID]) {
				OptimizationHelper.computeSoftCounts(features, instanceID,
						tMarginals, empiricalCounts, unlabeledWeight);
				OptimizationHelper.computeSoftCounts(graph, instanceID,
						tMarginals, nodeCounts, +1);
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
		double[][][] tMarginals = new double[length+1][numStates][numStates];
		double[][] tEdgeGradient = new double[numStates][numStates],
					tNodeGradient = new double[length][numTargetStates];
		double tPenalty;
		
		// subtract from objective
		objective += entropy[instanceID];
		objective -= 0.5 * lambda1 * ArrayHelper.l2NormSquared(parameters);
		tPenalty = graph.computeTotalPenalty(instanceID, nodeCounts);
		totalGraphPenalty -= tPenalty;
		objective -= 0.25 * lambda2 * tPenalty;
		
		// backup marginals and subtract from marginalized counts
		model.computeMarginals(pNodes[instanceID], pEdges[instanceID],
				null, tMarginals);
		OptimizationHelper.computeSoftCounts(features, instanceID, tMarginals,
				expectedCounts, isLabeled[instanceID] ? -1 : -unlabeledWeight);
		
		// backup u-bar parameters
		if (!isLabeled[instanceID]) {
			model.computeMarginals(qNodes[instanceID],
					qEdges[instanceID], null, tMarginals);
			OptimizationHelper.computeSoftCounts(features, instanceID,
					tMarginals, empiricalCounts, -unlabeledWeight);
			OptimizationHelper.computeSoftCounts(graph, instanceID, tMarginals,
					nodeCounts, -1);
		}
		
		// compute gradient
		computeGradient(instanceID, tNodeGradient, tEdgeGradient);
		updateParameters(instanceID, pNodes[instanceID],
				pEdges[instanceID], tNodeGradient, tEdgeGradient, stepSize);
		updateEntropy(instanceID, tMarginals);
		OptimizationHelper.computeSoftCounts(features, instanceID, tMarginals,
				expectedCounts, isLabeled[instanceID] ? 1.0 : unlabeledWeight);
		
		// compute gradient for u-bar parameters
		if (!isLabeled[instanceID]) {
			computeQueGradient(instanceID, tNodeGradient, tEdgeGradient);
			updateParameters(instanceID, qNodes[instanceID],
					qEdges[instanceID], tNodeGradient, tEdgeGradient,
					stepSize);
			model.computeMarginals(qNodes[instanceID],
					qEdges[instanceID], null, tMarginals);
			OptimizationHelper.computeSoftCounts(features, instanceID,
					tMarginals, empiricalCounts, unlabeledWeight);
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
		logNorm[instanceID] = model.computeMarginals(pNodes[instanceID],
				pEdges[instanceID], null, edgeMarginals);
		entropy[instanceID] = model.computeEntropy(pNodes[instanceID],
				pEdges[instanceID], edgeMarginals, logNorm[instanceID]);
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
		double weight = lambda1 * (isLabeled[instanceID] ? 1.0 :
			unlabeledWeight);
		for (int i = 0; i < numStates; i++) {
			for (int j = 0; j < numStates; j++) {
				edgeGradient[i][j] = pEdges[instanceID][i][j] -
					weight * features.computeEdgeScore(i, j, parameters);
			}
		}
		for (int i = 0; i < length; i++) {
			for (int j = 0; j < numTargetStates; j++) {
				nodeGradient[i][j] = pNodes[instanceID][i][j] -
					weight * features.computeNodeScore(instanceID, i, j,
							parameters);
			}
		}
	}
	
	private void computeQueGradient(int instanceID, double[][] nodeGradient,
			double[][] edgeGradient) {
		int length = features.getInstanceLength(instanceID);
		for (int i = 0; i < numStates; i++) {
			for (int j = 0; j < numStates; j++) {
				edgeGradient[i][j] = qEdges[instanceID][i][j] - 
						unlabeledWeight * lambda1 *
						features.computeEdgeScore(i, j, parameters);
			}
		}
		for (int i = 0; i < length; i++) {
			for (int j = 0; j < numTargetStates; j++) {
				nodeGradient[i][j] = qNodes[instanceID][i][j] -
					unlabeledWeight * lambda1 *
					features.computeNodeScore(instanceID, i, j, parameters) -
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
			model.computeMarginals(pNodes[i], pEdges[i], 
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
		OptimizationHelper.testModel(features, eval, instList, labels, null,
				theta);
	}
}
