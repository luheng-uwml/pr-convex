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
public class PQEGTrainer implements AbstractOptimizer {
	SequentialFeatures features;
	SequentialInference model;
	GraphRegularizer graph;
	OptimizationHistory history;
	Evaluator eval;
	int[][] labels, predictions;
	int[] trainList, devList, workList;
	boolean[] isLabeled;
	double[] parameters, empiricalCounts, expectedCounts, pLogNorm, pEntropy,
			 qLogNorm, qEntropy;
	double[][] nodeCounts;
	double[][][] pEdges, pNodes, qEdges, qNodes; 
	double lambda1, lambda2, objective, initialStepSize, totalGraphPenalty,
		   unlabeledWeight;
	int numTrains, numInstances, numFeatures, maxNumIterations, numStates,
		numTargetStates;
	Random randomGen;
	static final double stoppingCriterion = 1e-5;
	static final boolean maxEntQ = true;
	
	public PQEGTrainer(
			SequentialFeatures features, GraphRegularizer graph,
			int[][] labels, int[] trainList, int[] devList, Evaluator eval,
			double lambda1, double lambda2, double unlabeledWeight,
			double initialStepSize, int maxNumIterations,
			int warmStartIterations, int randomSeed) {
		this.features = features;
		this.graph = graph;
		this.labels = labels;
		this.trainList = trainList;
		this.devList = devList;
		this.eval = eval;
		this.lambda1 = lambda1;
		this.lambda2 = lambda2;
		this.unlabeledWeight = unlabeledWeight;
		this.initialStepSize = initialStepSize;
		this.maxNumIterations = maxNumIterations;
		this.randomGen = new Random(randomSeed);
		this.history = new OptimizationHistory();
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
		pLogNorm = new double[numInstances];
		qLogNorm = new double[numInstances];
		pEntropy = new double[numInstances];
		qEntropy = new double[numInstances];
		workList = new int[trainList.length + devList.length];
		isLabeled = new boolean[numInstances];
		predictions = new int[numInstances][];
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
			ArrayHelper.deepFill(pNodes[i], 0.0);
			if (!isLabeled[i]) {
				qNodes[i] = new double[length][numTargetStates];
				ArrayHelper.deepFill(qNodes[i], 0.0);
			}
			predictions[i] = new int[length];
		}
		ArrayHelper.deepFill(parameters, 0.0);
		ArrayHelper.deepFill(empiricalCounts, 0.0);
		ArrayHelper.deepFill(expectedCounts, 0.0);
		ArrayHelper.deepFill(nodeCounts, 0.0);
		ArrayHelper.deepFill(pEdges, 0.0);
		ArrayHelper.deepFill(qEdges, 0.0);
	}
	
	public OptimizationHistory getOptimizationHistory() {
		return history;
	}
	
	public int[][] getPrediction() {
		return predictions;
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
		for (int instanceID : trainList) {
			int length = features.getInstanceLength(instanceID);
			double[][][] tMarginals = new double[length+1][numStates][numStates];
			updateEntropy(instanceID, tMarginals, 'P');
			OptimizationHelper.computeSoftCounts(features, instanceID,
					tMarginals, expectedCounts,
					isLabeled[instanceID] ? 1.0 : unlabeledWeight);
			objective -= pEntropy[instanceID];
		}
		updatePrimalParameters();
		objective += 0.5 * lambda1 * ArrayHelper.l2NormSquared(parameters);
		System.out.println("initial objective (P)::\t" + objective);
	}
	
	private void initializeQ() {
		double[] theta = new double[numFeatures];
		for (int i = 0; i < numFeatures; i++) {
			theta[i] = lambda1 * (empiricalCounts[i] - expectedCounts[i]);
		}
		for (int instanceID : devList) {
			int length = features.getInstanceLength(instanceID);
			double[][][] tMarginals = new double[length+1][numStates][numStates];			
			features.computeNodeScores(instanceID, pNodes[instanceID], theta);
			features.computeEdgeScores(pEdges[instanceID], theta);
			updateEntropy(instanceID, tMarginals, 'P');
			ArrayHelper.deepCopy(pNodes[instanceID], qNodes[instanceID]);
			ArrayHelper.deepCopy(pEdges[instanceID], qEdges[instanceID]);
			qLogNorm[instanceID] = pLogNorm[instanceID];
			qEntropy[instanceID] = pEntropy[instanceID];
			objective -= pEntropy[instanceID] + qEntropy[instanceID];
			OptimizationHelper.computeSoftCounts(features, instanceID,
					tMarginals, expectedCounts, unlabeledWeight);
			OptimizationHelper.computeSoftCounts(features, instanceID,
					tMarginals, empiricalCounts, unlabeledWeight);
			OptimizationHelper.computeSoftCounts(graph, instanceID,
					tMarginals, nodeCounts, 1);
		}
		updatePrimalParameters();
		totalGraphPenalty = graph.computeTotalPenalty(nodeCounts);
		objective += 0.25 * lambda2 * totalGraphPenalty;
		System.out.println("initial objective (P+Q) ::\t" + objective);
	}
	
	public void optimize() {
		double lambda1Final = lambda1;
		lambda1 = Math.min(1.0, lambda1Final);
		objective += 0.5 * (lambda1 - lambda1Final) *
				ArrayHelper.l2NormSquared(parameters);
		double stepSize = initialStepSize;
		double prevObjective = objective;
		
		// warm start
		for (int iteration = 0; iteration < 100; iteration ++) {
			for (int k = 0; k < trainList.length; k++) {
				updateP(trainList[randomGen.nextInt(trainList.length)], stepSize);
			}
			System.out.println("ITER::\t" + iteration +
					"\tSTEP:\t" + stepSize +
					"\tLAMBDA:\t" + lambda1 +
					"\tOBJ::\t" + objective +
					"\tPREV::\t" + prevObjective +
					"\tPARA::\t" + ArrayHelper.l2NormSquared(parameters) +
					"\tGRAPH::\t" + totalGraphPenalty);
			if (iteration % 10 == 9) {
				validateP(trainList);
				computeAccuracy(devList);
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
		
		initializeQ();
		prevObjective = objective;
		
		double improvement = Double.MAX_VALUE;
		for (int iteration = 0; iteration < maxNumIterations; iteration ++) {
			for (int k = 0; k < workList.length; k++) {
				int lottery = randomGen.nextInt(workList.length + devList.length);
				if (lottery < workList.length) {
					updateP(workList[lottery], stepSize);
				} else {
					updateQ(devList[lottery - workList.length], stepSize);
				}
			}
			
			double paraNorm = ArrayHelper.l2NormSquared(parameters);
			
			System.out.println("ITER::\t" + iteration +
					"\tSTEP:\t" + stepSize +
					"\tLAMBDA:\t" + lambda1 +
					"\tOBJ::\t" + objective +
					"\tIMPR::\t" + improvement +
					"\tPREV::\t" + prevObjective +
					"\tPARA::\t" + paraNorm +
					"\tGRAPH::\t" + totalGraphPenalty);
			
			double[] devAcc = validateP(devList);
			history.add(iteration, "objective", objective);
			history.add(iteration, "stepsize", stepSize);
			history.add(iteration, "para_norm", paraNorm);
			history.add(iteration, "lambda1", lambda1);
			history.add(iteration, "graph_norm", totalGraphPenalty);
			history.add(iteration, "dev_f1", devAcc[2]);

			if (iteration % 5 == 4) {
				validateP(trainList);
				validateP(devList);
				validateQ(devList);
				computeAccuracy(devList);
				computePrimalObjective();
			}
			if (objective < prevObjective) {
				stepSize *= 1.02;
			} else {
				stepSize *= 0.5;
			}
			improvement = Math.abs((prevObjective - objective) / prevObjective);
			if (improvement < 1e-3 && lambda1 < lambda1Final) {
				double oldLambda = lambda1;
				lambda1 = Math.min(lambda1 * 1.5, lambda1Final);
				objective += 0.5 * (lambda1 - oldLambda) *
						ArrayHelper.l2NormSquared(parameters);
			}
			if (improvement < stoppingCriterion && lambda1 >= lambda1Final) {
				break;
			}
			prevObjective = objective;
		}
		System.out.println("Optimization finished.");
		validateP(trainList);
		validateP(devList);
		validateQ(devList);
		computeAccuracy(devList);
		computePrimalObjective();
	}
	
	private void updatePrimalParameters() {
		for (int i = 0; i < numFeatures; i++) {
			parameters[i] = empiricalCounts[i] - expectedCounts[i];
		}
	}
	
	private void updateP(int instanceID, double stepSize) {
		int length = features.getInstanceLength(instanceID);
		double[][][] tMarginals = new double[length+1][numStates][numStates];
		double[][] tEdgeGradient = new double[numStates][numStates],
					tNodeGradient = new double[length][numTargetStates];
		
		objective += pEntropy[instanceID];
		objective -= 0.5 * lambda1 * ArrayHelper.l2NormSquared(parameters);
		model.computeMarginals(pNodes[instanceID], pEdges[instanceID], null,
				tMarginals);
		OptimizationHelper.computeSoftCounts(features, instanceID, tMarginals,
				expectedCounts, isLabeled[instanceID] ? -1 : -unlabeledWeight);
		
		computeGradientP(instanceID, tNodeGradient, tEdgeGradient);
		updateParameters(instanceID, pNodes[instanceID], pEdges[instanceID],
				tNodeGradient, tEdgeGradient, stepSize);
		updateEntropy(instanceID, tMarginals, 'P');
		OptimizationHelper.computeSoftCounts(features, instanceID, tMarginals,
				expectedCounts, isLabeled[instanceID] ? 1 : unlabeledWeight);
		
		updatePrimalParameters();
		objective -= pEntropy[instanceID];
		objective += 0.5 * lambda1 * ArrayHelper.l2NormSquared(parameters);
	}
	
	private void updateQ(int instanceID, double stepSize) {
		int length = features.getInstanceLength(instanceID);
		double[][][] tMarginals = new double[length+1][numStates][numStates];
		double[][] tEdgeGradient = new double[numStates][numStates],
					tNodeGradient = new double[length][numTargetStates];
		double tPenalty;
		
		if (maxEntQ) {
			objective += qEntropy[instanceID];
		}
		objective -= 0.5 * lambda1 * ArrayHelper.l2NormSquared(parameters);
		tPenalty = graph.computeTotalPenalty(instanceID, nodeCounts);
		totalGraphPenalty -= tPenalty;
		objective -= 0.25 * lambda2 * tPenalty;
		
		model.computeMarginals(qNodes[instanceID], qEdges[instanceID], null,
				tMarginals);
		OptimizationHelper.computeSoftCounts(features, instanceID, tMarginals,
				empiricalCounts, -unlabeledWeight);
		OptimizationHelper.computeSoftCounts(graph, instanceID, tMarginals,
				nodeCounts, -1);
		
		computeGradientQ(instanceID, tNodeGradient, tEdgeGradient);
		updateParameters(instanceID, qNodes[instanceID], qEdges[instanceID],
				tNodeGradient, tEdgeGradient, stepSize);
		model.computeMarginals(qNodes[instanceID], qEdges[instanceID], null,
				tMarginals);
		OptimizationHelper.computeSoftCounts(features, instanceID, tMarginals,
				empiricalCounts, unlabeledWeight);
		OptimizationHelper.computeSoftCounts(graph, instanceID, tMarginals,
				nodeCounts, 1);
		updatePrimalParameters();
		if (maxEntQ) {
			updateEntropy(instanceID, tMarginals, 'Q');
			objective -= qEntropy[instanceID];
		}
		objective += 0.5 * lambda1 * ArrayHelper.l2NormSquared(parameters);
		tPenalty = graph.computeTotalPenalty(instanceID, nodeCounts);
		totalGraphPenalty += tPenalty;
		objective += 0.25 * lambda2 * tPenalty;
	}
	
	private void updateEntropy(int instanceID, double[][][] marginals,
			char varSet) {
		if (varSet == 'P') {
			pLogNorm[instanceID] = model.computeMarginals(pNodes[instanceID],
					pEdges[instanceID], null, marginals);
			pEntropy[instanceID] = model.computeEntropy(pNodes[instanceID],
					pEdges[instanceID], marginals, pLogNorm[instanceID]);
		} else if (maxEntQ) {
			qLogNorm[instanceID] = model.computeMarginals(qNodes[instanceID],
					qEdges[instanceID], null, marginals);
			qEntropy[instanceID] = model.computeEntropy(qNodes[instanceID],
					qEdges[instanceID], marginals, qLogNorm[instanceID]);
		}
	}
	
	private void computePrimalObjective() {
		double[] theta = new double[numFeatures];
		for (int i = 0; i < numFeatures; i++) {
			theta[i] = parameters[i] * lambda1;
		}
		double primalObjective = 0.5 / lambda1 * ArrayHelper.l2NormSquared(theta);
		double[][] tEdges = new double[numStates][numStates];
		features.computeEdgeScores(tEdges, theta);
		// Compute E_lab[P] and E_q[P]
		for (int i : workList) {
			int length = features.getInstanceLength(i);
			double[][] tNodes = new double[length][numTargetStates];
			double[][][] pMargs = new double[length + 1][numStates][numStates];
			features.computeNodeScores(i, tNodes, theta);
			double pLogNorm = model.computeMarginals(tNodes, tEdges, null, pMargs);
			if (isLabeled[i]) {
				primalObjective -= model.computeLabelLikelihood(tNodes, tEdges,
						pLogNorm, labels[i]);
			} else {
				double[][][] qMargs = new double[length + 1][numStates][numStates];
				model.computeMarginals(qNodes[i], qEdges[i], null, qMargs);
				primalObjective -= model.computeWeightedLikelihood(tNodes,
						tEdges, pLogNorm, qMargs);
			}
		}
		System.out.println("primal objective::\t" + primalObjective);
	}
	
	private void computeGradientP(int instanceID, double[][] nodeGradient,
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
					weight * features.computeNodeScore(instanceID, i, j, parameters);
			}
		}
	}
	
	private void computeGradientQ(int instanceID, double[][] nodeGradient,
			double[][] edgeGradient) {
		int length = features.getInstanceLength(instanceID);
		double w0 = maxEntQ ? 1.0 : 0;
		double w1 = unlabeledWeight * lambda1;
		for (int i = 0; i < numStates; i++) {
			for (int j = 0; j < numStates; j++) {
				edgeGradient[i][j] = w0 * qEdges[instanceID][i][j] +
					w1 * features.computeEdgeScore(i, j, parameters);
			}
		}
		for (int i = 0; i < length; i++) {
			for (int j = 0; j < numTargetStates; j++) {
				nodeGradient[i][j] = w0 * qNodes[instanceID][i][j] +
					w1 * features.computeNodeScore(instanceID, i, j, parameters) -
					lambda2 * graph.computePenalty(instanceID, i, nodeCounts[j]);
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
	
	private double[] validateP(int[] instList) {
		double[] runningAccuracy = new double[3];
		ArrayHelper.deepFill(runningAccuracy, 0.0);
		// compute objective and likelihood
		for (int i : instList) {
			int length = features.getInstanceLength(i);
			int[] decoded = new int[length];
			/*double[][] nodeMarginals = new double[length][numStates];
			double[][][] edgeMarginals =
					new double[length + 1][numStates][numStates];
			model.computeMarginals(pNodes[i], pEdges[i], 
					nodeMarginals, edgeMarginals);
			model.posteriorDecoding(nodeMarginals, decoded);
			*/
			model.viterbiDecoding(pNodes[i], pEdges[i], decoded);
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
		return runningAccuracy;
	}
	
	private double[] validateQ(int[] instList) {
		double[] runningAccuracy = new double[3];
		ArrayHelper.deepFill(runningAccuracy, 0.0);
		// compute objective and likelihood
		for (int i : instList) {
			int length = features.getInstanceLength(i);
			int[] decoded = new int[length];
			model.viterbiDecoding(qNodes[i], qEdges[i], decoded);
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
		return runningAccuracy;
	}
	
	public void computeAccuracy(int[] instList) {
		double[] theta = new double[numFeatures];
		for (int i = 0; i < numFeatures; i++) {
			theta[i] = parameters[i] * lambda1;
		}
		OptimizationHelper.testModel(features, eval, instList, labels,
				predictions, theta);
	}
}
