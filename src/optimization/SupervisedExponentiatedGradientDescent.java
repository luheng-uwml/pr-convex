package optimization;

import java.util.Arrays;
import java.util.Random;

import inference.SequentialInference;
import data.Evaluator;
import feature.SequentialFeatures;
import graph.GraphRegularizer;

/**
 * not exactly supervised ..
 * when computing feature agreement, only use labeled data ..
 * @author luheng
 *
 */
public class SupervisedExponentiatedGradientDescent {
	public int[][] predictions;
	
	SequentialFeatures features;
	SequentialInference model;
	GraphRegularizer graph;
	Evaluator eval;
	int[][] labels;
	int[] trainList, devList;
	double[] parameters, empiricalCounts, expectedCounts, trainRatio, logNorm,
	         entropy;
	double[][] nodeCounts;
	double[][][] edgeScores, nodeScores; 
	double lambda1, lambda2, finalLambda1, objective, initialStepSize,
		   totalGraphPenalty;
	int numTrains, numInstances, numFeatures, maxNumIterations, numStates,
		numTargetStates;
	Random randomGen;
	static final double stoppingCriterion = 1e-5, minLambda = 1e-2,
						lambdaScale = 1.5; 
	
	public SupervisedExponentiatedGradientDescent(
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
		this.finalLambda1 = 1.0 / lambda1;
		this.lambda2 = lambda2;
		this.initialStepSize = initialStepSize;
		this.maxNumIterations = maxNumIterations;
		this.randomGen = new Random(randomSeed);

		initializeDataStructure();
		// trainRatio just for test use, remove later
		initializeTrainRatio();
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
		trainRatio = new double[numFeatures];
		nodeCounts = new double[numTargetStates][graph.numNodes];
		edgeScores = new double[numInstances][numStates][numStates];
		nodeScores = new double[numInstances][][];
		predictions = new int[numInstances][];
		logNorm = new double[numInstances];
		entropy = new double[numInstances];
		for (int i : trainList) {
			int length = features.getInstanceLength(i);
			nodeScores[i] = new double[length][numTargetStates];
			ArrayHelper.deepFill(nodeScores[i], 0.0);
			predictions[i] = new int[length];
		}
		for (int i : devList) {
			predictions[i] = new int[features.getInstanceLength(i)];
		}
		ArrayHelper.deepFill(parameters, 0.0);
		ArrayHelper.deepFill(empiricalCounts, 0.0);
		ArrayHelper.deepFill(expectedCounts, 0.0);
		ArrayHelper.deepFill(nodeCounts, 0.0);
		ArrayHelper.deepFill(edgeScores, 0.0);
		ArrayHelper.deepFill(entropy, 0.0);
		ArrayHelper.deepFill(logNorm, 0.0);
	}
	
	private void initializeTrainRatio() {
		double[] trainFeatureCounts = new double[numFeatures];
		double[] devFeatureCounts = new double[numFeatures];
		Arrays.fill(trainFeatureCounts, 0);
		Arrays.fill(devFeatureCounts, 0);
		features.countFeatures(trainList, trainFeatureCounts);
		features.countFeatures(devList, devFeatureCounts);
		// features that are not in training data should be ignored 
		
		for (int i = 0; i < numFeatures; i++) {
			if (trainFeatureCounts[i] > 0) {
				trainRatio[i] = 1.0 / trainFeatureCounts[i] *
						(trainFeatureCounts[i] + devFeatureCounts[i]);
			} else {
				trainRatio[i] = 0.0;
			}
		}
		System.out.println("bias-train:\t" + trainFeatureCounts[0] +
						   "\tbias-dev:\t" + devFeatureCounts[0] +
						   "\ttrain ratio:\t" + trainRatio[0]);
		double avgRatio = 0, nnzRatio = 0;
		for (int i = 0; i < numFeatures; i++) {
			if (trainRatio[i] != 0) {
				avgRatio += trainRatio[i];
				nnzRatio += 1;
			}
		}
		System.out.println("Averaged non-zero train ratio::\t" +
				avgRatio / nnzRatio +
				".\tNumber of non-zero train ratio elements::\t" +
				(int) nnzRatio + " out of all " + numFeatures + " features.");
	
	}
	
	private void initializeObjective() {
		objective = 0;
		graph.setNodeCounts(trainList);
		// compute empirical counts
		for (int instanceID : trainList) {
			OptimizationHelper.computeHardCounts(features, instanceID, labels,
					empiricalCounts);
			//OptimizationHelper.computeHardCounts(graph, instanceID, labels,
			//		nodeCounts, 1.0);
		}
		// compute expected counts
		for (int instanceID : trainList) {
			int length = features.getInstanceLength(instanceID);
			double[][][] tMarginals = new double[length+1][numStates][numStates];
			updateEntropy(instanceID, tMarginals);
			objective -= entropy[instanceID];
			OptimizationHelper.computeSoftCounts(features, instanceID,
					tMarginals, expectedCounts, 1.0);
			OptimizationHelper.computeSoftCounts(graph, instanceID, tMarginals,
					nodeCounts, +1);
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
		lambda1 = Math.min(minLambda, finalLambda1);
		for (int iteration = 0; iteration < maxNumIterations; iteration ++) {
			for (int k = 0; k < trainList.length; k++) {
				int instanceID = trainList[randomGen.nextInt(trainList.length)];
				update(instanceID, stepSize);
			}
			System.out.println("ITER::\t" + iteration +
					"\tSTEP:\t" + stepSize +
					"\tOBJ::\t" + objective +
					"\tPREV::\t" + prevObjective +
					"\tLAMBDA::\t" + lambda1 +
					"\tPARA::\t" + ArrayHelper.l2NormSquared(parameters) +
					"\tGRAPH::\t" + totalGraphPenalty);
			if (iteration % 5 == 4) {
				validate(trainList);
				predict(devList);
				computePrimalObjective();
				computeFeatureAgreement();
			}
			if (objective < prevObjective) {
				stepSize *= 1.02;
			} else {
				stepSize *= 0.5;
			}
			double impr = Math.abs((prevObjective - objective) / prevObjective);
			if (impr < 1e-3) {
				lambda1 = Math.min(lambda1 * lambdaScale, finalLambda1);
				// recompute objective
				objective = 0.5 * lambda1 * ArrayHelper.l2NormSquared(parameters);
				for (int i : trainList) {
					objective -= entropy[i];
				}
				stepSize *= 0.5;
			} else if (impr < stoppingCriterion && lambda1 >= finalLambda1) {
				break;
			}
			prevObjective = objective;
		}
		System.out.println("Optimization finished.");
		validate(trainList);
		predict(devList);
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
		model.computeMarginals(nodeScores[instanceID], edgeScores[instanceID],
				null, tMarginals);
		OptimizationHelper.computeSoftCounts(features, instanceID, tMarginals,
				expectedCounts, -1);
		OptimizationHelper.computeSoftCounts(graph, instanceID, tMarginals,
				nodeCounts, -1);
		
		// compute gradient
		computeGradient(instanceID, tNodeGradient, tEdgeGradient);
		updateParameters(instanceID, nodeScores[instanceID],
				edgeScores[instanceID], tNodeGradient, tEdgeGradient, stepSize);
		updateEntropy(instanceID, tMarginals);
		OptimizationHelper.computeSoftCounts(features, instanceID, tMarginals,
				expectedCounts, 1.0);
		OptimizationHelper.computeSoftCounts(graph, instanceID, tMarginals,
				nodeCounts, 1.0);

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
					new double[length+1][numStates][numStates];
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
	

	public void predict(int[] instList) {
		double[] theta = new double[numFeatures];
		for (int i = 0; i < numFeatures; i++) {
			theta[i] = parameters[i] * lambda1;
		}
		OptimizationHelper.testModel(features, eval, instList, labels,
				predictions, theta);
	}
	
	private void computeFeatureAgreement() {
		double[] counts = new double[numFeatures];
		double[] theta = new double[numFeatures];
		for (int i = 0; i < numFeatures; i++) {
			theta[i] = parameters[i] * lambda1;
		}
		
		ArrayHelper.deepCopy(expectedCounts, counts);
		double[][] edgeScores = new double[numStates][numStates];
		features.computeEdgeScores(edgeScores, parameters);
		for (int i : devList) {
			int length = features.getInstanceLength(i);
			double[][] nodeScores = new double[length][numTargetStates];
			double[][][] tMarginals =
					new double[length + 1][numStates][numStates];
			features.computeNodeScores(i, nodeScores, parameters);
			model.computeMarginals(nodeScores, edgeScores,  null, tMarginals);
			OptimizationHelper.computeSoftCounts(features, i, tMarginals,
					counts);
		}
		
		double totalDiff = 0;
		double tr = 1.0 * (trainList.length + devList.length) / trainList.length;
		double minDiff = Double.MAX_VALUE, maxDiff = Double.MIN_VALUE,
			   avgDiff = 0;
		for (int i = 0; i < numFeatures; i++) {
			double diff = empiricalCounts[i] * trainRatio[i] - counts[i];
			totalDiff += diff * diff;
			minDiff = Math.min(diff * diff, minDiff);
			maxDiff = Math.max(diff * diff, maxDiff);
			avgDiff += diff * diff / numFeatures;
			/*
			if (diff * diff > 10000) {
				System.out.println(i + "\t" + trainRatio[i] + "\t" +
								   features.getFeatureName(i) + "\t" + diff +
								   "\t" + empiricalCounts[i] + "\t" +
								   counts[i] + "\t" +
								   (1.0 * counts[i] / empiricalCounts[i]));
			} */
		}
		System.out.println("gold diff:\t" + totalDiff + "\tmin:\t" + minDiff +
					       "\tmax:\t" + maxDiff + "\tavg diff:\t" + avgDiff);
	}
}
