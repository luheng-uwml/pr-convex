package optimization;

import data.Evaluator;
import inference.SequentialInference;
import feature.SequentialFeatures;
import graph.GraphRegularizer;

public class OptimizationHelper {
	public static void computeHardCounts(SequentialFeatures features,
			int instanceID, int[][] labels, double[] counts) {
		int length = features.getInstanceLength(instanceID);
		for (int i = 0; i <= length; i++) {
			int s = (i == length) ? features.SN : labels[instanceID][i];
			int sp = (i == 0) ? features.S0 : labels[instanceID][i - 1];
			features.addToCounts(instanceID, i, s, sp, counts, 1.0);
		}
	}
	
	public static void computeSoftCounts(SequentialFeatures features,
			int instanceID, double[][][] edgeMarginals, double[] counts) {
		int length = features.getInstanceLength(instanceID);
		for (int s = 0; s < features.numTargetStates; s++) {
			features.addToCounts(instanceID, 0, s, features.S0, counts,
					edgeMarginals[0][s][features.S0]);
			features.addToCounts(instanceID, length, features.SN, s, counts,
					edgeMarginals[length][features.SN][s]);
		}
		for (int i = 1; i < length; i++) {
			for (int s = 0; s < features.numTargetStates; s++) {
				for (int sp = 0; sp < features.numTargetStates; sp++) {
					features.addToCounts(instanceID, i, s, sp, counts,
							edgeMarginals[i][s][sp]);
				}
			}
		}
	}
	
	public static void computeSoftCounts(SequentialFeatures features,
			int instanceID, double[][][] edgeMarginals, double[] counts,
			double scale) {
		int length = features.getInstanceLength(instanceID);
		for (int s = 0; s < features.numTargetStates; s++) {
			features.addToCounts(instanceID, 0, s, features.S0, counts,
					scale * edgeMarginals[0][s][features.S0]);
			features.addToCounts(instanceID, length, features.SN, s, counts,
					scale * edgeMarginals[length][features.SN][s]);
		}
		for (int i = 1; i < length; i++) {
			for (int s = 0; s < features.numTargetStates; s++) {
				for (int sp = 0; sp < features.numTargetStates; sp++) {
					features.addToCounts(instanceID, i, s, sp, counts,
							scale * edgeMarginals[i][s][sp]);
				}
			}
		}
	}
	
	public static void computeHardCounts(GraphRegularizer graph,
			int instanceID, int[][] labels, double[][] counts,
			double scale) {
		for (int i = 0; i < labels[instanceID].length; i++) {
			int s = labels[instanceID][i];
			graph.addToCounts(instanceID, i, counts[s], 1.0);
		}
	}
	
	public static void computeSoftCounts(GraphRegularizer graph,
			int instanceID, double[][][] edgeMarginals, double[][] counts,
			double scale) {
		int numTargetStates = counts.length;
		for (int i = 0; i < edgeMarginals.length - 1; i++) {
			for (int s = 0; s < numTargetStates; s++) {
				for (int sp = 0; sp < edgeMarginals[i][s].length; sp++) {
					graph.addToCounts(instanceID, i, counts[s],
							scale * edgeMarginals[i][s][sp]);
				}
			}
		}
	}
	
	public static void testModel(SequentialFeatures features, Evaluator eval,
			int[][] labels, int[] devList, double[] parameters) {
		double[] runningAccuracy = new double[3];
		ArrayHelper.deepFill(runningAccuracy, 0.0);
		// compute objective and likelihood
		int numStates = features.numStates;
		int numTargetStates = features.numTargetStates;
		SequentialInference model= new SequentialInference(1000, numStates);
		double[][] edgeScores = new double[numStates][numStates];
		features.computeEdgeScores(edgeScores, parameters);
		for (int i : devList) {
			int length = features.getInstanceLength(i);
			double[][] nodeScores = new double[length][numTargetStates];
			double[][] nodeMarginals = new double[length][numStates];
			double[][][] edgeMarginals =
					new double[length + 1][numStates][numStates];
			int[] decoded = new int[length];
			features.computeNodeScores(i, nodeScores, parameters);
			model.computeMarginals(nodeScores, edgeScores, 
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
}
