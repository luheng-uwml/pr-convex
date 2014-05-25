package optimization;

import feature.SequentialFeatures;

public class OptimizationHelper {
	public static void computeHardCounts(SequentialFeatures features,
			int[][] labels, int instanceID, double[] counts) {
		int length = features.getInstanceLength(instanceID);
		for (int i = 0; i < length; i++) {
			int s = labels[instanceID][i];
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
}
