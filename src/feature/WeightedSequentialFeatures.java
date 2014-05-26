package feature;

public class WeightedSequentialFeatures extends SequentialFeatures {
	protected double[] discountRatio; // discount
	public WeightedSequentialFeatures(SparseVector[][] nodeFeatures,
			SparseVector[][] edgeFeatures,
			int numNodeFeatures, int numEdgeFeatures) {
		super(nodeFeatures, edgeFeatures, numNodeFeatures, numEdgeFeatures);
		discountRatio = new double[numAllFeatures];
	}
	
	public int getInstanceLength(int instanceID) {
		return nodeFeatures[instanceID].length;
	}
	
	public void computeNodeScores(int instanceID, double[][] nodeScores,
			double[] weights) {
		int length = nodeFeatures[instanceID].length;
		for (int i = 0; i < length; i++) {
			for (int j = 0; j < numTargetStates; j++) {
				nodeScores[i][j] = computeNodeScore(instanceID, i, j, weights);
			}
		}
	}
	
	public void computeEdgeScores(double[][] edgeScores, double[] weights) {
		for (int i = 0; i < numStates; i++) { 
			for (int j = 0; j < numStates; j++) {
				edgeScores[i][j] = computeEdgeScore(i, j, weights);
			}
		}
	}
	
	public double computeScore(int instanceID, int position, int stateID,
			int prevStateID, double[] weights) {
		return computeNodeScore(instanceID, position, stateID, weights) +
				computeEdgeScore(stateID, prevStateID, weights);
	}
	
	public double computeNodeScore(int instanceID, int position, int stateID,
			double[] weights) {
		// edge features + node features * (previous states)
		// not defined for dummy states
		int offset = numEdgeFeatures + stateID * numNodeFeatures;
		return nodeFeatures[instanceID][position].dotProduct(weights, offset);
	}
	
	public double computeEdgeScore(int stateID, int prevStateID,
			double[] weights) {
		SparseVector fvec = edgeFeatures[stateID][prevStateID];
		return (fvec == null) ? 0 : fvec.dotProduct(weights);
	}
	
	public void addToCounts(int instanceID, int position, int stateID,
			int prevStateID, double[] counts, double weight) {
		if (Double.isInfinite(weight) || Double.isNaN(weight)) {
			return;
		}
		if (stateID < numTargetStates) {
			int offset = numEdgeFeatures + stateID * numNodeFeatures;
			nodeFeatures[instanceID][position].addTo(counts, weight, offset);
		}
		edgeFeatures[stateID][prevStateID].addTo(counts, weight);
	}
	
	protected void countEdgeFeature(int stateID, int prevStateID, int scale,
			int[] counts) {
		SparseVector fvec = edgeFeatures[stateID][prevStateID];
		if (fvec != null) {
			for (int i : fvec.indices) {
				counts[i] += scale;
			}
		}
	}
	
	protected void countNodeFeature(int instanceID, int position, int stateID,
			int scale, int[] counts) {
		int offset = numEdgeFeatures + stateID * numNodeFeatures;
		SparseVector fvec = nodeFeatures[instanceID][position];
		for (int i : fvec.indices) {
			counts[i + offset] += scale;
		}
	}
	
	// count feature frequency:
	public void countFeatures(int[] instList, int[] counts) {
		int totalLength = 0;
		for (int instanceID : instList) {
			int length = getInstanceLength(instanceID);
			totalLength += length;
			for (int i = 0; i < length; i++) {
				for (int j = 0; j < numTargetStates; j++) {
					countNodeFeature(instanceID, i, j, 1, counts);
				}
			}
		}
		for (int i = 0; i < numTargetStates; i++) {
			countEdgeFeature(i, S0, instList.length, counts);
			countEdgeFeature(SN, i, instList.length, counts);
			for (int j = 0; j < numTargetStates; j++) {
				countEdgeFeature(i, j, totalLength, counts);
			}
		}
	}
}
