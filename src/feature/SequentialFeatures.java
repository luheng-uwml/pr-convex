package feature;

public class SequentialFeatures {
	SparseVector[][] nodeFeatures; // instances x  postions  
	SparseVector[][] edgeFeatures; // states x states
	int numStates, numTargetStates, numNodeFeatures, numEdgeFeatures;
	
	public SequentialFeatures(SparseVector[][] nodeFeatures,
			SparseVector[][] edgeFeatures,
			int numNodeFeatures, int numEdgeFeatures) {
		this.nodeFeatures = nodeFeatures;
		this.edgeFeatures = edgeFeatures;
		this.numStates = edgeFeatures.length;
		this.numTargetStates = numStates - 2; // excluding dummy states
		this.numNodeFeatures = numNodeFeatures;
		this.numEdgeFeatures = numEdgeFeatures;
	}
	
	public void computeScores(int instanceID, double[][] nodeScores,
			double[][] edgeScores, double[] weights) {
		int length = nodeFeatures[instanceID].length;
		for (int i = 0; i < length; i++) {
			for (int j = 0; j < numTargetStates; j++) {
				nodeScores[i][j] = computeNodeScore(instanceID, i, j, weights);
			}
		}
		for (int i = 0; i < numStates; i++) { 
			for (int j = 0; j < numStates; j++) {
				edgeScores[i][j] = computeEdgeScore(i, j, weights);
			}
		}
	}
	
	public double computeScore(int instanceID, int position, int stateID,
			int prevStateID, double[] weights) {
		return weights[0] +
				computeNodeScore(instanceID, position, stateID, weights) +
				computeEdgeScore(stateID, prevStateID, weights);
	}
	
	public double computeNodeScore(int instanceID, int position, int stateID,
			double[] weights) {
		// bias + edge features + node features * (previous states)
		int offset = 1 + numEdgeFeatures + stateID * numNodeFeatures;
		return nodeFeatures[instanceID][position].dotProduct(weights, offset);
	}
	
	public double computeEdgeScore(int stateID, int prevStateID,
			double[] weights) {
		return edgeFeatures[stateID][prevStateID].dotProduct(weights);
	}
	
	public void addToSoftCounts(int instanceID, int position, int stateID,
			int prevStateID, double[] counts, double weight) {
		if (Double.isInfinite(weight) || Double.isNaN(weight)) {
			return;
		}
		int offset = 1 + numEdgeFeatures + stateID * numNodeFeatures;
		nodeFeatures[instanceID][position].addTo(counts, weight, offset);
		edgeFeatures[stateID][prevStateID].addTo(counts, weight);
	}
}
