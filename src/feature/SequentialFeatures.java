package feature;

public class SequentialFeatures {
	SparseVector[][] nodeFeatures; // instances x  postions  
	SparseVector[][] edgeFeatures; // states x states
	public final int numStates, numTargetStates, numInstances, numNodeFeatures,
			numEdgeFeatures, numAllFeatures, S0, SN;
	
	public SequentialFeatures(SparseVector[][] nodeFeatures,
			SparseVector[][] edgeFeatures,
			int numNodeFeatures, int numEdgeFeatures) {
		this.nodeFeatures = nodeFeatures;
		this.edgeFeatures = edgeFeatures;
		this.numInstances = nodeFeatures.length;
		this.numStates = edgeFeatures.length;
		this.numTargetStates = numStates - 2; // excluding dummy states
		this.S0 = numStates - 2;
		this.SN = numStates - 1;
		this.numNodeFeatures = numNodeFeatures;
		this.numEdgeFeatures = numEdgeFeatures;
		this.numAllFeatures = numEdgeFeatures + numNodeFeatures *
				numTargetStates;
		System.out.println("states:\t" + this.numStates);
		System.out.println("features:\t" + this.numAllFeatures);
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
		// bias + edge features + node features * (previous states)
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
}
