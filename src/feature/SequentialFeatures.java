package feature;

import data.CountDictionary;

public class SequentialFeatures {
	protected CountDictionary nodeFeatureDict, edgeFeatureDict, stateDict;
	protected SparseVector[][] nodeFeatures; // instances x  postions  
	protected SparseVector[][] edgeFeatures; // states x states
	public final int numStates, numTargetStates, numInstances, numNodeFeatures,
			numEdgeFeatures, numAllFeatures, S0, SN;
	
	public SequentialFeatures(SparseVector[][] nodeFeatures,
			SparseVector[][] edgeFeatures,
			int numNodeFeatures, int numEdgeFeatures,
			CountDictionary nodeFeatureDict, CountDictionary edgeFeatureDict,
			CountDictionary stateDict) {
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
		this.nodeFeatureDict = nodeFeatureDict;
		this.edgeFeatureDict = edgeFeatureDict;
		this.stateDict = stateDict;
		System.out.println("states:\t" + this.numStates);
		System.out.println("features:\t" + this.numAllFeatures);
	}
	
	public int getInstanceLength(int instanceID) {
		return nodeFeatures[instanceID].length;
	}

	public String getFeatureName(int featureID) {
		if (featureID < numEdgeFeatures) {
			return edgeFeatureDict.getString(featureID);
		} else {
			int featID = (featureID - numEdgeFeatures) % numNodeFeatures;
			int stateID = (featureID - numEdgeFeatures - featID) /
					numNodeFeatures;
			return nodeFeatureDict.getString(featID) + " + " +
					stateDict.getString(stateID); 
		}
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
	
	public void addEdgeToCounts(int instanceID, int stateID, int prevStateID,
			double[] counts, double weight) {
		if (Double.isInfinite(weight) || Double.isNaN(weight)) {
			return;
		}
		edgeFeatures[stateID][prevStateID].addTo(counts, weight);
	}
	
	public void addNodeToCounts(int instanceID, int position, int stateID,
			double[] counts, double weight) {
		if (Double.isInfinite(weight) || Double.isNaN(weight) ||
			stateID >= numTargetStates) {
			return;
		}
		int offset = numEdgeFeatures + stateID * numNodeFeatures;
		nodeFeatures[instanceID][position].addTo(counts, weight, offset);
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
			totalLength += (length - 1);
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
