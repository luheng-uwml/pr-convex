package regularization;

import feature.SparseVector;

public class SimilarityRegularizationFeatures {
	protected SparseVector[][] features;   
	protected SparseVector[] edges;
	public final int numInstances, numEdges, numFeatures;
	
	public SimilarityRegularizationFeatures(SparseVector[][] features,
			SparseVector[] edges, int numFeatures) {
		this.features = features;
		this.edges = edges;
		this.numInstances = features.length;
		this.numFeatures = numFeatures;
		this.numEdges = edges.length;
	}
	
	// weight associated with a particular state
	public double computeScore(int instanceID, int position, double[] weights) {
		return features[instanceID][position].dotProduct(weights);
	}
	
	public void addToCounts(int instanceID, int position, double[] counts,
			double weight) {
		features[instanceID][position].addTo(counts, weight);
	}
	
	
}
