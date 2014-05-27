package graph;

import java.util.Arrays;

import optimization.ArrayHelper;
import feature.DynamicSparseVector;
import feature.SparseVector;
import gnu.trove.set.hash.TIntHashSet;

public class GraphRegularizer {
	protected int[][] nodes;   
	protected double[] nodeCounts;
	protected SparseVector[] edges;
	protected int[][] allEdges;
	protected double[] allEdgeWeights;
	public final int numInstances, numEdges, numNodes, numTargetStates;
	
	public GraphRegularizer(int[][] nodes,
			SparseVector[] edges, int numTargetStates) {
		this.nodes = nodes;
		this.edges = edges;
		this.numInstances = nodes.length;
		this.numNodes = edges.length;
		this.numTargetStates = numTargetStates;
		int edgeCount = 0;
		for (int i = 0; i < numNodes; i++) {
			edgeCount += edges[i].length;
		}
		// prepare edge list
		numEdges = edgeCount / 2;
		allEdges = new int[numEdges][2];
		allEdgeWeights = new double[numEdges];
		edgeCount = 0;
		for (int i = 0; i < numNodes; i++) {
			for (int j = 0; j < edges[i].length; j++) {
				if (i < edges[i].indices[j]) {
					allEdges[edgeCount][0] = i;
					allEdges[edgeCount][1] = edges[i].indices[j];
					allEdgeWeights[edgeCount] = edges[i].values[j];
					edgeCount ++;
				}
			}
		}
		// prepare node counts (for normalization)
		/*
		nodeCounts = new double[numNodes];
		Arrays.fill(nodeCounts, 0.0);
		for (int i = 0; i < numInstances; i++) {
			for (int j = 0; j < nodes[i].length; j++) {
				nodeCounts[nodes[i][j]] ++;
			}
		}
		*/
	}
	
	public void setNodeCounts(int[] instList) {
		nodeCounts = new double[numNodes];
		Arrays.fill(nodeCounts, 0.0);
		for (int instanceID : instList) {
			for (int j = 0; j < nodes[instanceID].length; j++) {
				nodeCounts[nodes[instanceID][j]] ++;
			}
		}
	}
	
	public int getInstanceLength(int instanceID) {
		return nodes[instanceID].length;
	}
	
	public void addToCounts(int instanceID, int position, double[] counts,
			double weight) {
		int nodeID = nodes[instanceID][position];
		if (nodeCounts[nodeID] > 0) {
			counts[nodeID] += weight / nodeCounts[nodeID];
		}
	}

	// TODO: fix the naming here ...
	public double computePenalty(int instanceID, int position,
			double[] counts) {
		double result = 0.0;
		int nodeID = nodes[instanceID][position];
		for (int i = 0; i < edges[nodeID].length; i++) {
			int neighborID = edges[nodeID].indices[i]; 
			double weight = edges[nodeID].values[i];
			result += weight * (counts[neighborID] - counts[nodeID]);
		}
		return result;
	}
	
	// compute part of the penalty affected by a single instance update
	public double computeTotalPenalty(int instanceID, double[][] counts) {
		double result = 0.0;
		TIntHashSet visited = new TIntHashSet();
		for (int i = 0; i < nodes[instanceID].length; i++) {
			int nodeID = nodes[instanceID][i];
			for (int j = 0; j < edges[nodeID].length; j++) {
				int neighborID = edges[nodeID].indices[j];
				double weight = edges[nodeID].values[j];
				if (visited.contains(neighborID)) {
					continue;
				}
				for (int k = 0; k < numTargetStates; k++) {
					double diff = counts[k][nodeID] - counts[k][neighborID];
					result += weight * diff * diff;
				}
			}
			visited.add(nodeID);
		}
		return result * 2.0;
	}
	
	public double computeTotalPenalty(double[][] counts) {
		double result = 0.0;
		for (int i = 0; i < numEdges; i++) {
			int n1 = allEdges[i][0];
			int n2 = allEdges[i][1];
			double weight = allEdgeWeights[i];
			for (int j = 0; j < numTargetStates; j++) {
				double diff = counts[j][n1] - counts[j][n2];
				result += weight * diff * diff; 
			}
		}
		return result * 2.0;
	}
	
	// use gold labels to check graph quality
	public double computeTotalPenalty(int[][] labels) {
		double[][] counts = new double[numTargetStates][numNodes];
		ArrayHelper.deepFill(counts, 0.0);
		for (int i = 0; i < labels.length; i++) {
			for (int j = 0; j < labels[i].length; j++) {
				int nodeID = nodes[i][j];
				int stateID = labels[i][j];
				counts[stateID][nodeID] += 1.0 / nodeCounts[nodeID];
			}
		}
		return computeTotalPenalty(counts);
	}
}
