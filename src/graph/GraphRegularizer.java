package graph;

import java.util.Arrays;

import data.CountDictionary;
import optimization.ArrayHelper;
import feature.SparseVector;
import gnu.trove.set.hash.TIntHashSet;

public class GraphRegularizer {
	protected int[][] nodes;   
	protected double[] nodeCounts;
	protected SparseVector[] edges;
	protected int[][] allEdges;
	protected double[] allEdgeWeights;
	public final int numEdges, numNodes, numTargetStates;
	
	protected GraphRegularizer(int numTargetStates) {
		this.numTargetStates = numTargetStates;
		this.numEdges = 0;
		this.numNodes = 0;
	}
	
	public GraphRegularizer(int[][] nodes,
			SparseVector[] edges, int numTargetStates) {
		this.nodes = nodes;
		this.edges = edges;
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
		nodeCounts = new double[numNodes];
		Arrays.fill(nodeCounts, 0.0);
		for (int i = 0; i < nodes.length; i++) {
			for (int j = 0; j < nodes[i].length; j++) {
				nodeCounts[nodes[i][j]] ++;
			}
		}
	}
	
	public void setNodeCounts(int[] instList) {
		Arrays.fill(nodeCounts, 0.0);
		for (int instanceID : instList) {
			for (int j = 0; j < nodes[instanceID].length; j++) {
				nodeCounts[nodes[instanceID][j]] ++;
			}
		}
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
			if (i >= nodes.length) {
				continue;
			}
			for (int j = 0; j < labels[i].length; j++) {
				int nodeID = nodes[i][j];
				int stateID = labels[i][j];
				counts[stateID][nodeID] += 1.0 / nodeCounts[nodeID];
			}
		}
		return computeTotalPenalty(counts);
	}
	
	// TODO: estimate graph quality using gold labels
	public void validate(int[][] labels, CountDictionary labelDict,
			CountDictionary ngramDict) {
		double[][] counts = new double[numTargetStates][numNodes];
		int[] dominantLabels = new int[numNodes];
		double[] purity = new double[numNodes];
		int badEdges = 0;
		double totalPenalty = 0.0, avgPurity = 0.0;
		ArrayHelper.deepFill(counts, 0.0);
		ArrayHelper.deepFill(purity, 0.0);
		for (int i = 0; i < labels.length; i++) {
			if (i >= nodes.length) {
				continue;
			}
			for (int j = 0; j < labels[i].length; j++) {
				int nodeID = nodes[i][j];
				int stateID = labels[i][j];
				counts[stateID][nodeID] += 1.0 / nodeCounts[nodeID];
			}
		}
		// compute dominant labels
		for (int i = 0; i < numNodes; i++) {
			double nodeSum = counts[0][i];
			int maxLabel = 0; 
			for (int j = 1; j < numTargetStates; j++) {
				nodeSum += counts[j][i];
				if (counts[j][i] > counts[maxLabel][i]) {
					maxLabel = j;
				}
			}
			if (Math.abs(nodeSum - 1.0) > 1e-5) {
				System.out.println("unnormalized node!!\t" + nodeSum);
			}
			dominantLabels[i] = maxLabel;
			purity[i] = counts[maxLabel][i];
			avgPurity += purity[i];
		}
		avgPurity /= numNodes;
		for (int i = 0; i < numEdges; i++) {
			int n1 = allEdges[i][0];
			int n2 = allEdges[i][1];
			double weight = allEdgeWeights[i];
			for (int j = 0; j < numTargetStates; j++) {
				double diff = counts[j][n1] - counts[j][n2];
				totalPenalty += weight * diff * diff; 
			}
			if (dominantLabels[n1] != dominantLabels[n2]) {
				badEdges ++;
				// print bad edges
				System.out.println(ngramDict.getString(n1) + "\t" +
							ngramDict.getString(n2) + "\t" + weight + "\t" +
							labelDict.getString(dominantLabels[n1]) + "\t" +
							labelDict.getString(dominantLabels[n2]));
			}
		}
		System.out.println(String.format("Total penalty::\t%.5f",
				2 * totalPenalty));
		System.out.println(String.format("Bad edges::\t%d (%.5f%%)",
				badEdges, 100.0 * badEdges / numEdges));
		System.out.println(String.format("Averaged purity::\t%.5f",
				avgPurity));
	}
}
