package graph;

import graph.KNNGraphConstructor.EdgeBuilderThread;

public class WordVectorGraphConstructor extends KNNGraphConstructor {
	double[][] features;
	
	public WordVectorGraphConstructor(double[][] features, int numNeighbors,
			boolean mutualKNN, double similarityThreshold, int numThreads) {
		super();
		this.features = features;
		this.numNodes = features.length;
		this.numNeighbors = numNeighbors;
		this.mutualKNN = mutualKNN;
		this.similarityThreshold = similarityThreshold;
		this.numThreads = numThreads;
	}
	
	public void run() {
		System.out.println(String.format("Starting to build graph with " +
				"%d nodes, and %d neighbors with %s KNN method. ",
				numNodes, numNeighbors, (mutualKNN ? "mutual" : "symmetric")));
		edges = new EdgeList[numNodes];
		for(int i = 0; i < numNodes; i++) {
			edges[i] = new EdgeList(numNeighbors);
		}
		int batchSize = numNodes / numThreads;
		EdgeBuilderThread[] threads = new DenseEdgeBuilderThread[numThreads];
		for (int i = 0; i < numThreads; i++) {
			int start = i * batchSize;
			int end = (i == numThreads - 1 ? numNodes : start + batchSize);
			threads[i] = new DenseEdgeBuilderThread(i, start, end);
			threads[i].start();
		}
		for (int i = 0; i < numThreads; i++) {
			try {
				threads[i].join();
			} catch (InterruptedException e) { }
		}
		symmetrifyGraph();
	}
	
	protected static double dotProduct(double[] x, double[] y) {
		double result = 0;
		for (int i = 0; i < x.length; i++) {
			result += x[i] * y[i];
		}
		return result;
	}
	
	protected class DenseEdgeBuilderThread extends
		KNNGraphConstructor.EdgeBuilderThread {
		
		DenseEdgeBuilderThread(int id, int start, int end) {
			super(id, start, end);
		}

		@Override
		public void run() {
			long startTime = System.currentTimeMillis();
			for (int i = start; i < end; i++) {
				for (int j = 0; j < numNodes; j++) {
					if (i != j) {
						double sim = dotProduct(features[i], features[j]);
						if (sim >= similarityThreshold) {
							edges[i].add(new Edge(j, sim));
						}
					}
				}
				if (i > start && (i - start) % 1000 == 0) {
					int offset = (i - start);
					long timeElapsedMin = 1 + (System.currentTimeMillis()
							- startTime) / 60000;
					long nodesPerMin = offset / timeElapsedMin;
					print("Thread" + id + "\t::" + offset
							+ " nodes finished, " + (end -start - offset)
							+ " nodes to go. " + nodesPerMin
							+ " nodes per minute\n");
				}
			}
		}
	}
}
