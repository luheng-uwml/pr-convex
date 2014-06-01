package graph;

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
	
	protected static double dotProduct(double[] x, double[] y) {
		double result = 0;
		for (int i = 0; i < x.length; i++) {
			result += x[i] * y[i];
		}
		return result;
	}
	
	protected class EdgeBuilderThread extends Thread {
		int start, end, id;
		boolean finished;

		EdgeBuilderThread(int id, int start, int end) {
			this.id = id;
			this.start = start;
			this.end = end;
			this.finished = false;
			print("Thread " + id + " processing nodes " + start + " to "
					+ end + "\n");
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
