package graph;

public class DummyGraphRegularizer extends GraphRegularizer {
	
	public DummyGraphRegularizer(int numTargetStates) {
		super(numTargetStates);
	}
	
	@Override
	public void setNodeCounts(int[] instList) {
		// do nothing
	}
	
	@Override
	public void addToCounts(int instanceID, int position, double[] counts,
			double weight) {
		// do nothing
	}

	@Override
	public double computePenalty(int instanceID, int position,
			double[] counts) {
		return 0.0;
	}
	
	@Override
	public double computeTotalPenalty(int instanceID, double[][] counts) {
		return 0.0;
	}
	
	@Override
	public double computeTotalPenalty(double[][] counts) {
		return 0.0;
	}
	
	@Override
	public double computeTotalPenalty(int[][] labels) {
		return 0.0;
	}
}
