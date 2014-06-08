package optimization;

public interface AbstractOptimizer {

	public void optimize();
	
	public OptimizationHistory getOptimizationHistory();
	
	public int[][] getPrediction();
}
