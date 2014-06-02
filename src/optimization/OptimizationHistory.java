package optimization;

import gnu.trove.list.array.TDoubleArrayList;

import java.util.HashMap;

public class OptimizationHistory {
	public int totalIterations;
	public HashMap<String, TDoubleArrayList> history;
	
	public OptimizationHistory() {
		totalIterations = 0;
		history = new HashMap<String, TDoubleArrayList>();
	}
	
	public void add(int iter, String label, double val) {
		TDoubleArrayList hist = null;
		if (iter == 0) {
			hist = new TDoubleArrayList();
			history.put(label, hist);
		} else {
			hist = history.get(label);
		}
		hist.add(val);
		assert (hist.size() == iter + 1);
	}
}
