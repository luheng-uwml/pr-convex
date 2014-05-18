package feature;

import gnu.trove.map.hash.TIntDoubleHashMap;

public class DynamicSparseVector {
	public TIntDoubleHashMap vmap;
	
	public DynamicSparseVector() {
		vmap = new TIntDoubleHashMap();
	}
	
	public void add(int idx, double val) {
		vmap.adjustOrPutValue(idx, val, val);
	}
	
	public double get(int idx) {
		return vmap.get(idx);
	}
	
	public int size() {
		return vmap.size();
	}
}
