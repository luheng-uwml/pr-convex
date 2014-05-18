package feature;

import java.util.Arrays;

public class SparseVector {
	public int[] indices;
	public double[] values;
	public int length;
	
	public SparseVector(DynamicSparseVector vec) {
		length = vec.size();
		indices = vec.vmap.keys();
		values = new double[length];
		Arrays.sort(indices);
		for (int i = 0; i < length; i++) {
			values[i] = vec.get(indices[i]);
		}
	}
	
	public int size() {
		return length;
	}
	/*
	public SparseVector(TIntArrayList indices, TDoubleArrayList values) {
		this.indices = indices.toArray();
		this.values = values.toArray();
		this.length = this.indices.length;
	}
	*/
	
	public double dotProduct(double[] weights) {
		double result = 0;
		for (int i = 0; i < length; i ++) {
			result += values[i] * weights[indices[i]];
		}
		return result;
	}
}
