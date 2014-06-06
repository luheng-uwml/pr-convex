package feature;

import java.util.Arrays;

public class SparseVector {
	public int[] indices;
	public double[] values;
	public final int length;
	
	public SparseVector(DynamicSparseVector vec) {
		length = vec.size();
		indices = vec.vmap.keys();
		values = new double[length];
		Arrays.sort(indices);
		for (int i = 0; i < length; i++) {
			values[i] = vec.get(indices[i]);
		}
	}
	
	// TODO: sort indices here
	public SparseVector(int[] indices, double[] values) {
		this.length = indices.length;
		this.indices = new int[length];
		this.values = new double[length];
		for (int i = 0; i < length; i++) {
			this.indices[i] = indices[i];
			this.values[i] = values[i];
		}
	}
	
	public SparseVector() {
		length = 0;
	}

	public int size() {
		return length;
	}
	
	public void normalize() {
		double norm = 0;
		for (int i = 0; i < length; i++) {
			norm += values[i] * values[i];
		}
		if (norm == 0) {
			return;
		}
		norm = Math.sqrt(norm);
		for (int i = 0; i < length; i++) {
			values[i] /= norm;
		}
	}
	
	// Compute dot product for sparse vectors, I think this might be fast ...
	public double dotProduct(SparseVector other) {
		if(length == 0 || other.length == 0 ||
			indices[0] > other.indices[other.length-1] ||
			other.indices[0] > indices[length-1]) { 
			return 0.0;
		}
		double result = 0;
		for (int i = 0, j = 0; i < length && j < other.length; ) {
			if (indices[i] == other.indices[j]) {
				result += values[i] * other.values[j];
				i ++;
				j ++;
			} else if (indices[i] < other.indices[j]) {
				i ++;
			}
			else {
				j ++;
			}
		}
		return result;
	}
	
	public double dotProduct(double[] weights) {
		double result = 0;
		for (int i = 0; i < length; i ++) {
			result += values[i] * weights[indices[i]];
		}
		return result;
	}
	
	public double dotProduct(double[] weights, int offset) {
		double result = 0;
		for (int i = 0; i < length; i ++) {
			result += values[i] * weights[indices[i] + offset];
		}
		return result;
	}
	
	public void addTo(double[] counts, double weight) {
		for (int i = 0; i < length; i ++) {
			counts[indices[i]] += values[i] * weight;
		}
	}
	
	public void addTo(double[] counts, double weight, int offset) {
		for (int i = 0; i < length; i ++) {
			counts[indices[i] + offset] += values[i] * weight;
		}
	}
}
