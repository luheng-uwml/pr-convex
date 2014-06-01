package optimization;

public class ArrayHelper {
	public static void deepFill(double[] arr, double filler) {
		for (int i = 0; i < arr.length; i++) {
			arr[i] = filler;
		}
	}
		
	public static void deepFill(double[][] arr, double filler) {
		for (int i = 0; i < arr.length; i++) {
			for (int j = 0; j < arr[i].length; j++) {
				arr[i][j] = filler;
			}
		}
	}
	
	public static void deepFill(double[][][] arr, double filler) {
		for (int i = 0; i < arr.length; i++) {
			if (arr[i] == null) {
				continue;
			}
			for (int j = 0; j < arr[i].length; j++) {
				if (arr[i][j] == null) {
					continue;
				}
				for (int k = 0; k < arr[i][j].length; k++) {
					arr[i][j][k] = filler;
				}
			}
		}
	}
	
	public static void deepCopy(double[] src, double[] dest) {
		for (int i = 0; i < src.length; i++) {
			dest[i] = src[i];
		}
	}
	
	public static void deepCopy(double[][] src, double[][] dest) {
		for (int i = 0; i < src.length; i++) {
			if (src[i] == null) {
				continue;
			}
			for (int j = 0; j < src[i].length; j++) {
				dest[i][j] = src[i][j];
			}
		}
	}
	
	public static void deepCopy(double[][][] src, double[][][] dest) {
		for (int i = 0; i < src.length; i++) {
			if (src[i] == null) {
				continue;
			}
			for (int j = 0; j < src[i].length; j++) {
				if (src[i][j] == null) {
					continue;
				}
				for (int k = 0; k < src[i][j].length; k++) { 
					dest[i][j][k] = src[i][j][k];
				}
			}
		}
	}
	
	public static void addTo(int[] src, int[] dest) {
		for (int i = 0; i < src.length; i++) {
			dest[i] += src[i];
		}
	}
	
	public static void addTo(double[] src, double[] dest) {
		for (int i = 0; i < src.length; i++) {
			dest[i] += src[i];
		}
	}
	
	public static double l1Norm(double[] arr) {
		double norm = 0;
		for (int i = 0; i < arr.length; i++) {
			norm += arr[i];
		}
		return norm;
	}
	
	public static double l2NormSquared(double[] arr) {
		double norm = 0;
		for (int i = 0; i < arr.length; i++) {
			norm += arr[i] * arr[i];
		}
		return norm;
	}
	
	public static double l2NormSquared(double[][] arr) {
		double norm = 0;
		for (int i = 0; i < arr.length; i++) {
			if (arr[i] != null) {
				norm += l2NormSquared(arr[i]);
			}
		}
		return norm;
	}
	
	public static double l2NormSquared(double[][][] arr) {
		double norm = 0;
		for (int i = 0; i < arr.length; i++) {
			if (arr[i] != null) {
				norm += l2NormSquared(arr[i]);
			}
		}
		return norm;
	}
	
	public static double maximum(double[] arr) {
		double maxVal = Double.NEGATIVE_INFINITY;
		for (int i = 0; i < arr.length; i++) {
			if (arr[i] > maxVal) {
				maxVal = arr[i];
			}
		}
		return maxVal;
	}
	
	public static double maximum(double[][] arr) {
		double maxVal = Double.NEGATIVE_INFINITY;
		for (int i = 0; i <arr.length; i++) {
			if (arr[i] != null) {
				maxVal = Math.max(maxVal, maximum(arr[i]));
			}
		}
		return maxVal;
	}
	
	public static double maximum(double[][][] arr) {
		double maxVal = Double.NEGATIVE_INFINITY;
		for (int i = 0; i <arr.length; i++) {
			if (arr[i] != null) {
				maxVal = Math.max(maxVal, maximum(arr[i]));
			}
		}
		return maxVal;
	}
	
	public static double minimum(double[] arr) {
		double minVal = Double.POSITIVE_INFINITY;
		for (int i = 0; i < arr.length; i++) {
			if (arr[i] < minVal) {
				minVal = arr[i];
			}
		}
		return minVal;
	}
	
	public static double minimum(double[][] arr) {
		double minVal = Double.POSITIVE_INFINITY;
		for (int i = 0; i <arr.length; i++) {
			if (arr[i] != null) {
				minVal = Math.min(minVal, minimum(arr[i]));
			}
		}
		return minVal;
	}
	
	public static double minimum(double[][][] arr) {
		double minVal = Double.POSITIVE_INFINITY;
		for (int i = 0; i <arr.length; i++) {
			if (arr[i] != null) {
				minVal = Math.min(minVal, minimum(arr[i]));
			}
		}
		return minVal;
	}

	public static void normalize(double[] arr) {
		double norm = Math.sqrt(l2NormSquared(arr));
		if (norm == 0) {
			return;
		}
		for (int i = 0; i < arr.length; i++) {
			arr[i] /= norm;
		}
	}	
}
