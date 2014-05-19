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
			for (int j = 0; j < arr[i].length; j++) {
				for (int k = 0; k < arr[i][j].length; k++) {
					arr[i][j][k] = filler;
				}
			}
		}
	}
	
	public static void deepCopy(double[][] src, double[][] dest) {
		for (int i = 0; i < src.length; i++) {
			for (int j = 0; j < src[i].length; j++) {
				dest[i][j] = src[i][j];
			}
		}
	}
	
	public static void deepCopy(double[][][] src, double[][][] dest) {
		for (int i = 0; i < src.length; i++) {
			for (int j = 0; j < src[i].length; j++) {
				for (int k = 0; k < src[i][j].length; k++) { 
					dest[i][j][k] = src[i][j][k];
				}
			}
		}
	}
	
	public static double l2Norm(double[] arr) {
		double norm = 0;
		for (int i = 0; i < arr.length; i++) {
			norm += arr[i] * arr[i];
		}
		return norm;
	}
}
