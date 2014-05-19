package inference;

public class LatticeHelper {

	public static int getMaxIndex(double[] p) {
		int maxi = 0;
		for (int i = 1; i < p.length; i++) {
			if (p[maxi] < p[i]) {
				maxi = i;
			}
		}
		return maxi;
	}
	
	public static double logsum(double loga, double logb) {		
		if (Double.isInfinite(loga)) {
			return logb;
		}
		if (Double.isInfinite(logb)) {
			return loga;
		}
		if (loga > logb) { 
			return Math.log1p(Math.exp(logb - loga)) + loga;
		}
		else { 
			return Math.log1p(Math.exp(loga - logb)) + logb;
		}
	}
	
	public static double logsum(double[] tosum, int length) {	
		if (length == 1) {
			return tosum[0];
		}			
		int idx = 0;
		for (int i = 1; i < length; i++) {
			if (tosum[i] > tosum[idx]) {
				idx = i;
			}
		}
		double maxx = tosum[idx];
		double sumexp = 0;
		for (int i = 0; i < length; i++) {
			if (i != idx) {
				sumexp += Math.exp(tosum[i] - maxx);
			}
		}
		return Math.log1p(sumexp) + maxx;
	}
}

