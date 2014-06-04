package inference;

public class SequentialInference {
	public double[][] alpha, beta;
	private int S0, SN, numTargetStates;
	private double[] sTemp;
	
	public SequentialInference(int maxInstanceLength, int numStates) {
		alpha = new double[maxInstanceLength + 1][numStates];
		beta = new double[maxInstanceLength + 1][numStates];
		sTemp = new double[numStates + 1];
		//this.numStates = numStates;
		this.numTargetStates = numStates - 2;
		// FIXME: this is hacky, fix this.
		this.S0 = numStates - 2;
		this.SN = numStates - 1;
	}
	
	public double computeLabelLikelihood(double[][] nodeScores,
			double[][] edgeScores, double logNorm, int[] labels) {
		int length = labels.length;
		double logLikelihood = - logNorm + nodeScores[0][labels[0]] +
				edgeScores[labels[0]][S0] + edgeScores[SN][labels[length - 1]];
		for (int i = 1; i < length; i++) {
			logLikelihood += nodeScores[i][labels[i]] +
					edgeScores[labels[i]][labels[i-1]];
		}
		return logLikelihood;
	}
	
	/* The forward-backword algorithm
	 * 		nodeScores [position][possible target tag]
	 * 		edgeScoers [current tag][previous tag]
	 */
	public double computeMarginals(double[][] nodeScores, double[][] edgeScores,
			double[][] nodeMarginals, double[][][] edgeMarginals) {
		int length = nodeScores.length;
		for (int s = 0; s < numTargetStates; s++) { 
			alpha[0][s] = edgeScores[s][S0] + nodeScores[0][s]; 
		}
		int tlen;
		for (int i = 1; i < length; i++) {
			for (int s = 0; s < numTargetStates; s++) {
				tlen = 0;
				for (int sp = 0; sp < numTargetStates; sp++) {  
					sTemp[tlen++] = alpha[i-1][sp] + edgeScores[s][sp] +
							nodeScores[i][s];
				}
				alpha[i][s] = LatticeHelper.logsum(sTemp, tlen);
			}
		}
		tlen = 0;
		for (int sp = 0; sp < numTargetStates; sp++) {
			sTemp[tlen++] = alpha[length-1][sp] + edgeScores[SN][sp];
		}
		alpha[length][SN] = LatticeHelper.logsum(sTemp, tlen);
		double logNorm = alpha[length][SN];	
		beta[length][SN] = 0;
		for (int sp = 0; sp < numTargetStates; sp++) {
			beta[length-1][sp] = edgeScores[SN][sp];
		}
		for (int i = length - 1; i > 0; i--) { 
			for (int sp = 0; sp < numTargetStates; sp++) {
				tlen = 0;
				for (int s = 0; s < numTargetStates; s++) { 
					sTemp[tlen++] = beta[i][s] + edgeScores[s][sp] +
							nodeScores[i][s];
				}
				beta[i-1][sp] = LatticeHelper.logsum(sTemp, tlen);
			}
		}
		if (nodeMarginals != null) {
			for (int i = 0; i < length; i++) {
				for (int s = 0; s < numTargetStates; s++) {
					nodeMarginals[i][s] = Math.exp(alpha[i][s] + beta[i][s] -
						logNorm);
				}	
			}
		}
		for (int s = 0; s < numTargetStates; s++) {
			edgeMarginals[0][s][S0] = Math.exp(alpha[0][s] + beta[0][s]
					- logNorm);
			edgeMarginals[length][SN][s] = Math.exp(alpha[length-1][s] +
					beta[length-1][s] - logNorm);
			for (int i = 1; i < length; i++) {
				for (int sp = 0; sp < numTargetStates; sp++) {
					edgeMarginals[i][s][sp] = Math.exp(
						alpha[i-1][sp] + beta[i][s] + edgeScores[s][sp] +
						nodeScores[i][s] - logNorm);
				}
			}
		}
		return logNorm;
	}
	
	// TODO: implement sanity check:
	// edge marginals of a position should sum up to one
	public void sanityCheck(double[][][] edgeMarginals) {
		
	}
	
	public void posteriorDecoding(double[][] nodeMarginals, int[] prediction) {
		for (int i = 0; i < nodeMarginals.length; i++) {
			prediction[i] = LatticeHelper.getMaxIndex(nodeMarginals[i]);
		}
	}
	
	public void viterbiDecoding(double[][] nodeScores, double[][] edgeScores,
			int[] prediction) {
		int length = nodeScores.length;
		double[][] best = new double[length][numTargetStates];
		int[][] prev = new int[length][numTargetStates];
		for (int j = 0; j < numTargetStates; j++) {
			best[0][j] = edgeScores[j][S0] + nodeScores[0][j];
			prev[0][j] = S0;
		}
		for(int i = 1; i < length; i++) {
			for (int j = 0; j < numTargetStates; j++) {
				best[i][j] = Double.NEGATIVE_INFINITY;
				prev[i][j] = -1;
				for (int k = 0; k < numTargetStates; k++) {
					double r = best[i-1][k] + edgeScores[j][k] +
							   nodeScores[i][j];
					if(r > best[i][j]) {
						best[i][j] = r;
						prev[i][j] = k;
					}
				}
			}
		}
		prediction[length - 1] = LatticeHelper.getMaxIndex(best[length-1]);
		for(int i = length - 1; i > 0; i--) {
			prediction[i-1] = prev[i][prediction[i]];
		}
	}
	
	
	public double computeEntropy(double[][] nodeScores, double[][] edgeScores,
			double[][][] edgeMarginals, double logNorm) {
		double entropy = logNorm;
		int length = nodeScores.length;
		for (int s = 0; s < numTargetStates; s++) {
			entropy -= edgeMarginals[0][s][S0] *
					(nodeScores[0][s] + edgeScores[s][S0]);
			entropy -= edgeMarginals[length][SN][s] * edgeScores[SN][s];
		}
		for (int i = 1; i < length; i++) {
			for (int s = 0; s < numTargetStates; s++) {
				for (int sp = 0; sp < numTargetStates; sp++) {
					entropy -= edgeMarginals[i][s][sp] *
							(nodeScores[i][s] + edgeScores[s][sp]);
				}
			}
		}
		return entropy;
	}
}
