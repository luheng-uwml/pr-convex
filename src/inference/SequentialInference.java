package inference;

public class SequentialInference {
	public double[][] alpha, beta;
	public int S0, SN, numTargetStates;
	private double[] sTemp;
	
	public SequentialInference(int maxInstanceLength, int numStates) {
		alpha = new double[maxInstanceLength + 1][numStates];
		beta = new double[maxInstanceLength + 1][numStates];
		sTemp = new double[numStates + 1];
		//this.numStates = numStates;
		this.numTargetStates = numStates - 2;
		// FIXME: this is hacky, fix this.
		S0 = numStates - 2;
		SN = numStates - 1;
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
			double[][] nodeMarginal, double[][][] edgeMarginal) {
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
		for (int i = 0; i < length; i++) {
			for (int s = 0; s < numTargetStates; s++) {
				nodeMarginal[i][s] = Math.exp(alpha[i][s] + beta[i][s] -
					logNorm);
			}	
		}
		for (int s = 0; s < numTargetStates; s++) {
			edgeMarginal[0][s][S0] = nodeMarginal[0][s];
			for (int i = 1; i < length; i++) {
				for (int sp = 0; sp < numTargetStates; sp++) {
					edgeMarginal[i][s][sp] = Math.exp(
						alpha[i-1][sp] + beta[i][s] + edgeScores[s][sp] +
						nodeScores[i][s] - logNorm);
				}
			}
		}
		return logNorm;
	}
	
	public void posteriorDecoding(double[][] nodeMarginals, int[] prediction) {
		for (int i = 0; i < nodeMarginals.length; i++) {
			prediction[i] = LatticeHelper.getMaxIndex(nodeMarginals[i]);
		}
	}
	
	public void viterbiDecoding(double[][] nodeScores, double[][] edgeScores,
			int[] prediction) {
		// TODO: viterbi decoding
	}
	
	public double computeEntropy(double[][] nodeScores, double[][] edgeScores,
			double[][][] edgeMarginal, double logNorm) {
		double entropy = logNorm;
		int length = nodeScores.length;
		for (int s = 0; s < numTargetStates; s++) {
			entropy -= edgeMarginal[0][s][S0] *
					(nodeScores[0][s] + edgeScores[s][S0]);
			entropy -= edgeMarginal[length][SN][s] * edgeScores[SN][s];
		}
		for (int i = 1; i < length; i++) {
			for (int s = 0; s < numTargetStates; s++) {
				for (int sp = 0; sp < numTargetStates; sp++) {
					entropy -= edgeMarginal[i][s][sp] *
							(nodeScores[i][s] + edgeScores[s][sp]);
				}
			}
		}
		return entropy;
	}
	
}
