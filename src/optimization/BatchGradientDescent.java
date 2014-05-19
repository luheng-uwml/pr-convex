package optimization;

import inference.SequentialInference;
import feature.SequentialFeatures;
import gnu.trove.list.array.TIntArrayList;

public class BatchGradientDescent {
	SequentialFeatures features;
	SequentialInference model;
	int[][] labels;
	int[] trainList, devList;
	double[] parameters, empiricalCounts, gradient;
	double lambda, objective, initialStepSize, stepSize;
	int numFeatures, maxNumIterations, numStates, numTargetStates;
	
	public BatchGradientDescent(SequentialFeatures features, int[][] labels,
			double lambda, double initialStepSize, int maxNumIterations) {
		this.features = features;
		this.labels = labels;
		this.lambda = lambda;
		this.initialStepSize = initialStepSize;
		this.maxNumIterations = maxNumIterations;
		initialize();
	}
	
	public void initialize() {
		numFeatures = features.getNumFeatures();
		numStates = features.getNumStates();
		numTargetStates = numStates - 2;
		TIntArrayList tempTrainList = new TIntArrayList();
		TIntArrayList tempDevList = new TIntArrayList();
		for (int i = 0; i < labels.length; i++) {
			if (labels[i] != null && labels[i].length > 0) {
				tempTrainList.add(i);
			} else {
				tempDevList.add(i);
			}
		}
		trainList = tempTrainList.toArray();
		devList = tempDevList.toArray();
		model= new SequentialInference(1000, numStates);
		parameters = new double[numFeatures];
		gradient = new double[numFeatures];
		empiricalCounts = new double[numFeatures];
		ArrayHelper.deepFill(parameters, 0.0);
		ArrayHelper.deepFill(empiricalCounts, 0.0);
		// compute empirical counts from labels
		for (int i : trainList) {
			computeHardCounts(i, empiricalCounts);
		}
	}
	
	public void optimize() {
		stepSize = initialStepSize;
		double prevObjective = Double.POSITIVE_INFINITY;
		for (int iteration = 0; iteration < maxNumIterations; iteration ++) {
			updateObjectiveAndGradient();
			System.out.println("OBJ::\t" + objective + "\tPREV::\t" +
					prevObjective + "\tGRAD::\t" +
					ArrayHelper.l2Norm(gradient));
			if (Math.abs(objective - prevObjective) / prevObjective < 1e-5) {
				System.out.println("succeed!");
				break;
			}
			for (int i = 0; i < numFeatures; i++) {
				parameters[i] -= stepSize * gradient[i];
			}
			prevObjective = objective;
		}
	}
	
	/* Log-linear objective
	 * - \sum_x log (y* | x*) + \labmda / 2 || theta ||^2 
	 */
	private void updateObjectiveAndGradient() {
		// initialize new gradient value
		for (int i = 0; i < numFeatures; i++) {
			gradient[i] = lambda * parameters[i] - empiricalCounts[i]; 
		}
		// compute objective and likelihood
		double negLikelihood = 0;
		double[][] edgeScores = new double[numStates][numStates];
		features.computeEdgeScores(edgeScores, parameters);
		for (int i : trainList) {
			int length = features.getInstanceLength(i);
			double[][] nodeScores = new double[length][numTargetStates];
			double[][] nodeMarginals = new double[length + 1][numStates];
			double[][][] edgeMarginals =
					new double[length + 1][numStates][numStates];
			double logNorm = Double.NEGATIVE_INFINITY;
			features.computeNodeScores(i, nodeScores, parameters);
			logNorm = model.computeMarginals(nodeScores, edgeScores, 
					nodeMarginals, edgeMarginals);
			negLikelihood -= model.computeLabelLikelihood(nodeScores,
					edgeScores, logNorm, labels[i]);
			computeSoftCounts(i, edgeMarginals, gradient);
		}
		objective = negLikelihood - 0.5 * lambda *
				ArrayHelper.l2Norm(parameters);
	}
	
	private void computeHardCounts(int instanceID, double[] counts) {
		int length = features.getInstanceLength(instanceID);
		for (int i = 0; i < length; i++) {
			int s = labels[instanceID][i];
			int sp = (i == 0) ? model.S0 : labels[instanceID][i - 1];
			features.addToCounts(instanceID, i, s, sp, counts, 1.0);
		}
	}
	
	private void computeSoftCounts(int instanceID, double[][][] edgeMarginals,
			double[] counts) {
		int length = features.getInstanceLength(instanceID);
		for (int s = 0; s < numTargetStates; s++) {
			features.addToCounts(instanceID, 0, s, model.S0, counts,
					edgeMarginals[0][s][model.S0]);
			features.addToCounts(instanceID, length, model.SN, s, counts,
					edgeMarginals[length][model.SN][s]);
		}
		for (int i = 1; i < length; i++) {
			for (int s = 0; s < numTargetStates; s++) {
				for (int sp = 0; sp < numTargetStates; sp++) {
					features.addToCounts(instanceID, i, s, sp, counts,
							edgeMarginals[i][s][sp]);
				}
			}
		}
	}
}
