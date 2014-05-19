package optimization;

import data.Evaluator;
import inference.SequentialInference;
import feature.SequentialFeatures;
import gnu.trove.list.array.TIntArrayList;

public class BatchGradientDescent {
	SequentialFeatures features;
	SequentialInference model;
	Evaluator eval;
	int[][] labels;
	int[] trainList, devList;
	double[] parameters, empiricalCounts, gradient, runningAccuracy;
	double lambda, objective, initialStepSize, stepSize;
	int numFeatures, maxNumIterations, numStates, numTargetStates;
	
	public BatchGradientDescent(SequentialFeatures features, int[][] labels,
			Evaluator eval, double lambda, double initialStepSize,
			int maxNumIterations) {
		this.features = features;
		this.labels = labels;
		this.eval = eval;
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
		runningAccuracy = new double[3]; // Precision, Recall, F1
	}
	
	public void optimize() {
		stepSize = initialStepSize;
		double prevObjective = Double.POSITIVE_INFINITY;
		for (int iteration = 0; iteration < maxNumIterations; iteration ++) {
			updateObjectiveAndGradient();
			// compute accuracy
			double precision = runningAccuracy[2] / runningAccuracy[1];
			double recall = runningAccuracy[2] / runningAccuracy[0];
			double f1 = (precision + recall > 0) ?
					(2 * precision * recall) / (precision + recall) : 0.0;
					
			System.out.println("ITER::\t" + iteration + "\tOBJ::\t" +
					objective + "\tPREV::\t" + prevObjective + "\tGRAD::\t" +
					ArrayHelper.l2Norm(gradient) + "\tPREC::\t" + precision +
					"\tREC::\t" + recall + "\tF1::\t" + f1);
			
			if (Math.abs(objective - prevObjective) / prevObjective < 1e-5) {
				System.out.println("succeed!");
				break;
			}
			double currStepSize = this.backtrackingLineSearch(stepSize, 0.25, 0.5,
					20);
			for (int i = 0; i < numFeatures; i++) {
				parameters[i] -= currStepSize * gradient[i];
			}
			prevObjective = objective;
			stepSize = currStepSize * 1.5;
		}
	}
	
	/* Log-linear objective
	 * - \sum_x log (y* | x*) + \labmda / 2 || theta ||^2 
	 */
	private void updateObjectiveAndGradient() {
		ArrayHelper.deepFill(runningAccuracy, 0.0);
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
			double[][] nodeMarginals = new double[length][numStates];
			double[][][] edgeMarginals =
					new double[length + 1][numStates][numStates];
			int[] decoded = new int[length];
			double logNorm = Double.NEGATIVE_INFINITY;
			features.computeNodeScores(i, nodeScores, parameters);
			logNorm = model.computeMarginals(nodeScores, edgeScores, 
					nodeMarginals, edgeMarginals);
			negLikelihood -= model.computeLabelLikelihood(nodeScores,
					edgeScores, logNorm, labels[i]);
			computeSoftCounts(i, edgeMarginals, gradient);
			// evaluate
			model.posteriorDecoding(nodeMarginals, decoded);
			int[] result = eval.evaluate(labels[i], decoded);
			runningAccuracy[0] += result[0];
			runningAccuracy[1] += result[1];
			runningAccuracy[2] += result[2];
		}
		System.out.println("neg log likelihood:\t" + negLikelihood);
		objective = negLikelihood + 0.5 * lambda *
				ArrayHelper.l2Norm(parameters);
	}
	
	private double computeObjective(double[] tempParameters) {
		double negLikelihood = 0;
		double[][] edgeScores = new double[numStates][numStates];
		features.computeEdgeScores(edgeScores, tempParameters);
		for (int i : trainList) {
			int length = features.getInstanceLength(i);
			double[][] nodeScores = new double[length][numTargetStates];
			double[][] nodeMarginals = new double[length][numStates];
			double[][][] edgeMarginals =
					new double[length + 1][numStates][numStates];
			features.computeNodeScores(i, nodeScores, tempParameters);
			double logNorm = model.computeMarginals(nodeScores, edgeScores, 
					nodeMarginals, edgeMarginals);
			negLikelihood -= model.computeLabelLikelihood(nodeScores,
					edgeScores, logNorm, labels[i]);
		}
		return negLikelihood + 0.5 * lambda *
				ArrayHelper.l2Norm(tempParameters);
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
	
	// try alpha = 0.5, beta = 0.9 ?
	private double backtrackingLineSearch(double initStepSize, double alpha,
			double beta, int maxIterations) {
		double[] tempParameters = new double[numFeatures];
		double tStep = initStepSize;
		double gradientNorm = ArrayHelper.l2Norm(gradient);
		for (int i = 0; i < maxIterations; i++) {
			for (int j = 0; j < numFeatures; j++) {
				tempParameters[j] = parameters[j] - tStep * gradient[j];
			}
			System.out.println("paramters norm:\t" + ArrayHelper.l2Norm(tempParameters));
			double tempObjective = computeObjective(tempParameters);
			System.out.println("temp objective:\t" + tempObjective);
			if (tempObjective < objective - alpha * tStep * gradientNorm) {
				return tStep;
			}
			tStep *= beta;
		}
		System.out.println("unable to find step size");
		return initStepSize;
	}
}
