package optimization;

import data.Evaluator;
import inference.SequentialInference;
import feature.SequentialFeatures;
//import gnu.trove.list.array.TIntArrayList;

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
			int[] trainList, int[] devList,
			Evaluator eval, double lambda, double initialStepSize,
			int maxNumIterations) {
		this.features = features;
		this.labels = labels;
		this.trainList = trainList;
		this.devList = devList;
		this.eval = eval;
		this.lambda = lambda;
		this.initialStepSize = initialStepSize;
		this.maxNumIterations = maxNumIterations;
		initialize();
	}
	
	public void initialize() {
		numFeatures = features.numAllFeatures;
		numStates = features.numStates;
		numTargetStates = features.numTargetStates;
		model= new SequentialInference(1000, numStates);
		parameters = new double[numFeatures];
		gradient = new double[numFeatures];
		empiricalCounts = new double[numFeatures];
		ArrayHelper.deepFill(parameters, 0.0);
		ArrayHelper.deepFill(empiricalCounts, 0.0);
		// compute empirical counts from labels
		for (int i : trainList) {
			OptimizationHelper.computeHardCounts(features, i, labels,
					empiricalCounts);
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
					ArrayHelper.l2NormSquared(gradient) + "\tPREC::\t" + precision +
					"\tREC::\t" + recall + "\tF1::\t" + f1);
			
			if (Math.abs(objective - prevObjective) / prevObjective < 1e-5) {
				System.out.println("succeed!");
				break;
			}
			double currStepSize = this.backtrackingLineSearch(stepSize, 0.5,
					0.5, 20);
			for (int i = 0; i < numFeatures; i++) {
				parameters[i] -= currStepSize * gradient[i];
			}
			prevObjective = objective;
			stepSize = currStepSize * 1.5;
			
			if (iteration % 100 == 99) {
				OptimizationHelper.testModel(features, eval, devList, labels,
						null, parameters);
			}
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
			OptimizationHelper.computeSoftCounts(features, i, edgeMarginals,
					gradient);
			// evaluate
			model.posteriorDecoding(nodeMarginals, decoded);
			int[] result = eval.evaluate(labels[i], decoded);
			runningAccuracy[0] += result[0];
			runningAccuracy[1] += result[1];
			runningAccuracy[2] += result[2];
		}
		System.out.println("neg log likelihood:\t" + negLikelihood);
		objective = negLikelihood + 0.5 * lambda *
				ArrayHelper.l2NormSquared(parameters);
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
				ArrayHelper.l2NormSquared(tempParameters);
	}
	
	private double backtrackingLineSearch(double initStepSize, double alpha,
			double beta, int maxIterations) {
		double[] tempParameters = new double[numFeatures];
		double tStep = initStepSize;
		double gradientNorm = ArrayHelper.l2NormSquared(gradient);
		for (int i = 0; i < maxIterations; i++) {
			for (int j = 0; j < numFeatures; j++) {
				tempParameters[j] = parameters[j] - tStep * gradient[j];
			}
			System.out.println("paramters norm:\t" +
			ArrayHelper.l2NormSquared(tempParameters));
			double tempObjective = computeObjective(tempParameters);
			System.out.println("stepsize:\t" + tStep + "\ttemp objective:\t" + 
					tempObjective);
			if (tempObjective < objective - alpha * tStep * gradientNorm) {
				return tStep;
			}
			tStep *= beta;
		}
		System.out.println("unable to find step size");
		return initStepSize;
	}
}
