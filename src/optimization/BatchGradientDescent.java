package optimization;

import inference.SequentialInference;
import feature.SequentialFeatures;
import gnu.trove.list.array.TIntArrayList;

public class BatchGradientDescent {
	SequentialFeatures features;
	SequentialInference model;
	int[][] labels;
	int[] trainList, devList;
	double[] theta;
	double lambda;
	double initialStepSize;
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
	
	private void initialize() {
		numFeatures = features.getNumFeatures();
		numStates = features.getNumStates();
		numTargetStates = numStates - 2;
		theta = new double[numFeatures];
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
	}
	
	/* Log-linear objective
	 * - \sum_x log (y* | x*) + \labmda / 2 || theta ||^2 
	 */
	public double updateObjectiveAndGradient() {
		double negLikelihood = 0;
		double[][] edgeScores = new double[numStates][numStates];
		features.computeEdgeScores(edgeScores, theta);
		for (int i : trainList) {
			int length = features.getInstanceLength(i);
			double[][] nodeScores = new double[length][numTargetStates];
			double[][] nodeMarginal = new double[length + 1][numStates];
			double[][][] edgeMarginal =
					new double[length + 1][numStates][numStates];
			double logNorm = Double.NEGATIVE_INFINITY;
			
			features.computeNodeScores(i, nodeScores, theta);
			logNorm = model.computeMarginals(nodeScores, edgeScores, 
					nodeMarginal, edgeMarginal);
			negLikelihood -= model.computeLabelLikelihood(nodeScores,
					edgeScores, logNorm, labels[i]);
		}
		return negLikelihood - 0.5 * lambda * getL2Norm(theta);
	}
	
	private double getL2Norm(double[] arr) {
		double norm = 0;
		for (int i = 0; i < arr.length; i++) {
			norm += arr[i] * arr[i];
		}
		return norm;
	}
}
