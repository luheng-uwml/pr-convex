package optimization;

import java.util.Random;

import inference.SequentialInference;
import data.Evaluator;
import feature.SequentialFeatures;

public class StructuredPerceptron {
    SequentialFeatures features;
    SequentialInference model;
    Evaluator eval;
    int[][] labels;
    int[] trainList, devList;
    double[] parameters, runningAccuracy;
    double[][][] edgeScores, edgeGradient; // pre-tag x current-tag
    double[][][] nodeScores, nodeGradient; // sentence-id x sentence-length x current-tag
    double[][][] marginalsOld;
    double[] logNorm, entropy, learningRate;
    double lambda, objective, initialStepSize;
    int numInstances, numFeatures, maxNumIterations, numStates, numTargetStates;
    Random randomGen;
    // line search conditions
    private static final int maxLineSearchIterations = 1;
    private static final double lsAlpha = 0.5, lsBeta = 0.5;

    public StructuredPerceptron(SequentialFeatures features,
                                int[][] labels, int[] trainList, int[] devList,
                                Evaluator eval, double lambda, double initialStepSize,
                                int maxNumIterations, int randomSeed) {
        this.features = features;
        this.labels = labels;
        this.trainList = trainList;
        this.devList = devList;
        this.eval = eval;
        this.lambda = 1.0 / lambda;
        this.initialStepSize = initialStepSize;
        this.maxNumIterations = maxNumIterations;
        this.randomGen = new Random(randomSeed);
        initialize();
    }

    private void initialize() {
        numInstances = features.numInstances;
        numFeatures = features.numAllFeatures;
        numStates = features.numStates;
        numTargetStates = features.numTargetStates;
        model= new SequentialInference(1000, numStates);
        parameters = new double[numFeatures];

        edgeScores = new double[numInstances][numStates][numStates];
        edgeGradient = new double[numInstances][numStates][numStates];
        nodeScores = new double[numInstances][][];
        nodeGradient = new double[numInstances][][];
        logNorm = new double[numInstances];
        entropy = new double[numInstances];
        learningRate = new double[numInstances];
        for (int i : trainList) {
            int length = features.getInstanceLength(i);
            nodeScores[i] = new double[length][numTargetStates];
            nodeGradient[i] = new double[length][numTargetStates];
            ArrayHelper.deepFill(nodeScores[i], 0.0);
            ArrayHelper.deepFill(nodeGradient[i], 0.0);
        }
        ArrayHelper.deepFill(parameters, 0.0);
        ArrayHelper.deepFill(edgeScores, 0.0);
        ArrayHelper.deepFill(edgeGradient, 0.0);
        ArrayHelper.deepFill(entropy, 0.0);
        ArrayHelper.deepFill(logNorm, 0.0);
        ArrayHelper.deepFill(learningRate, initialStepSize);
        objective = 0;
        runningAccuracy = new double[3]; // Precision, Recall, F1
        // initialize objective
        marginalsOld = null;

        for (int instanceID : trainList) {
            int length = features.getInstanceLength(instanceID);
            double[][][] edgeMarginals =
                    new double[length + 1][numStates][numStates];
            logNorm[instanceID] = model.computeMarginals(nodeScores[instanceID],
                    edgeScores[instanceID], null, edgeMarginals);
            entropy[instanceID] = model.computeEntropy(nodeScores[instanceID],
                    edgeScores[instanceID], edgeMarginals, logNorm[instanceID]);
            OptimizationHelper.computeSoftCounts(features, instanceID,
                    edgeMarginals, parameters, -1.0);
            objective -= entropy[instanceID];
        }
        objective += 0.5 * lambda * ArrayHelper.l2NormSquared(parameters);
        System.out.println("initial objective::\t" + objective);
    }

    public void computeAccuracy(int[] instList) {
        double[] theta = new double[numFeatures];
        for (int i = 0; i < numFeatures; i++) {
            theta[i] = parameters[i] * lambda;
        }
        OptimizationHelper.testModel(features, eval, instList, labels, null, theta);
    }

    private double computeAvgStepSize() {
        double result = 0;
        for (int i : trainList) {
            result += learningRate[i];
        }
        return result / trainList.length;
    }

    public void optimize() {
        double stepSize = initialStepSize;
        double prevObjective = Double.POSITIVE_INFINITY;
        for (int iteration = 0; iteration < maxNumIterations; iteration ++) {
            for (int k = 0; k < trainList.length; k++) {
                int instanceID = trainList[randomGen.nextInt(trainList.length)];

                computeGradient(instanceID);
                //double foundStep = armijoLineSearch(instanceID);
                learningRate[instanceID] = stepSize;
                //learningRate[instanceID] = Math.min(initialStepSize, foundStep * 1.5);
            }
            System.out.println("ITER::\t" + iteration +
                    "\tAVG-STEP:\t" + computeAvgStepSize() +
                    "\tOBJ::\t" + objective +
                    "\tPREV::\t" + prevObjective +
                    "\tPARA::\t" + ArrayHelper.l2NormSquared(parameters));
            if (iteration % 10 == 9) {
                validate();
                computeAccuracy(devList);
                //computePrimalObjective();
            }
            if (objective < prevObjective) {
                stepSize *= 1.05;
            } else {
                stepSize *= 0.5;
            }
            prevObjective = objective;
			/*
			if (iteration > 50) {
				stepSize = Math.max(initialStepSize / (iteration + 1),
						1e-5);
			}
			*/
            // TODO: stopping criterion
        }
    }

    private void computeGradient(int instanceID) {
        for (int i = 0; i < numStates; i++) {
            for (int j = 0; j < numStates; j++) {
                edgeGradient[instanceID][i][j] = edgeScores[instanceID][i][j] -
                        lambda * features.computeEdgeScore(i, j, parameters);
            }
        }
        int length = features.getInstanceLength(instanceID);
        for (int i = 0; i < length; i++) {
            for (int j = 0; j < numTargetStates; j++) {
                nodeGradient[instanceID][i][j] =
                        nodeScores[instanceID][i][j] - lambda *
                                features.computeNodeScore(instanceID, i, j, parameters);
            }
        }
    }

    private void validate() {
        double[] runningAccuracy = new double[3];
        ArrayHelper.deepFill(runningAccuracy, 0.0);
        // compute objective and likelihood
        int numStates = features.numStates;
        for (int i : trainList) {
            int length = features.getInstanceLength(i);
            double[][] nodeMarginals = new double[length][numStates];
            double[][][] edgeMarginals =
                    new double[length + 1][numStates][numStates];
            int[] decoded = new int[length];
            model.computeMarginals(nodeScores[i], edgeScores[i],
                    nodeMarginals, edgeMarginals);
            model.posteriorDecoding(nodeMarginals, decoded);
            int[] result = eval.evaluate(labels[i], decoded);
            runningAccuracy[0] += result[0];
            runningAccuracy[1] += result[1];
            runningAccuracy[2] += result[2];
        }
        double precision = runningAccuracy[2] / runningAccuracy[1];
        double recall = runningAccuracy[2] / runningAccuracy[0];
        double f1 = (precision + recall > 0) ?
                (2 * precision * recall) / (precision + recall) : 0.0;
        System.out.println("\tPREC::\t" + precision + "\tREC::\t" + recall +
                "\tF1::\t" + f1);
    }

    private void updateObjective(int instanceID) {
        int length = features.getInstanceLength(instanceID);
        double[][][] edgeMarginals =
                new double[length + 1][numStates][numStates];
        double entropyOld = entropy[instanceID];
        logNorm[instanceID] = model.computeMarginals(nodeScores[instanceID],
                edgeScores[instanceID], null, edgeMarginals);
        entropy[instanceID] = model.computeEntropy(nodeScores[instanceID],
                edgeScores[instanceID], edgeMarginals, logNorm[instanceID]);
        // update objective
        objective += entropyOld  - entropy[instanceID];
        objective -= 0.5 * lambda * ArrayHelper.l2NormSquared(parameters);
        OptimizationHelper.computeSoftCounts(features, instanceID,
                marginalsOld, parameters);
        OptimizationHelper.computeSoftCounts(features, instanceID,
                edgeMarginals, parameters, -1.0);
        objective += 0.5 * lambda * ArrayHelper.l2NormSquared(parameters);
    }

    private void backupMarginals(int instanceID) {
        int length = features.getInstanceLength(instanceID);
        marginalsOld = new double[length + 1][numStates][numStates];
        model.computeMarginals(nodeScores[instanceID], edgeScores[instanceID],
                null, marginalsOld);
    }

    private void updateDualParameters(int instanceID, double stepSize) {
        int length = features.getInstanceLength(instanceID);
        for (int i = 0; i < numStates; i++) {
            for (int j = 0; j < numStates; j++) {
                edgeScores[instanceID][i][j] -= stepSize *
                        edgeGradient[instanceID][i][j];
            }
        }
        for (int i = 0; i < length; i++) {
            for (int j = 0; j < numTargetStates; j++) {
                nodeScores[instanceID][i][j] -= stepSize *
                        nodeGradient[instanceID][i][j];
            }
        }
    }


}
