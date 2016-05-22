package optimization;

import java.util.*;

import inference.LatticeHelper;
import inference.SequentialInference;
import data.Evaluator;
import feature.SequentialFeatures;

public class StructuredPerceptron {
    SequentialFeatures features;
    SequentialInference model;
    Evaluator eval;
    int[][] labels;
    int[] trainList, devList;
    double[] parameters, avgParameters, runningAccuracy;
    double initialStepSize, stepSize;
    int numFeatures, maxNumIterations, numStates, numTargetStates;
    Random randomGen;

    public StructuredPerceptron(SequentialFeatures features, int[][] labels,
                                int[] trainList, int[] devList,
                                Evaluator eval, double initialStepSize,
                                int maxNumIterations, int randomSeed) {
        this.features = features;
        this.labels = labels;
        this.trainList = trainList;
        this.devList = devList;
        this.eval = eval;
        this.initialStepSize = initialStepSize;
        this.maxNumIterations = maxNumIterations;
        this.randomGen = new Random(randomSeed);
        initialize();
    }

    public void initialize() {
        numFeatures = features.numAllFeatures;
        numStates = features.numStates;
        numTargetStates = features.numTargetStates;
        model= new SequentialInference(1000, numStates);
        parameters = new double[numFeatures];
        avgParameters = new double[numFeatures];
        ArrayHelper.deepFill(parameters, 0.0);
        ArrayHelper.deepFill(avgParameters, 0.0);
        runningAccuracy = new double[3]; // Precision, Recall, F1
    }

    public void optimize() {
        stepSize = initialStepSize;
        int numUpdates = 0;
        List<Integer> trainIds = new ArrayList<>();
        for (int tid : trainList) {
            trainIds.add(tid);
        }
        for (int iteration = 0; iteration < maxNumIterations; iteration ++) {
            ArrayHelper.deepFill(runningAccuracy, 0.0);
            Collections.shuffle(trainIds, randomGen);
            for (int instanceId : trainIds) {
                // Random update.
                int instanceID = trainList[randomGen.nextInt(trainList.length)];
                int length = features.getInstanceLength(instanceID);
                final double[][] edgeScores = new double[numStates][numStates];
                final double[][] nodeScores = new double[length][numTargetStates];
                final int[] gold = labels[instanceID];
                final int[] prediction = new int[length];
                features.computeEdgeScores(edgeScores, parameters);
                features.computeNodeScores(instanceID, nodeScores, parameters);
                model.viterbiDecoding(nodeScores, edgeScores, prediction);

                for (int i = 0; i < length; i++) {
                    if (i == 0) {
                        features.addEdgeToCounts(instanceID, gold[0], features.S0, parameters, stepSize);
                        features.addEdgeToCounts(instanceID, prediction[0], features.S0, parameters, -stepSize);
                    } else {
                        features.addEdgeToCounts(instanceID, gold[i], gold[i - 1], parameters, stepSize);
                        features.addEdgeToCounts(instanceID, prediction[i], prediction[i - 1], parameters, -stepSize);
                    }
                    if (i == length - 1) {
                        features.addEdgeToCounts(instanceID, features.SN, gold[i], parameters, stepSize);
                        features.addEdgeToCounts(instanceID, features.SN, prediction[i], parameters, -stepSize);
                    }
                    features.addNodeToCounts(instanceID, i, gold[i], parameters, stepSize);
                    features.addNodeToCounts(instanceID, i, prediction[i], parameters, -stepSize);
                }

                numUpdates++;
                for (int i = 0; i < numFeatures; i++) {
                    avgParameters[i] += parameters[i];
                }

                // compute accuracy
                int[] result = eval.evaluate(gold, prediction);
                runningAccuracy[0] += result[0];
                runningAccuracy[1] += result[1];
                runningAccuracy[2] += result[2];
                double precision = runningAccuracy[2] / runningAccuracy[1];
                double recall = runningAccuracy[2] / runningAccuracy[0];
                double f1 = (precision + recall > 0) ? (2 * precision * recall) / (precision + recall) : 0.0;


                if (iteration % 100 == 99) {
                    System.out.println("ITER::\t" + iteration + "\tPARAM::\t" + ArrayHelper.l2NormSquared(parameters)
                            + "\tPREC::\t" + precision + "\tREC::\t" + recall + "\tF1::\t" + f1);
                    double[] weights = new double[numFeatures];
                    for (int i = 0; i < numFeatures; i++) {
                        weights[i] = avgParameters[i] / numUpdates;
                    }
                    OptimizationHelper.testModel(features, eval, devList, labels, null, weights);
                }
            }
        }
    }

}
