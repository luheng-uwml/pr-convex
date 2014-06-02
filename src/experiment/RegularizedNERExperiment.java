package experiment;

import java.io.IOException;
import java.util.ArrayList;

import optimization.*;
import data.Evaluator;
import data.IOHelper;
import data.NERCorpus;
import data.NERSequence;
import feature.*;
import graph.*;
import gnu.trove.list.array.TIntArrayList;

public class RegularizedNERExperiment {
	
	private static void runNonRegularizedExperiment() {
		NERCorpus corpusTrain = new NERCorpus();
		corpusTrain.readFromCoNLL2003("./data/eng.train");
	
		NERCorpus corpusDev = new NERCorpus(corpusTrain, false);
		corpusDev.readFromCoNLL2003("./data/eng.testa");
		corpusTrain.printCorpusInfo();
		corpusDev.printCorpusInfo();
		
		int numAllTokens = 0;
		for (NERSequence instance : corpusTrain.instances) {
			numAllTokens += instance.length;
		}
		System.out.println("Number of all tokens:\t" + numAllTokens);
		
		//int numLabeled = corpusTrain.size(), numTrains = numLabeled;
		int numLabeled = 1000, numTrains = numLabeled;
		int numInstances = numTrains + corpusDev.size();
		ArrayList<NERSequence> trainInstances = new ArrayList<NERSequence>();
		ArrayList<NERSequence> allInstances = new ArrayList<NERSequence>();
		TIntArrayList trainList = new TIntArrayList(),
					  devList = new TIntArrayList();
		int[][] labels = new int[numInstances][];
		// add from original TRAIN corpus
		for (int i = 0; i < Math.min(numTrains, corpusTrain.size()); i++) {
			int instanceID = allInstances.size();
			allInstances.add(corpusTrain.instances.get(i));
			labels[instanceID] = corpusTrain.instances.get(i).getLabels();
			if (instanceID < numLabeled) {
				trainInstances.add(corpusTrain.instances.get(i));
				trainList.add(instanceID);
			} else {
				devList.add(instanceID);
			}
		}
		// add from original DEV corpus
		for (int i = 0; i < Math.min(numInstances - numTrains,
				corpusDev.size()); i++) {
			int instanceID = allInstances.size();
			allInstances.add(corpusDev.instances.get(i));
			labels[instanceID] = corpusDev.instances.get(i).getLabels();
			devList.add(instanceID);
		}
		System.out.println("num trains::\t" + numTrains +
						   "\tnum all::\t" + numInstances);
		
		NERFeatureExtractor extractor = new NERFeatureExtractor(corpusTrain,
				trainInstances, 1);
		extractor.printInfo();
		
		SequentialFeatures features = extractor.
				getSequentialFeatures(allInstances);
		Evaluator eval = new Evaluator(corpusTrain);
		GraphRegularizer graph =
				new DummyGraphRegularizer(features.numTargetStates);
		double goldPenalty = graph.computeTotalPenalty(labels);
		System.out.println("gold penalty::\t" + goldPenalty);
		
		// here lambda = 1 / C
		RegularizedExponentiatedGradientDescent optimizer =
				new RegularizedExponentiatedGradientDescent(features, graph,
						labels, trainList.toArray(), devList.toArray(), eval,
						lambda1, 0, 0.5, 500, 12345);
		
		optimizer.optimize();
		try {
			IOHelper.saveOptimizationHistory(optimizer.history, "./experiments/temp.mat");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	private static void runRegularizedExperiment() {
		NERCorpus corpusTrain = new NERCorpus();
		corpusTrain.readFromCoNLL2003("./data/eng.train");
	
		NERCorpus corpusDev = new NERCorpus(corpusTrain, false);
		corpusDev.readFromCoNLL2003("./data/eng.testa");
		corpusTrain.printCorpusInfo();
		corpusDev.printCorpusInfo();
		
		int numAllTokens = 0;
		for (NERSequence instance : corpusTrain.instances) {
			numAllTokens += instance.length;
		}
		System.out.println("Number of all tokens:\t" + numAllTokens);
		
		int numLabeled = 1000, numTrains = numLabeled;
		int numInstances = numTrains + corpusDev.size();
		ArrayList<NERSequence> trainInstances = new ArrayList<NERSequence>();
		ArrayList<NERSequence> allInstances = new ArrayList<NERSequence>();
		TIntArrayList trainList = new TIntArrayList(),
					  devList = new TIntArrayList();
		int[][] labels = new int[numInstances][];
		// add from original TRAIN corpus
		for (int i = 0; i < Math.min(numTrains, corpusTrain.size()); i++) {
			int instanceID = allInstances.size();
			allInstances.add(corpusTrain.instances.get(i));
			labels[instanceID] = corpusTrain.instances.get(i).getLabels();
			if (instanceID < numLabeled) {
				trainInstances.add(corpusTrain.instances.get(i));
				trainList.add(instanceID);
			} else {
				devList.add(instanceID);
			}
		}
		// add from original DEV corpus
		for (int i = 0; i < Math.min(numInstances - numTrains,
				corpusDev.size()); i++) {
			int instanceID = allInstances.size();
			allInstances.add(corpusDev.instances.get(i));
			labels[instanceID] = corpusDev.instances.get(i).getLabels();
			devList.add(instanceID);
		}
		System.out.println("num trains::\t" + numTrains +
						   "\tnum all::\t" + numInstances);
		
		NERFeatureExtractor extractor = new NERFeatureExtractor(corpusTrain,
				trainInstances, 1);
		extractor.printInfo();
		
		NGramFeatureExtractor ngramExtractor = new NGramFeatureExtractor(
				corpusTrain, allInstances);
		KNNGraphConstructor graphConstructor = new KNNGraphConstructor(
				ngramExtractor.getNGramFeatures(), 10, true, 0.3, 8);
		graphConstructor.run();
		
		SequentialFeatures features = extractor.
				getSequentialFeatures(allInstances);
		Evaluator eval = new Evaluator(corpusTrain);
		GraphRegularizer graph =
				new GraphRegularizer(ngramExtractor.getNGramIDs(),
					graphConstructor.getEdgeList(), features.numTargetStates);
				//new DummyGraphRegularizer(features.numTargetStates);
		
		double goldPenalty = graph.computeTotalPenalty(labels);
		System.out.println("gold penalty::\t" + goldPenalty);
		
		graph.validate(labels, corpusTrain.nerDict, ngramExtractor.ngramDict);
		
		// here lambda = 1 / C
		RegularizedExponentiatedGradientDescent optimizer =
				new RegularizedExponentiatedGradientDescent(features, graph,
						labels, trainList.toArray(), devList.toArray(), eval,
						lambda1, lambda2, 0.5, 500, 12345);
		
		optimizer.optimize();
		try {
			IOHelper.saveOptimizationHistory(optimizer.history, "./experiments/temp.mat");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	private static double lambda1 = 10, lambda2 = 0;
	
	public static void main(String[] args) {
		runNonRegularizedExperiment();
	}
}
