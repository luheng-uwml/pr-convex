package experiment;

import java.util.ArrayList;

import optimization.*;
import data.Evaluator;
import data.NERCorpus;
import data.NERSequence;
import feature.*;
import graph.*;

import gnu.trove.list.array.TIntArrayList;

public class RegularizedNERExperiment {
	
	public static void main(String[] args) {
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
		
		int numTrains = 1000;
		int numInstances = numTrains + corpusDev.size();
		ArrayList<NERSequence> allInstances = new ArrayList<NERSequence>();
		TIntArrayList trainList = new TIntArrayList(),
					  devList = new TIntArrayList();
		int[][] labels = new int[numInstances][];
		// add from original TRAIN corpus
		for (int i = 0; i < Math.min(numTrains, corpusTrain.size()); i++) {
			int instanceID = allInstances.size();
			allInstances.add(corpusTrain.instances.get(i));
			labels[instanceID] = corpusTrain.instances.get(i).getLabels();
			trainList.add(instanceID);
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
				allInstances, 5);
		extractor.printInfo();
		
		NGramFeatureExtractor ngramExtractor = new NGramFeatureExtractor(
				corpusTrain, allInstances);
		ngramExtractor.printInfo();
		KNNGraphConstructor graphConstructor = new KNNGraphConstructor(
				ngramExtractor.ngramFeatures, 20, true, 0.3, 8);
		graphConstructor.run();
		
		SequentialFeatures features = extractor.getSequentialFeatures();
		Evaluator eval = new Evaluator(corpusTrain);
		GraphRegularizer graph =
				//new GraphRegularizer(ngramExtractor.ngramIDs,
				//	graphConstructor.getEdgeList(), features.numTargetStates);
				new DummyGraphRegularizer(features.numTargetStates);
		
		double goldPenalty = graph.computeTotalPenalty(labels);
		System.out.println("gold penalty::\t" + goldPenalty);
		
		graph.validate(labels, ngramExtractor.ngramDict);
		
		// here lambda = 1 / C
		RegularizedExponentiatedGradientDescent optimizer =
				new RegularizedExponentiatedGradientDescent(features, graph,
						labels, trainList.toArray(), devList.toArray(), eval,
						1, 1, 0.5, 200, 12345);
		
		optimizer.optimize();
	}
}
