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
		int numInstances = 2000; //corpusTrain.size();
		ArrayList<NERSequence> allInstances = new ArrayList<NERSequence>();
		TIntArrayList trainList = new TIntArrayList(),
					  devList = new TIntArrayList();
		int[][] labels = new int[numInstances][];
		for (int i = 0; i < Math.min(numInstances, corpusTrain.size()); i++) {
			int instanceID = allInstances.size();
			allInstances.add(corpusTrain.instances.get(i));
			labels[instanceID] = corpusTrain.instances.get(i).getLabels();
			if (i < numTrains) {
				trainList.add(instanceID);
			} else if (i < numInstances) {
				devList.add(instanceID);
			}
		}
		
		NERFeatureExtractor extractor = new NERFeatureExtractor(corpusTrain,
				allInstances, 3);
		extractor.printInfo();
		
		/*
		NGramFeatureExtractor ngramExtractor = new NGramFeatureExtractor(
				corpusTrain, allInstances);
		ngramExtractor.printInfo();
		KNNGraphConstructor graphConstructor = new KNNGraphConstructor(
				ngramExtractor.ngramFeatures, 10, true, 0, 8);
		graphConstructor.run();
		*/
		
		SequentialFeatures features = extractor.getSequentialFeatures();
		Evaluator eval = new Evaluator(corpusTrain);
		GraphRegularizer graph =
				//new GraphRegularizer(ngramExtractor.ngramIDs,
				//	graphConstructor.getEdgeList(), features.numTargetStates);
				new DummyGraphRegularizer(features.numTargetStates);
		
		double goldPenalty = graph.computeTotalPenalty(labels);
		System.out.println("gold penalty::\t" + goldPenalty);
		
		// here lambda = 1 / C
		RegularizedExponentiatedGradientDescent optimizer =
				new RegularizedExponentiatedGradientDescent(features, graph,
						labels, trainList.toArray(), devList.toArray(), eval,
						1, 0, 0.5, 1000, 12345);
		
		optimizer.optimize();
	}
}
