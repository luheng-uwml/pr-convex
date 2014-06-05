package experiment;

import java.util.ArrayList;

import data.NERCorpus;
import data.NERSequence;
import feature.NGramFeatureExtractor;
import graph.KNNGraphConstructor;

public class GraphBuildingExperiment {
	
	public static void main(String[] args) {
		GraphBuildConfig config = new GraphBuildConfig(args);
		
		System.out.println("Train:");
		NERCorpus corpusTrain = new NERCorpus();
		corpusTrain.readFromCoNLL2003("./data/eng.train");
		corpusTrain.printCorpusInfo();
		
		System.out.println("Dev A:");
		NERCorpus corpusDevA = new NERCorpus(corpusTrain, false);
		corpusDevA.readFromCoNLL2003("./data/eng.testa");
		corpusDevA.printCorpusInfo();
		
		System.out.println("Dev B:");
		NERCorpus corpusDevB = new NERCorpus(corpusTrain, false);
		corpusDevB.readFromCoNLL2003("./data/eng.testb");
		corpusDevB.printCorpusInfo();
		
		ArrayList<NERSequence> allInstances = new ArrayList<NERSequence>();
		allInstances.addAll(corpusTrain.instances);
		if (config.useDevA) {
			allInstances.addAll(corpusDevA.instances);
		}
		if (config.useDevB) {
			allInstances.addAll(corpusDevB.instances);
		}
		
		// build a graph
		NGramFeatureExtractor ngramExtractor = new NGramFeatureExtractor(
				corpusTrain, allInstances);
		KNNGraphConstructor graphConstructor = new KNNGraphConstructor(
				ngramExtractor.getNGramFeatures(), config.numNeighbors, true,
				config.edgeWeightThreshold, config.numThreads);
		
		// save
	}
}
