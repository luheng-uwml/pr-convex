package experiment;

import java.util.ArrayList;

import optimization.SupervisedExponentiatedGradientDescent3;
import data.Evaluator;
import data.NERCorpus;
import data.NERSequence;
import data.WordEmbeddingDictionary;
import feature.NERFeatureExtractor;
import feature.NGramFeatureExtractor;
import feature.SequentialFeatures;
import gnu.trove.list.array.TIntArrayList;
import graph.GraphRegularizer;
import graph.KNNGraphConstructor;
import graph.WordVectorGraphConstructor;

public class WordVectorGraphExperiment {
	
	public static void main(String[] args) {
		NERCorpus corpusTrain = new NERCorpus();
		corpusTrain.readFromCoNLL2003("./data/eng.train");
	
		NERCorpus corpusDev = new NERCorpus(corpusTrain, false);
		corpusDev.readFromCoNLL2003("./data/eng.testa");
		corpusTrain.printCorpusInfo();
		corpusDev.printCorpusInfo();
		
		WordEmbeddingDictionary wvecDict = new WordEmbeddingDictionary(
				"./data/all_unigrams.vec");
		
		int numAllTokens = 0;
		for (NERSequence instance : corpusTrain.instances) {
			numAllTokens += instance.length;
		}
		System.out.println("Number of all tokens:\t" + numAllTokens);
		
		int numLabeled = 1000, numTrains = 1000;
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
		
		// construct graph ..
		NGramFeatureExtractor ngramExtractor = new NGramFeatureExtractor(
				corpusTrain, trainInstances);
	
		// compute normalized embedding for each ngram
		double[][] wvecs = ngramExtractor.getWordEmbeddingFeatures(wvecDict);
		WordVectorGraphConstructor graphConstructor =
				new WordVectorGraphConstructor(wvecs, 5, true, 0.3, 8);
		graphConstructor.run();
		
		GraphRegularizer graph =
				new GraphRegularizer(ngramExtractor.getNGramIDs(),
					graphConstructor.getEdgeList(), features.numTargetStates);
				
		double goldPenalty = graph.computeTotalPenalty(labels);
		System.out.println("gold penalty::\t" + goldPenalty);
		
		graph.validate(labels, corpusTrain.nerDict, ngramExtractor.ngramDict);
		
		// here lambda = 1 / C
		Evaluator eval = new Evaluator(corpusTrain);
		SupervisedExponentiatedGradientDescent3 optimizer =
				new SupervisedExponentiatedGradientDescent3(features, graph,
						labels, trainList.toArray(), devList.toArray(), eval,
						1, 0.5, 0.5, 500, 12345);
		
		optimizer.optimize();
	}
}
