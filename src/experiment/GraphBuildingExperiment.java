package experiment;

import java.io.IOException;
import java.util.ArrayList;

import optimization.FeatureRescaledEGTrainer;
import optimization.SupervisedEGTrainer;
import data.CountDictionary;
import data.Evaluator;
import data.IOHelper;
import data.NERCorpus;
import data.NERSequence;
import data.WordEmbeddingDictionary;
import feature.NERFeatureExtractor;
import feature.NGramFeatureExtractor;
import feature.SequentialFeatures;
import feature.SparseVector;
import gnu.trove.list.array.TIntArrayList;
import graph.GraphRegularizer;
import graph.KNNGraphConstructor;
import graph.WordVectorGraphConstructor;

public class GraphBuildingExperiment {
	
	private static void buildGraph(GraphBuildConfig config) {
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
		if (!config.buildSmallGraph) {
			allInstances.addAll(corpusTrain.instances);
		}
		if (config.useDevA) {
			allInstances.addAll(corpusDevA.instances);
		}
		if (config.useDevB) {
			allInstances.addAll(corpusDevB.instances);
		}
		
		System.out.println("Building graph using " + allInstances.size() +
				" sentences");
		
		// extract ngrams
		NGramFeatureExtractor ngramExtractor = new NGramFeatureExtractor(
				corpusTrain, allInstances);
		
		KNNGraphConstructor graphConstructor = null;
		if (config.useWordEmbedding) {
			WordEmbeddingDictionary wvecDict = new WordEmbeddingDictionary(
					config.wordVectorsPath);
			double[][] wordVectors = ngramExtractor.getWordEmbeddingFeatures(
					wvecDict);
			graphConstructor = new WordVectorGraphConstructor(wordVectors,
										config.numNeighbors,
										config.mutualKNN,
										config.edgeWeightThreshold,
										config.numThreads);
			
		} else {
			graphConstructor = new KNNGraphConstructor(
										ngramExtractor.getNGramFeatures(),
										config.numNeighbors,
										config.mutualKNN,
										config.edgeWeightThreshold,
										config.numThreads);
		}
		graphConstructor.run();
		// save
		try {
			IOHelper.saveCountDictionary(ngramExtractor.ngramDict,
					config.ngramFilePath);
			IOHelper.saveSparseVectors(graphConstructor.getEdgeList(),
					config.graphFilePath);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	private static void loadGraph(GraphBuildConfig config) {
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
		TIntArrayList trainList = new TIntArrayList();
		TIntArrayList devList = new TIntArrayList();
		
		ArrayList<NERSequence> allInstances = new ArrayList<NERSequence>();
		if (!config.buildSmallGraph) {
			allInstances.addAll(corpusTrain.instances);
		}
		if (config.useDevA) {
			allInstances.addAll(corpusDevA.instances);
		}
		if (config.useDevB) {
			allInstances.addAll(corpusDevB.instances);
		}
		
		int numInstances = allInstances.size();
		int[][] labels = new int[numInstances][];
		for (int i = 0; i < numInstances; i++) {
			if (i < corpusDevA.instances.size()) {
				trainList.add(i);
			} else {
				devList.add(i);
			}
			labels[i] = allInstances.get(i).getLabels();
		}
		
		// load graph info
		CountDictionary ngramDict = null;
		SparseVector[] edges = null;
		try {
			ngramDict = IOHelper.loadCountDictionary(
					config.ngramFilePath);
			edges = IOHelper.loadSparseVectors(
					config.graphFilePath);
		} catch (IOException e) {
			e.printStackTrace();
		}
		NGramFeatureExtractor ngramExtractor = new NGramFeatureExtractor(
				corpusTrain, allInstances, 3, true, ngramDict);
		NERFeatureExtractor extractor = new NERFeatureExtractor(corpusTrain,
				corpusTrain.instances, 1);
		extractor.printInfo();
		SequentialFeatures features = extractor.getSequentialFeatures(
				allInstances);
		Evaluator eval = new Evaluator(corpusTrain);
		GraphRegularizer graph = new GraphRegularizer(
				ngramExtractor.getNGramIDs(), edges, features.numTargetStates);
		
		graph.validate(labels, corpusTrain.nerDict, ngramDict);
		
		SupervisedEGTrainer optimizer =
				new SupervisedEGTrainer(features, graph,
					labels, trainList.toArray(), devList.toArray(), eval,
					1, 1, 0.5, 500, 12345);
		optimizer.optimize();
	}
	
	public static void main(String[] args) {
		GraphBuildConfig config = new GraphBuildConfig(args);
		buildGraph(config);
		if (config.testGraph) {
			loadGraph(config);
		}
	}
}
