package experiment;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;

import optimization.*;
import data.Evaluator;
import data.NERCorpus;
import data.NERSequence;
import feature.NERFeatureExtractor;
import feature.NGramFeatureExtractor;
import feature.SequentialFeatures;
import feature.SparseVector;
import gnu.trove.list.array.TIntArrayList;
import graph.KNNGraphConstructor;

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
		
		ArrayList<NERSequence> allInstances = new ArrayList<NERSequence>();
		allInstances.addAll(corpusTrain.instances);
		allInstances.addAll(corpusDev.instances);
		
		int numTrains = corpusTrain.instances.size();
		int[][] labels = new int[allInstances.size()][];
		TIntArrayList trainList = new TIntArrayList(),
					  devList = new TIntArrayList();
		for (int i = 0; i < allInstances.size(); i++) {
			labels[i] = i < numTrains ? corpusTrain.instances.get(i).getLabels() :
				corpusDev.instances.get(i-numTrains).getLabels();
			//if (i < numTrains) {
			if (i < 1000) {
				trainList.add(i);
			} else if (i >= numTrains) {
				devList.add(i);
			}
		}
		NERFeatureExtractor extractor = new NERFeatureExtractor(corpusTrain,
				allInstances, 5);
		extractor.printInfo();
		
		NGramFeatureExtractor ngramExtractor = new NGramFeatureExtractor(
				corpusTrain, allInstances);
		ngramExtractor.printInfo();
		int[][] ngramIDs = ngramExtractor.ngramIDs;
		SparseVector[] ngramFeatures = ngramExtractor.ngramFeatures;
		
		KNNGraphConstructor graphConstructor = new KNNGraphConstructor(
				ngramFeatures, 10, true, 1e-2, 4);
		graphConstructor.run();
		SparseVector[] edges = graphConstructor.getEdgeList();
		
		// TODO:  regularization features (ngramIDs[][], edges[]) 
		
		SequentialFeatures features = extractor.getSequentialFeatures();
		Evaluator eval = new Evaluator(corpusTrain);
		
		// here lambda = 1 / C
		RegularizedExponentiatedGradientDescent optimizer =
				new RegularizedExponentiatedGradientDescent(features,
						labels, trainList.toArray(), devList.toArray(),  eval,
						1, 0.5, 1000, 12345);
		
		optimizer.optimize();
	}
}
