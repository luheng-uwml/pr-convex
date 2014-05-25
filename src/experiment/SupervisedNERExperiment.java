package experiment;

import java.util.ArrayList;

import optimization.BatchGradientDescent;
import optimization.ExponentiatedGradientDescent;
import optimization.OnlineExponentiatedGradientDescent;
import data.Evaluator;
import data.NERCorpus;
import data.NERSequence;
import feature.NERFeatureExtractor;
import feature.SequentialFeatures;
import gnu.trove.list.array.TIntArrayList;

public class SupervisedNERExperiment {
	
	public static void main(String[] args) {
		NERCorpus corpusTrain = new NERCorpus();
		corpusTrain.readFromCoNLL2003("./data/eng.train");
		//System.out.println("Read " + corpusTrain.size() + " sentences.");
		
		NERCorpus corpusDev = new NERCorpus(corpusTrain, false);
		corpusDev.readFromCoNLL2003("./data/eng.testa");
		//System.out.println("Read " + corpusDev.size() + " sentences.");
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
			if (i < numTrains) {
			//if (i < 1000) {
				trainList.add(i);
			} else if (i >= numTrains) {
				devList.add(i);
			}
		}
		NERFeatureExtractor extractor = new NERFeatureExtractor(corpusTrain,
				allInstances, 5);
		extractor.printInfo();
		SequentialFeatures features = extractor.getSequentialFeatures();
		Evaluator eval = new Evaluator(corpusTrain);
		/*
		BatchGradientDescent optimizer = new BatchGradientDescent(features,
				labels, trainList.toArray(), devList.toArray(), 
				eval, 1e-4, 1e-5, 1000);
		
		*/
		/*
		ExponentiatedGradientDescent optimizer =
				new ExponentiatedGradientDescent(features, labels,
						trainList.toArray(), devList.toArray(),  eval, 1e-2,
						0.1, 1000);
		
		*/
		OnlineExponentiatedGradientDescent optimizer =
				new OnlineExponentiatedGradientDescent(features, labels,
						trainList.toArray(), devList.toArray(),  eval, 1,
						0.1, 1000, 12345);
		
		optimizer.optimize();
		//optimizer.testModel();
	}
}
