package experiment;

import java.util.ArrayList;

import optimization.*;
import data.Evaluator;
import data.NERCorpus;
import data.NERSequence;
import feature.*;

import gnu.trove.list.array.TIntArrayList;

public class SemiSupervisedNERExperiment {
	
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
				trainInstances, 3);
		extractor.printInfo();
	
		SequentialFeatures features = extractor.
				getSequentialFeatures(allInstances);
		Evaluator eval = new Evaluator(corpusTrain);
		
		// here lambda = 1 / C
		SemiSupervisedExponentiatedGradientDescent2 optimizer =
				new SemiSupervisedExponentiatedGradientDescent2(features,
						labels, trainList.toArray(), devList.toArray(), eval,
						1, 0.5, 200, 12345);
		
		optimizer.optimize();
	}
}
