package experiment;

import java.util.ArrayList;

import optimization.BatchGradientDescent;
import data.NERCorpus;
import data.NERSequence;
import feature.NERFeatureExtractor;
import feature.SequentialFeatures;

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
		
		int numLabeled = corpusTrain.instances.size();
		int[][] labels = new int[allInstances.size()][];
		for (int i = 0; i < numLabeled; i++) {
			labels[i] = corpusTrain.instances.get(i).getLabels();
		}
		NERFeatureExtractor extractor = new NERFeatureExtractor(corpusTrain,
				allInstances);
		extractor.printInfo();
		SequentialFeatures features = extractor.getSequentialFeatures();
		BatchGradientDescent optimizer = new BatchGradientDescent(features,
				labels, 1.0, 1e-5, 100);
		optimizer.optimize();
	}
}
