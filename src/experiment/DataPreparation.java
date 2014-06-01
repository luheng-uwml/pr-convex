package experiment;

import java.io.IOException;
import java.util.ArrayList;

import data.CountDictionary;
import data.IOHelper;
import data.NERCorpus;
import data.NERSequence;
import feature.NGramFeatureExtractor;

public class DataPreparation {
	
	public static void main(String[] args) {
		ArrayList<NERSequence> instances = new ArrayList<NERSequence>();
		
		NERCorpus corpusTrain = new NERCorpus();
		corpusTrain.readFromCoNLL2003("./data/eng.train");
		instances.addAll(corpusTrain.instances);
		
		NERCorpus corpusDev = new NERCorpus(corpusTrain, false);
		corpusDev.readFromCoNLL2003("./data/eng.testa");
		instances.addAll(corpusDev.instances);
		
		NERCorpus corpusDev2 = new NERCorpus(corpusTrain, false);
		corpusDev2.readFromCoNLL2003("./data/eng.testb");
		instances.addAll(corpusDev2.instances);
		
		corpusTrain.printCorpusInfo();
		corpusDev.printCorpusInfo();
		corpusDev2.printCorpusInfo();
		
		// output all words, lowercased words and bigrams
		NGramFeatureExtractor ngramExtractor = new NGramFeatureExtractor(
				corpusTrain, instances, 2, false);
		System.out.println("\nAll unigrams:\t" + corpusTrain.tokenDict.size());
		System.out.println("All bigrams:\t" + ngramExtractor.ngramDict.size());
		
		// put all words in one dictionary
		CountDictionary allWords = new CountDictionary(corpusTrain.tokenDict);
		for (int i = 0; i < corpusTrain.tokenDict.size(); i++) {
			allWords.addString(corpusTrain.tokenDict.getString(i).toLowerCase());
		}
		/*
		for (int i = 0; i < ngramExtractor.ngramDict.size(); i++) {
			String[] words = ngramExtractor.ngramDict.getString(i).split(" ");
			String bigram = words[0] + "_" + words[1];
			allWords.addString(bigram);
			allWords.addString(bigram.toLowerCase());
		}
		*/
		System.out.println("Everything:\t" + allWords.size());
		
		try {
			IOHelper.saveCountDictionary(allWords, "./data/all_unigrams.txt");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
