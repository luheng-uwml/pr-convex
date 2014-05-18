package feature;

import java.util.ArrayList;

import data.CountDictionary;
import data.NERCorpus;
import data.NERSequence;

public class NERPotentialFunction {
	NERCorpus corpus;
	CountDictionary featureDict, stateDict;
	// total sentences, tokens, tags, previous tags,
	SparseVector[][][][] features;  
	int numStates;
	
	public NERPotentialFunction(NERCorpus corpus, int totalNumInstances) {
		this.corpus = corpus;
		featureDict = new CountDictionary();
		stateDict = new CountDictionary(corpus.nerDict);
		stateDict.addString("O-START"); // dummy start state
		stateDict.addString("O-END");   // dummy end state
		numStates = stateDict.size();
		features = new SparseVector[totalNumInstances][][][];
	}
	
	public void extractLocalFeatures(ArrayList<NERSequence> instances) {
		for (NERSequence instance : instances) {
			
		}
	}
	
	public void printInfo() {
		
	}
}
