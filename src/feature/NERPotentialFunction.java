package feature;

import java.util.ArrayList;

import data.CountDictionary;
import data.NERCorpus;
import data.NERSequence;

public class NERPotentialFunction {
	NERCorpus corpus;
	CountDictionary observationDict, featureDict, stateDict;
	// total sentences, tokens, tags, previous tags,
	SparseVector[][][][] features;  
	int numStates;
	
	public NERPotentialFunction(NERCorpus corpus) {
		this.corpus = corpus;
		observationDict = new CountDictionary();
		featureDict = new CountDictionary();
		stateDict = new CountDictionary(corpus.nerDict);
		stateDict.addString("O-START"); // dummy start state
		stateDict.addString("O-END");   // dummy end state
		numStates = stateDict.size();
		//features = new SparseVector[totalNumInstances][][][];
	}
	
	public void extractFeatures(ArrayList<NERSequence> instances) {
		for (NERSequence instance : instances) {
			for (int i = 0; i < instance.length; i++) {
				extractObservedFeatures(instance, i);
			}
		}
	}
	
	public int getNumObservations() {
		return observationDict.size();
	}
	
	private void addFeature(DynamicSparseVector fvec, String feature) {
		if (fvec == null) {
			featureDict.addString(feature, true);
		} else {
			int fid = featureDict.addString(feature, false);
			if (fid >= 0) {
				fvec.add(fid, 1);
			}
		}
	}
	
	private void addFeature(CountDictionary dict, String feature) {
		dict.addString(feature, true);
	}
	
	/* Extract lexicon features:
	 *   lowercased word, suffix 2-4,
	 *   is_capitalized
	 *   has_number
	 *   is_punctuation
	 */
	private void extractObservedFeatures(NERSequence instance, int position) {
		String token = instance.getToken(position);
		String posTag = instance.getPosTag(position);
		String chunkTag = instance.getChunkTag(position);
		String ltok = token.toLowerCase();
		int tlen = token.length();
		addFeature(observationDict, "LTOK=" + ltok);
		if (tlen > 2) {
			addFeature(observationDict, "SUF2=" + ltok.substring(tlen - 2));
		}
		if (tlen > 3) {
			addFeature(observationDict, "SUF3=" + ltok.substring(tlen - 3));
		}
		if (tlen > 4) {
			addFeature(observationDict, "SUF4=" + ltok.substring(tlen - 4));
		}
		if (!token.startsWith(ltok)) {
			addFeature(observationDict, "IS_CAP");
		}
		if (RegexHelper.isNumerical(token)) {
			addFeature(observationDict, "IS_NUM");
		}
		if (RegexHelper.isPunctuation(token)) {
			addFeature(observationDict, "IS_PUN");
		}
		addFeature(observationDict, "POS=" + posTag);
		addFeature(observationDict, "CHK=" + chunkTag);
		// ngram tags
		if (position > 0) {
			String posL = instance.getPosTag(position - 1);
			String chunkL = instance.getChunkTag(position - 1);
			addFeature(observationDict, "POSL=" + posL);
			addFeature(observationDict, "CHKL=" + chunkL);
			addFeature(observationDict, "POS+L=" + posL + "," + posTag);
			addFeature(observationDict, "CHK+L=" + chunkL + "," + chunkTag);
		} else {
			addFeature(observationDict, "S_START");
		}
		if (position < instance.length - 1) {
			String posR = instance.getPosTag(position + 1);
			String chunkR = instance.getChunkTag(position + 1);
			addFeature(observationDict, "POSR=" + posR);
			addFeature(observationDict, "CHKR=" + chunkR);
			addFeature(observationDict, "POS+R=" + posTag + ", " + posR);
			addFeature(observationDict, "CHK+R=" + chunkTag + ", " + chunkR);
		} else {
			addFeature(observationDict, "S_END");
		}
	}
	
	public void printInfo() {
		System.out.println("Number of states\t" + numStates);
		System.out.println("Number of features\t" + featureDict.size());
	}
}
