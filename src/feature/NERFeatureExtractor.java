package feature;

import java.util.ArrayList;

import data.CountDictionary;
import data.NERCorpus;
import data.NERSequence;

public class NERFeatureExtractor {
	NERCorpus corpus;
	CountDictionary nodeFeatureDict, edgeFeatureDict, stateDict;
	SparseVector[][] nodeFeatures, edgeFeatures;  
	int numStates, startStateID, endStateID, numNodeFeatures, numEdgeFeatures;
	int minFeatureFrequency;
	
	public NERFeatureExtractor(NERCorpus corpus,
			ArrayList<NERSequence> instances, int minFeatureFrequency) {
		this.corpus = corpus;
		this.minFeatureFrequency = minFeatureFrequency;
		extractEdgeFeatures();
		extractNodeFeatures(instances);
	}
	
	public void extractEdgeFeatures() {
		edgeFeatureDict = new CountDictionary();
		stateDict = new CountDictionary(corpus.nerDict);
		startStateID = stateDict.addString("O-START"); // dummy start state
		endStateID = stateDict.addString("O-END");   // dummy end state
		numStates = stateDict.size();
		edgeFeatures = new SparseVector[numStates][numStates];
		for (int i = 0; i < numStates; i++) { // current state
			for (int j = 0; j < numStates; j++) { // prev state
				if (i == startStateID || j == endStateID) {
					continue;
				}
				edgeFeatures[i][j] = addEdgeFeatures(
						stateDict.getString(i), stateDict.getString(j));
			}
		}
		numEdgeFeatures = edgeFeatureDict.size();
	}
	
	public void extractNodeFeatures(ArrayList<NERSequence> instances) {
		CountDictionary rawFeatureDict = new CountDictionary();
		for (NERSequence instance : instances) {
			for (int i = 0; i < instance.length; i++) {
				// precompute = true
				addNodeFeatures(instance, i, rawFeatureDict, true);
			}
		}
		nodeFeatureDict = new CountDictionary(rawFeatureDict,
				minFeatureFrequency);
		numNodeFeatures = nodeFeatureDict.size();
	}
	
	public SequentialFeatures getSequentialFeatures(
			ArrayList<NERSequence> instances) {
		nodeFeatures = new SparseVector[instances.size()][];
		for (int i = 0; i < instances.size(); i++) {
			NERSequence instance = instances.get(i);
			nodeFeatures[i] = new SparseVector[instance.length];
			for (int j = 0; j < instance.length; j++) {
				// precompute = false
				nodeFeatures[i][j] =
					addNodeFeatures(instance, j, nodeFeatureDict, false); 
			}
		}
		return new SequentialFeatures(nodeFeatures, edgeFeatures,
				numNodeFeatures, numEdgeFeatures, nodeFeatureDict,
				edgeFeatureDict, stateDict);
	}
	
	private SparseVector addEdgeFeatures(String state, String prevState) {
		DynamicSparseVector fv0 = new DynamicSparseVector();
		String stateRaw = state.startsWith("O") ? "O" : state.split("-")[0];
		String prevStateRaw = prevState.startsWith("O") ? "O" :
							  prevState.split("-")[0];
		
		fv0.add(edgeFeatureDict.addString("BIAS"), 1);
		fv0.add(edgeFeatureDict.addString("NER=" + state), 1);
		fv0.add(edgeFeatureDict.addString("NER_prev=" + prevState), 1);
		fv0.add(edgeFeatureDict.addString(
				"NER_bi=" +state + "_" + prevState), 1);
		fv0.add(edgeFeatureDict.addString("BIO=" + stateRaw), 1);
		fv0.add(edgeFeatureDict.addString("BIO_prev=" + prevStateRaw), 1);
		fv0.add(edgeFeatureDict.addString(
				"BIO_bi=" +stateRaw + "_" + prevStateRaw), 1);
		return new SparseVector(fv0);
	}
	
	private void addFeature(String feature, DynamicSparseVector fvec,
			CountDictionary fdict, boolean acceptNew) {
		if (acceptNew) {
			fdict.addString(feature, true);
		} else {
			int fid = fdict.addString(feature, false);
			if (fid >= 0) {
				fvec.add(fid, 1);
			}
		}
	}
	
	/* Extract lexicon features:
	 *   lowercased word, suffix 2-4,
	 *   is_capitalized
	 *   has_number
	 *   is_punctuation
	 */
	private SparseVector addNodeFeatures(NERSequence instance, int position,
			CountDictionary featDict, boolean precompute) {
		DynamicSparseVector fvec = new DynamicSparseVector();
		String token = instance.getToken(position);
		String posTag = instance.getPosTag(position);
		String chunkTag = instance.getChunkTag(position);
		String ltok = token.toLowerCase();
		int tlen = token.length();
		addFeature("TOK=" + ltok, fvec, featDict, precompute);
		addFeature("TOK_POS=" + ltok + "_" + posTag, fvec, featDict,
				precompute);
		addFeature("TOK_CHK=" + ltok + "_" + chunkTag, fvec, featDict,
				precompute);
		if (tlen > 2) {
			addFeature("SUF2=" + ltok.substring(tlen - 2), fvec,
					featDict, precompute);
		}
		if (tlen > 3) {
			addFeature("SUF3=" + ltok.substring(tlen - 3), fvec,
					featDict, precompute);
		}
		if (tlen > 4) {
			addFeature("SUF4=" + ltok.substring(tlen - 4), fvec,
					featDict, precompute);
		}
		if (!token.startsWith(ltok)) {
			addFeature("IS_CAP", fvec, featDict, precompute);
		}
		if (RegexHelper.isNumerical(token)) {
			addFeature("IS_NUM", fvec, featDict, precompute);
		}
		if (RegexHelper.isPunctuation(token)) {
			addFeature("IS_PUN", fvec, featDict, precompute);
		}
		addFeature("POS=" + posTag, fvec, featDict, precompute);
		addFeature("CHK=" + chunkTag, fvec, featDict, precompute);
		addFeature("POS_CHK=" + posTag + "_" + chunkTag, fvec, featDict,
				precompute);
		// ngram tags
		if (position > 0) {
			String tokL = instance.getToken(position - 1);
			String posL = instance.getPosTag(position - 1);
			String chunkL = instance.getChunkTag(position - 1);
			addFeature("TOKL=" + tokL, fvec, featDict, precompute);
			if (!tokL.startsWith(tokL.toLowerCase())) {
				addFeature("IS_CAP_L", fvec, featDict, precompute);
			}
			if (RegexHelper.isNumerical(tokL)) {
				addFeature("IS_NUM_L", fvec, featDict, precompute);
			}
			if (RegexHelper.isPunctuation(tokL)) {
				addFeature("IS_PUN_L", fvec, featDict, precompute);
			}
			addFeature("POSL=" + posL, fvec, featDict, precompute);
			addFeature("CHKL=" + chunkL, fvec, featDict, precompute);
			addFeature("POS+L=" + posL + "," + posTag, fvec, featDict,
					precompute);
			addFeature("CHK+L=" + chunkL + "," + chunkTag, fvec, featDict,
					precompute);
		}
		if (position < instance.length - 1) {
			String tokR = instance.getToken(position + 1);
			String posR = instance.getPosTag(position + 1);
			String chunkR = instance.getChunkTag(position + 1);
			addFeature("TOKR=" + tokR, fvec, featDict, precompute);
			if (!tokR.startsWith(tokR.toLowerCase())) {
				addFeature("IS_CAP_R", fvec, featDict, precompute);
			}
			if (RegexHelper.isNumerical(tokR)) {
				addFeature("IS_NUM_R", fvec, featDict, precompute);
			}
			if (RegexHelper.isPunctuation(tokR)) {
				addFeature("IS_PUN_R", fvec, featDict, precompute);
			}
			addFeature("POSR=" + posR, fvec, featDict, precompute);
			addFeature("CHKR=" + chunkR, fvec, featDict, precompute);
			addFeature("POS+R=" + posTag + ", " + posR, fvec, featDict,
					precompute);
			addFeature("CHK+R=" + chunkTag + ", " + chunkR, fvec, featDict,
					precompute);
		}
		return new SparseVector(fvec);
	}
	
	public void printInfo() {
		System.out.println("Number of states\t" + numStates);
		System.out.println("Dummy start state:\t" + startStateID);
		System.out.println("Dummy end state:\t" + endStateID);
		System.out.println("Number of features\t" +
			(nodeFeatureDict.size() * numStates + edgeFeatureDict.size()));
	}
}
