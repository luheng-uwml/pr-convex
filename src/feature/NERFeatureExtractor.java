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
	int MIN_FEATURE_FREQ = 3; 
			
	public NERFeatureExtractor(NERCorpus corpus,
			ArrayList<NERSequence> instances) {
		this.corpus = corpus;
		computeEdgeFeatures();
		computeNodeFeatures(instances);
	}
	
	public void computeEdgeFeatures() {
		edgeFeatureDict = new CountDictionary();
		stateDict = new CountDictionary(corpus.nerDict);
		startStateID = stateDict.addString("O-START"); // dummy start state
		endStateID = stateDict.addString("O-END");   // dummy end state
		numStates = stateDict.size();
		edgeFeatures = new SparseVector[numStates][numStates];
		for (int i = 0; i < numStates; i++) { // current state
			for (int j = 0; j < numStates; j++) { // prev state
				if (j == endStateID) {
					continue;
				}
				edgeFeatures[i][j] = addEdgeFeatures(
						stateDict.getString(i), stateDict.getString(j));
			}
		}
		numEdgeFeatures = edgeFeatureDict.size();
	}
	
	public void computeNodeFeatures(ArrayList<NERSequence> instances) {
		CountDictionary rawFeatureDict = new CountDictionary();
		for (NERSequence instance : instances) {
			for (int i = 0; i < instance.length; i++) {
				// precompute = true
				addNodeFeatures(instance, i, rawFeatureDict, true);
			}
		}
		nodeFeatureDict = new CountDictionary(rawFeatureDict, MIN_FEATURE_FREQ);
		nodeFeatures = new SparseVector[instances.size()][]; 
		for (int i = 0; i < instances.size(); i++) {
			NERSequence instance = instances.get(i);
			nodeFeatures[i] = new SparseVector[instance.length];
			for (int j = 0; j < instance.length; j++) {
				// precompute = false
				nodeFeatures[i][j] = addNodeFeatures(instance, j,
						nodeFeatureDict, false); 
			}
		}
		numNodeFeatures = nodeFeatureDict.size();
	}
	
	public SequentialFeatures getSequentialFeatures() {
		return new SequentialFeatures(nodeFeatures, edgeFeatures,
				numNodeFeatures, numEdgeFeatures);
	}
	
	private SparseVector addEdgeFeatures(String state, String prevState) {
		DynamicSparseVector fv0 = new DynamicSparseVector();
		fv0.add(edgeFeatureDict.addString("NER=" + state), 1);
		fv0.add(edgeFeatureDict.addString("NER_prev=" + prevState), 1);
		fv0.add(edgeFeatureDict.addString(
				"NER_bi=" +state + "_" + prevState), 1);
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
		addFeature("LTOK=" + ltok, fvec, featDict, precompute);
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
		// ngram tags
		if (position > 0) {
			String posL = instance.getPosTag(position - 1);
			String chunkL = instance.getChunkTag(position - 1);
			addFeature("POSL=" + posL, fvec, featDict, precompute);
			addFeature("CHKL=" + chunkL, fvec, featDict, precompute);
			addFeature("POS+L=" + posL + "," + posTag, fvec, featDict,
					precompute);
			addFeature("CHK+L=" + chunkL + "," + chunkTag, fvec, featDict,
					precompute);
		}
		if (position < instance.length - 1) {
			String posR = instance.getPosTag(position + 1);
			String chunkR = instance.getChunkTag(position + 1);
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
		System.out.println("Number of features\t" +
				(nodeFeatureDict.size() * numStates + edgeFeatureDict.size() + 1));
	}
}
