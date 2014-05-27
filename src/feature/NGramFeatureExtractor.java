package feature;

import java.util.ArrayList;

import data.CountDictionary;
import data.NERCorpus;
import data.NERSequence;

public class NGramFeatureExtractor {
	NERCorpus corpus;
	CountDictionary ngramDict, ngramContextFeatureDict;
	public int[][] ngramIDs; // sentence_id x position 
	public SparseVector[] ngramFeatures;  
	int ngramSize, contextSize, minFeatureFrequency, numNGrams;
	int[][] featureTemplate = { {-1, 0, 1}, {-2, -1, 0}, {0, 1, 2}, {-2, 0, 2},
			{-1, 1}, {-2, 2}, {-2, -1}, {1, 2}, {-2}, {-1}, {0}, {1}, {2}};
	
	public NGramFeatureExtractor(NERCorpus corpus,
			ArrayList<NERSequence> instances) {
		this(corpus, instances, 3, 5, 1);
	}
	
	public NGramFeatureExtractor(NERCorpus corpus,
			ArrayList<NERSequence> instances, int ngramSize,
			int contextSize, int minFeatureFrequency) {
		this.corpus = corpus;
		this.ngramSize = ngramSize;
		this.contextSize = contextSize;
		this.minFeatureFrequency = minFeatureFrequency;
		extractNGrams(instances);
		computeNGramFeatures(instances);
		normalizeNGramFeatures();
	}
	
	public void extractNGrams(ArrayList<NERSequence> instances) {
		ngramDict = new CountDictionary();
		ngramIDs = new int[instances.size()][];
		int leftLen = - (ngramSize / 2),
			rightLen = (ngramSize / 2) + (ngramSize % 2);
		for (int instanceID = 0; instanceID < instances.size(); instanceID++) {
			NERSequence instance = instances.get(instanceID);
			ngramIDs[instanceID] = new int[instance.length];
			for (int i = 0; i < instance.length; i++) {
				String ngram = "";
				for (int j = i - leftLen; j < i + rightLen; j++) {
					if (j < 0 || j >= instance.length) {
						ngram += "[-]";
					} else {
						ngram += instance.getToken(j).toLowerCase(); 
					}
					if (j + 1 < i + rightLen) {
						ngram += "\t";
					}
				}
				int ngramID = ngramDict.addString(ngram);
				ngramIDs[instanceID][i] = ngramID;
			}
		}
		numNGrams = ngramDict.size();
		System.out.println("Extracted " + numNGrams + " ngrams");
	}
	
	public void computeNGramFeatures(ArrayList<NERSequence> instances) {
		ngramContextFeatureDict = new CountDictionary();
		DynamicSparseVector[] tFeatures = new DynamicSparseVector[numNGrams];
		for (int i = 0; i < numNGrams; i++) {
			tFeatures[i] = new DynamicSparseVector();
		}
		for (int instanceID = 0; instanceID < instances.size(); instanceID++) {
			NERSequence instance = instances.get(instanceID);
			for (int i = 0; i < instance.length; i++) {
				int ngramID = ngramIDs[instanceID][i];
				for (int j = 0; j < featureTemplate.length; j++) {
					String tokens = "W" + j + "=", posTags = "P" + j + "=",
						   chunkTags = "C" + j + "="; 
					for (int k : featureTemplate[j]) {
						int p = i + k;
						if (p < 0 || p >= instance.length) {
							tokens += "[-]";
							posTags += "[-]";
							chunkTags += "[-]";
						} else {
							tokens += " " + instance.getToken(p);
							posTags += " " + instance.getPosTag(p);
							chunkTags += " " + instance.getChunkTag(p);
						}
					}
					addFeature(tokens, tFeatures[ngramID],
							ngramContextFeatureDict, true);
					addFeature(posTags, tFeatures[ngramID],
							ngramContextFeatureDict, true);
					addFeature(chunkTags, tFeatures[ngramID],
							ngramContextFeatureDict, true);
				}
			}
		}
		ngramFeatures = new SparseVector[numNGrams];
		for (int i = 0; i < numNGrams; i++) {
			ngramFeatures[i] = new SparseVector(tFeatures[i]);
		}
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
	
	// Normalize feature vector using PMI (point-wise mutual information)
	private void normalizeNGramFeatures() {
		int totalNGrams = ngramDict.getTotalCount();
		for (int i = 0; i < numNGrams; i++) {
			SparseVector fvec = ngramFeatures[i];
			for (int j = 0; j < fvec.length;j++) {
				int fid = fvec.indices[j]; 
				double featFreq = 1.0 * ngramContextFeatureDict.getCount(fid);
				double ngramProb = 1.0 * ngramDict.getCount(i) / totalNGrams;
				fvec.values[j] = Math.log(fvec.values[j]) -
						Math.log(ngramProb) - Math.log(featFreq);
			}
			fvec.normalize();
		}
	}
	
	public void printInfo() {
		System.out.println("Number of ngrams\t" + numNGrams);
		System.out.println("Number of ngrams context features\t" +
				ngramContextFeatureDict.size());
	}
}
