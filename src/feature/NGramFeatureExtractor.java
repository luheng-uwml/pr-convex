package feature;

import java.util.ArrayList;

import optimization.ArrayHelper;
import data.CountDictionary;
import data.NERCorpus;
import data.NERSequence;
import data.WordEmbeddingDictionary;

public class NGramFeatureExtractor {
	NERCorpus corpus;
	ArrayList<NERSequence> instances;
	public CountDictionary ngramDict;
	CountDictionary ngramFeatureDict;
	private int[][] ngramIDs; // sentence_id x position 
	private SparseVector[] ngramFeatures;
	private double[][] denseFeatures;
	int ngramSize, numNGrams;
	boolean toLowerCase;
	int[][] featureTemplate = { {-1, 0, 1}, {-2, -1, 0}, {0, 1, 2}, {-2, 0, 2},
			{-1, 1}, {-2, 2}, {-2, -1}, {1, 2}, {-2}, {-1}, {0}, {1}, {2}};
	//double[][] denseFeatureTemplate = { {0, 0.1}, {1, 0.8}, {2, 0.1} };
	
	public NGramFeatureExtractor(NERCorpus corpus,
			ArrayList<NERSequence> instances) {
		this(corpus, instances, 3, true);
	}
	
	public NGramFeatureExtractor(NERCorpus corpus,
			ArrayList<NERSequence> instances, int ngramSize,
			boolean toLowerCase) {
		this.corpus = corpus;
		this.ngramSize = ngramSize;
		this.toLowerCase = toLowerCase;
		this.instances = instances;
		this.ngramIDs = null;
		this.ngramFeatures = null;
		this.denseFeatures = null;
		this.ngramDict = null;
	}
	
	public NGramFeatureExtractor(NERCorpus corpus,
			ArrayList<NERSequence> instances, int ngramSize,
			boolean toLowerCase, CountDictionary ngramDict) {
		this(corpus, instances, ngramSize, toLowerCase);
		this.ngramDict = ngramDict;
	}
	
	public int[][] getNGramIDs() {
		if (ngramIDs == null) {
			extractNGrams(instances);
		}
		return ngramIDs;
	}
	
	public SparseVector[] getNGramFeatures() {
		if (ngramIDs == null) {
			extractNGrams(instances);
		}
		if (ngramFeatures == null) {
			computeNGramFeatures(instances);
			normalizeNGramFeatures();
		}
		return ngramFeatures;
	}
	
	public double[][] getWordEmbeddingFeatures(
			WordEmbeddingDictionary word2vec) {
		if (ngramIDs == null) {
			extractNGrams(instances);
		}
		int vectorSize = word2vec.getVectorSize();
		denseFeatures = new double[numNGrams][vectorSize * ngramSize];
		double[] weights = { 0.5, 1, 0.5 };
		// simple additive ...
		for (int i = 0; i < numNGrams; i++) {
			String[] words = ngramDict.getString(i).split(" ");
			for (int j = 0; j < ngramSize; j++) {
				double[] wvec = word2vec.getVector(words[j]);
				for (int k = 0; k < vectorSize; k++) {
					denseFeatures[i][j*vectorSize+k] += weights[j] * wvec[k];
				}
			}
			ArrayHelper.normalize(denseFeatures[i]);
		}
		return denseFeatures;
	}
	
	private void extractNGrams(ArrayList<NERSequence> instances) {
		ngramIDs = new int[instances.size()][];
		boolean useOldNGrams = (ngramDict != null); 
		if (!useOldNGrams) {
			ngramDict = new CountDictionary();
		}
		int leftLen = (ngramSize / 2),
			rightLen = (ngramSize / 2) + (ngramSize % 2);
		for (int instanceID = 0; instanceID < instances.size(); instanceID++) {
			NERSequence instance = instances.get(instanceID);
			ngramIDs[instanceID] = new int[instance.length];
			for (int i = 0; i < instance.length; i++) {
				String ngram = "";
				for (int j = i - leftLen; j < i + rightLen; j++) {
					if (j < 0) {
						ngram += "<s>";
					} else if (j >= instance.length) {
						ngram += "</s>";
					} else if (toLowerCase) {
						ngram += instance.getToken(j).toLowerCase(); 
					} else {
						ngram += instance.getToken(j);
					}
					if (j + 1 < i + rightLen) {
						ngram += " ";
					}
				}
				int ngramID = useOldNGrams ? ngramDict.lookupString(ngram) :
											 ngramDict.addString(ngram);
				assert ngramID >= 0;
				ngramIDs[instanceID][i] = ngramID;
			}
		}
		numNGrams = ngramDict.size();
		System.out.println("Extracted " + numNGrams + " ngrams");
	}
	
	public void computeNGramFeatures(ArrayList<NERSequence> instances) {
		ngramFeatureDict = new CountDictionary();
		DynamicSparseVector[] tFeatures = new DynamicSparseVector[numNGrams];
		for (int i = 0; i < numNGrams; i++) {
			tFeatures[i] = new DynamicSparseVector();
		}
		for (int instanceID = 0; instanceID < instances.size(); instanceID++) {
			NERSequence instance = instances.get(instanceID);
			for (int i = 0; i < instance.length; i++) {
				int ngramID = ngramIDs[instanceID][i];
				for (int j = 0; j < featureTemplate.length; j++) {
					String tokens = "W" + j + "=",
						   posTags = "P" + j + "=",
						   chunkTags = "C" + j + "="; 
					for (int k : featureTemplate[j]) {
						int p = i + k;
						if (p < 0) { 
							tokens += " <s>";
							posTags += " <s>";
							chunkTags += " <s>";
						} else if(p >= instance.length) {
							tokens += " </s>";
							posTags += " </s>";
							chunkTags += " </s>";
						} else {
							tokens += " " + instance.getToken(p);
							posTags += " " + instance.getPosTag(p);
							chunkTags += " " + instance.getChunkTag(p);
						}
					}
					addFeature(tokens, tFeatures[ngramID], ngramFeatureDict);
					addFeature(posTags, tFeatures[ngramID], ngramFeatureDict);
					addFeature(chunkTags, tFeatures[ngramID], ngramFeatureDict);
				}
			}
		}
		ngramFeatures = new SparseVector[numNGrams];
		for (int i = 0; i < numNGrams; i++) {
			ngramFeatures[i] = new SparseVector(tFeatures[i]);
		}
	}
	
	private void addFeature(String feature, DynamicSparseVector fvec,
			CountDictionary fdict) {
		int fid = fdict.addString(feature, true);
		fvec.add(fid, 1);
	}
	
	// Normalize feature vector using PMI (point-wise mutual information)
	private void normalizeNGramFeatures() {
		int totalNGrams = ngramDict.getTotalCount();
		for (int i = 0; i < numNGrams; i++) {
			SparseVector fvec = ngramFeatures[i];
			for (int j = 0; j < fvec.length;j++) {
				int fid = fvec.indices[j]; 
				double featFreq = 1.0 * ngramFeatureDict.getCount(fid);
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
				ngramFeatureDict.size());
	}
}
