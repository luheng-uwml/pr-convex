package data;

import gnu.trove.map.hash.TObjectIntHashMap;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;

public class WordEmbeddingDictionary {
	TObjectIntHashMap<String> word2idx;
	ArrayList<double[]> vectors;
	int vectorSize;
	double[] emptyVector;
	
	public WordEmbeddingDictionary(String filePath) {
		word2idx = new TObjectIntHashMap<String>();
		vectors = new ArrayList<double[]>();
		readFromFile(filePath);
		vectorSize = vectors.get(0).length;
		emptyVector = new double[vectorSize];
		Arrays.fill(emptyVector, 0);
	}
	
	@SuppressWarnings("resource")
	private void readFromFile(String filePath) {
		String currLine = null;
		try {
			BufferedReader fin = new BufferedReader(new FileReader(filePath));
			while ((currLine = fin.readLine()) != null) {
				String[] info = currLine.split("\t");
				assert (info.length == 301);
				word2idx.put(info[0], vectors.size());
				double[] vec = new double[info.length - 1];
				for (int i = 1; i < info.length; i++) {
					vec[i-1] = Double.parseDouble(info[i]); 
				}
				vectors.add(vec);
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public int getVectorSize() {
		return vectorSize;
	}
	
	public double[] getVector(String word) {
		if (word2idx.contains(word)) {
			return vectors.get(word2idx.get(word));
		} else if (word2idx.contains(word.toLowerCase())) {
			return vectors.get(word2idx.get(word.toLowerCase()));
		}
		return emptyVector;
	}
}
