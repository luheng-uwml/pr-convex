package data;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

import com.jmatio.io.MatFileWriter;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLDouble;

import optimization.OptimizationHistory;
import feature.SparseVector;

public class IOHelper {
	public static String printJoin(int[] strIDs, CountDictionary strDict) {
		String joined = "";
		for (int sid : strIDs) {
			if (joined.length() > 0) {
				joined += " ";
			}
			joined += strDict.getString(sid);
		}
		return joined;
	}
	
	// file format:
	// 		each line corresponds to one sparse vector
	//		starts with number of elements, followed with each key-value pair
	//		example:
	//		5	1:0.3	2:0.4	8:0.5	9:0.7	10:1.0
	public static void saveSparseVectors(SparseVector[] vecs, String filePath)
			throws IOException {
		BufferedWriter fout = new BufferedWriter(new FileWriter(filePath));
		fout.write(String.format("%d\n", vecs.length));
		for (int i = 0; i < vecs.length; i++) {
			SparseVector fvec = vecs[i];
			if (fvec == null || fvec.length == 0) {
				fout.write("0\n");
				continue;
			}
			fout.write(String.format("%d", fvec.length));
			for (int j = 0; j < fvec.length; j++) {
				fout.write(String.format("\t%d:%.12f", fvec.indices[j],
						fvec.values[j]));
			}
			fout.write("\n");
		}
		fout.close();
		System.out.println("Saving sparse vector list to:\t" + filePath);
	}
	
	public static SparseVector[] loadSparseVectors(String filePath)
			throws IOException {
		BufferedReader fread = new BufferedReader(new FileReader(filePath));
		int numVecs = Integer.parseInt(fread.readLine());
		SparseVector[] vecs = new SparseVector[numVecs];
		for (int i = 0; i < numVecs; i++) {
			String[] info = fread.readLine().split("\t");
			int len = Integer.parseInt(info[0]);
			if (len == 0) {
				vecs[i] = new SparseVector();
			} else {
				int[] ids = new int[len]; 
				double[] vals = new double[len];
				for (int j = 0; j < len; j++) {
					String[] element = info[j+1].split(":");
					ids[j] = Integer.parseInt(element[0]);
					vals[j] = Double.parseDouble(element[1]);
				}
				vecs[i] = new SparseVector(ids, vals);
			}
		}
		fread.close();
		System.out.println("Loaded sparse vector list from:\t" + filePath);
		return vecs;
	}
	
	// file format:
	// first line contains single integer: size
	// each line contains: [id]\t[freq]\t[string]
	public static void saveCountDictionary(CountDictionary dict,
			String filePath) throws IOException {
		BufferedWriter fout = new BufferedWriter(new FileWriter(filePath));
		fout.write(dict.size() + "\n");
		for (int i = 0; i < dict.size(); i++) {
			int freq = dict.getCount(i);
			String str = dict.getString(i);
			fout.write(i + "\t" + freq + "\t" + str + "\n");
		}
		fout.close();
		System.out.println("Saving count dictionary to:\t" + filePath);
	}
	
	public static CountDictionary loadCountDictionary(String filePath)
			throws IOException {
		BufferedReader fread = new BufferedReader(new FileReader(filePath));
		CountDictionary dict = new CountDictionary();
		int numNGrams = Integer.parseInt(fread.readLine());
		for (int i = 0; i < numNGrams; i++) {
			String[] info = fread.readLine().split("\t");
			dict.insertTuple(Integer.parseInt(info[0]), info[2],
					         Integer.parseInt(info[1]));
		}
		fread.close();
		System.out.println("Loaded count dictionary from:\t" + filePath);
		return dict;
	}
	
	public static void saveOptimizationHistory(OptimizationHistory optHistory,
			String filePath) throws IOException {
		 ArrayList<MLArray> mlObjects = new ArrayList<MLArray>();
		 for (String label : optHistory.history.keySet()) {
			 double[] arr = optHistory.history.get(label).toArray();
			 MLDouble mlArr = new MLDouble(label, arr, 1);
			 mlObjects.add(mlArr);
		 }
		 new MatFileWriter(filePath, mlObjects);
		 System.out.println("Saving optimization history to:\t" + filePath);
	}
	
	// File used for CoNLLEval input
	// output format: token, pos-tag, chunk-tag, gold-tag, pred-tag
	public static void savePrediction(NERCorpus corpus,
			ArrayList<NERSequence> instances, int[] instList,
			int[][] prediction, String filePath) throws IOException {
		BufferedWriter fout = new BufferedWriter(new FileWriter(filePath));
		for (int i : instList) {
			NERSequence instance = instances.get(i);
			for (int j = 0; j < instance.length; j++) {
				fout.write(String.format("%s %s %s %s %s\n",
						instance.getToken(j),
						instance.getPosTag(j),
						instance.getChunkTag(j),
						instance.getNERTag(j),
						corpus.nerDict.getString(prediction[i][j])));
			}
			fout.write("\n");
		}
		fout.close();
		System.out.println("Saving prediction to:\t" + filePath);
	}
}
