package data;

import java.io.BufferedWriter;
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
		for (int i = 0; i < vecs.length; i++) {
			SparseVector fvec = vecs[i];
			if (fvec == null || fvec.length == 0) {
				fout.write("0\n");
				continue;
			}
			fout.write(fvec.length);
			for (int j = 0; j < fvec.length; j++) {
				fout.write(String.format("\t%d:%.12f", fvec.indices[j],
						fvec.values[j]));
			}
			fout.write("\n");
		}
		fout.close();
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
	}
}
