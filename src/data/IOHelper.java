package data;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

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
}
