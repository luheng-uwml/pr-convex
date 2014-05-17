package data;

import java.util.ArrayList;

public class NERCorpus {
	ArrayList<NERSequence> instances;
	CountDictionary tokenDict, posDict, chunkDict, nerDict;
	
	public NERCorpus(CountDictionary tokenDict, CountDictionary posDict,
			CountDictionary chunkDict, CountDictionary nerDict) {
		this.tokenDict = tokenDict;
		this.posDict = posDict;
		this.chunkDict = chunkDict;
		this.nerDict = nerDict;
		this.instances = new ArrayList<NERSequence>();
	}
	
	public void readFromCoNLL2003(String[] files) {
		for (String filename : files) {
			
		}
	}
	
	public int size() {
		return this.instances.size();
	}
}


