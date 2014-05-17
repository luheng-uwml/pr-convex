package data;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

public class NERCorpus {
	ArrayList<NERSequence> instances;
	CountDictionary tokenDict, posDict, chunkDict, nerDict;
	
	public NERCorpus() {
		this.tokenDict = new CountDictionary();
		this.posDict = new CountDictionary();
		this.chunkDict = new CountDictionary();
		this.nerDict = new CountDictionary();
		this.instances = new ArrayList<NERSequence>();
	}
	
	public NERCorpus(NERCorpus baseCorpus) {
		this.tokenDict = baseCorpus.tokenDict;
		this.posDict = baseCorpus.posDict;
		this.chunkDict = baseCorpus.chunkDict;
		this.nerDict = baseCorpus.nerDict;
		this.instances = new ArrayList<NERSequence>();
	}
	
	public NERCorpus(CountDictionary tokenDict, CountDictionary posDict,
			CountDictionary chunkDict, CountDictionary nerDict) {
		this.tokenDict = tokenDict;
		this.posDict = posDict;
		this.chunkDict = chunkDict;
		this.nerDict = nerDict;
		this.instances = new ArrayList<NERSequence>();
	}
	
	public void readFromCoNLL2003(String filePath) {
		BufferedReader fileReader;
		String currLine;
		ArrayList<String> tokens, posTags, chunkTags, nerTags;
		tokens = new ArrayList<String>();
		posTags = new ArrayList<String>();
		chunkTags = new ArrayList<String>();
		nerTags = new ArrayList<String>();
		try {
			fileReader = new BufferedReader(new FileReader(filePath));
			while ((currLine = fileReader.readLine()) != null) {
				if (currLine.length() == 0) {
					addSequenceFromBuffer(tokens, posTags, chunkTags, nerTags);
				} else {
					String[] info = currLine.split(" ");
					tokens.add(info[0]);
					posTags.add(info[1]);
					chunkTags.add(info[2]);
					nerTags.add(info[3]);
				}
			}
			addSequenceFromBuffer(tokens, posTags, chunkTags, nerTags);
			fileReader.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	private void addSequenceFromBuffer(ArrayList<String> tokens,
			ArrayList<String> posTags,
			ArrayList<String> chunkTags,
			ArrayList<String> nerTags) {
		int length = tokens.size();
		if (length == 0) {
			return;
		}
		int[] tokenIDs = new int[length], posTagIDs = new int[length],
			chunkTagIDs = new int[length], nerTagIDs = new int[length];
		for (int i = 0; i < length; i++) {
			tokenIDs[i] = tokenDict.addString(tokens.get(i));
			posTagIDs[i] = posDict.addString(posTags.get(i));
			chunkTagIDs[i] = chunkDict.addString(chunkTags.get(i));
			nerTagIDs[i] = nerDict.addString(nerTags.get(i));
		}
		int newInstanceID = instances.size();
		instances.add(new NERSequence(this, newInstanceID, tokenIDs, posTagIDs,
				chunkTagIDs, nerTagIDs));
		tokens.clear();
		posTags.clear();
		chunkTags.clear();
		nerTags.clear();
	}
	
	public int size() {
		return this.instances.size();
	}
	
	public static void main(String[] args) {
		NERCorpus corpus = new NERCorpus();
		corpus.readFromCoNLL2003("./data/eng.train");
		System.out.println("Read " + corpus.size() + " sentences.");
		for (NERSequence instance : corpus.instances) {
			System.out.println(instance.toString());
		}
	}
}



