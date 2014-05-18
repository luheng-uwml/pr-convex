package data;

public class NERSequence {
	NERCorpus corpus;
	public int sequenceID, length;
	public int[] tokens, posTags, chunkTags, nerTags;
	
	public NERSequence(NERCorpus corpus, int sequenceID, int[] tokens,
			int[] posTags, int[] chunkTags, int[] nerTags) {
		this.corpus = corpus;
		this.sequenceID = sequenceID;
		this.tokens = tokens;
		this.posTags = posTags;
		this.chunkTags = chunkTags;
		this.nerTags = nerTags;
		this.length = tokens.length;
	}
	
	@Override
	public String toString() {
		String retString = "";
		retString += IOHelper.printJoin(tokens, corpus.tokenDict) + "\n";
		retString += IOHelper.printJoin(nerTags, corpus.nerDict) + "\n";
		return retString;
	}
	
	public String getToken(int i) {
		return corpus.tokenDict.getString(tokens[i]);
	}
	
	public String getPosTag(int i) {
		return corpus.posDict.getString(posTags[i]);
	}
	
	public String getChunkTag(int i) {
		return corpus.chunkDict.getString(chunkTags[i]);
	}
}
