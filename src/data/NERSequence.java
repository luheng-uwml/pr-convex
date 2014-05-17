package data;

public class NERSequence {
	int sequenceID, length;
	int[] tokens, posTags, chunkTags, nerTags;
	
	public NERSequence(int sequenceID, int[] tokens, int[] posTags,
			int[] chunkTags, int[] nerTags) {
		this.sequenceID = sequenceID;
		this.tokens = tokens;
		this.posTags = posTags;
		this.length = tokens.length;
	}
	
	
}
