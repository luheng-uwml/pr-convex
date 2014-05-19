package data;

import java.util.ArrayList;

public class Evaluator {
	int ignoreTagID;
	
	public Evaluator(NERCorpus corpus) {
		ignoreTagID = corpus.nerDict.lookupString("O");
	}
	
	public int[] evaluate(int[] gold, int[] predict) {
		ArrayList<int[]> goldSpans = getSpanList(gold);
		ArrayList<int[]> predSpans = getSpanList(predict);
		int numGold = goldSpans.size();
		int numPred = predSpans.size();
		int numMatched = getMatchedSpans(gold, goldSpans, predSpans);
		/*
		double precision = 1.0 * numMatched / numPred;
		double recall = 1.0 * numMatched / numGold;
		double f1 = (precision + recall > 0) ?
				(2 * precision * recall) / (precision + recall) : 0.0;
		*/  
		int[] result = { numGold, numPred, numMatched };
		return result;
	}
	
	private ArrayList<int[]> getSpanList(int[] tags) {
		int length = tags.length, spanStart = -1, spanTag = -1;
		ArrayList<int[]> spans = new ArrayList<int[]>();
		for (int i = 0; i < length; i++) {
			if (tags[i] == ignoreTagID) {
				continue;
			}
			if (i == 0 || tags[i] != tags[i - 1]) {
				// detecting span opening
				spanStart = i;
				spanTag = tags[i];
			}
			if (i == length - 1 || tags[i] != tags[i + 1]) {
				// detecting span end
				int[] span = { spanStart, i, spanTag };
				spans.add(span); 
				spanStart = -1;
			}
		}
		return spans;
	}
	
	private int getMatchedSpans(int[] tags,
			ArrayList<int[]> spanList1, ArrayList<int[]> spanList2) {
		int length = tags.length;
		int[][] spanMap = new int[tags.length][tags.length];
		int matched = 0;
		for (int i = 0; i < length; i++) {
			for (int j = 0;j  < length; j++) {
				spanMap[i][j] = -1;
			}
		}
		for (int[] span : spanList1) {
			spanMap[span[0]][span[1]] = span[2];
		}
		for (int[] span : spanList2) {
			if (spanMap[span[0]][span[1]] == span[2]) {
				matched ++;
			}
		}
		return matched;
	}
}
