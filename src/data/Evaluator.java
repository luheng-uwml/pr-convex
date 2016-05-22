package data;

import java.io.*;
import java.util.ArrayList;

public class Evaluator {
	CountDictionary tagDict;
	int ignoreTagID;
	
	public Evaluator(NERCorpus corpus) {
		tagDict = corpus.nerDict;
		ignoreTagID = tagDict.lookupString("O");
	}
	
	public int[] evaluate(int[] gold, int[] predict) {
		ArrayList<int[]> goldSpans = getSpanList(gold);
		ArrayList<int[]> predSpans = getSpanList(predict);
		int numGold = goldSpans.size();
		int numPred = predSpans.size();
		int numMatched = getMatchedSpans(gold, goldSpans, predSpans);
		int[] result = { numGold, numPred, numMatched };
		//System.out.println("::::\t" + numGold + "\t, " + numPred + "\t" + numMatched);
		return result;
	}

	public void runCoNLLEval(NERCorpus corpus, int[][] predictions) {
		try {
			BufferedWriter writer = new BufferedWriter(new FileWriter(new File("eval.temp")));
			for (int t = 0; t < corpus.instances.size(); t++) {
				final NERSequence instance = corpus.instances.get(t);
				for (int i = 0; i < instance.length; i++) {
					writer.write(String.format("%s %s %s %s %s\n",
							instance.getToken(i),
							instance.getPosTag(i),
							instance.getChunkTag(i),
							instance.getNERTag(i),
							corpus.nerDict.getString(predictions[t][i])));
				}
				writer.write("\n");
			}
			writer.close();
			ProcessBuilder builder = new ProcessBuilder("./data/conlleval.txt");
			builder.redirectInput(new File("eval.temp"));
			Process pr = builder.start();
			pr.waitFor();

			BufferedReader stdInput = new BufferedReader(new InputStreamReader(pr.getInputStream()));
			BufferedReader stdError = new BufferedReader(new InputStreamReader(pr.getErrorStream()));

			String s = null;
			while ((s = stdInput.readLine()) != null) {
				System.out.println(s);
			}
			while ((s = stdError.readLine()) != null) {
				System.out.println(s);
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	private ArrayList<int[]> getSpanList(int[] tags) {
		int length = tags.length, spanStart = -1;
		String spanName = "";
		ArrayList<int[]> spans = new ArrayList<>();
		for (int i = 0; i < length; i++) {
			String tag = tagDict.getString(tags[i]);
			if (spanStart > -1 && (!tag.endsWith(spanName) || tag.charAt(0) == 'B')) {
				int[] span = { spanStart, i - 1, tags[spanStart] };
				spans.add(span);
				spanStart = -1;
			}
			if (tag.charAt(0) == 'B' || (spanStart == -1 && tag.charAt(0) == 'I')) {
				spanStart = i;
				spanName = tag.substring(2);
			}
		}
		if (spanStart > -1) {
			int[] span = { spanStart, length - 1, tags[spanStart] };
			spans.add(span);
		}
		return spans;
	}
	
	private int getMatchedSpans(int[] tags, ArrayList<int[]> spanList1, ArrayList<int[]> spanList2) {
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
