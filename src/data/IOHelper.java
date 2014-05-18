package data;

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
}
