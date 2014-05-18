package data;

import java.util.ArrayList;

import gnu.trove.map.hash.TObjectIntHashMap;

public class CountDictionary {
	TObjectIntHashMap<String> str2index;
	ArrayList<String> index2str;
	ArrayList<Integer> index2count;
	
	public CountDictionary() {
		this.str2index = new TObjectIntHashMap<String>();
		this.index2str = new ArrayList<String>();
		this.index2count = new ArrayList<Integer>();
	}
	
	// Copy from existing dictionary.
	public CountDictionary(CountDictionary dict) {
		this();
		for (int sid = 0; sid < dict.size(); sid ++) {
			String str = dict.getString(sid);
			index2str.add(str);
			str2index.put(str, sid);
			index2count.set(sid, dict.getCount(sid));
		}
	}
	
	public CountDictionary(CountDictionary dict, int minFrequency) { 
		this();
		for (int sid = 0; sid < dict.size(); sid ++) {
			int freq = dict.getCount(sid);
			if (freq < minFrequency) {
				continue;
			}
			String str = dict.getString(sid);
			int newSID = index2str.size(); 
			index2str.add(str);
			str2index.put(str, newSID);
			index2count.set(newSID, dict.getCount(sid));
		}
	}
	
	public int addString(String str) {
		if (str2index.contains(str)) {
			int sid = str2index.get(str);
			int count = index2count.get(sid);
			index2count.set(sid, count + 1);
			return sid;
		} else {
			int sid = index2str.size();
			index2str.add(str);
			index2count.add(1);
			str2index.put(str, sid);
			return sid;
		}
	}
	
	public int addString(String str, String unseenMarker) {
		return str2index.contains(str) ? addString(str) :
				addString(unseenMarker);
	}
	
	public int lookupString(String str) {
		if (!str2index.contains(str)) {
			return -1;
		}
		return str2index.get(str);
	}
	
	public int getCount(String str) {
		if (!str2index.contains(str)) {
			return 0;
		}
		return index2count.get(str2index.get(str));
	}
	
	public int getCount(int index) {
		return (index < index2count.size()) ? 0 : index2count.get(index); 
	}
	
	public int size() {
		return index2str.size();
	}
	
	public String getString(int index) {
		return index2str.get(index);
	}
}
