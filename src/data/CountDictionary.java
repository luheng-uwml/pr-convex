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
