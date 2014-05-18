package feature;

public class RegexHelper {
	
	public static boolean isNumerical(String token) {
		return token.trim().matches("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+");
	}
	
	public static boolean isPunctuation(String token) {
		/*
		if(lang.startsWith("german")) {
			return token.matches("(\\$.*)");
		}
		else if(lang.startsWith("swedish")) {
			return token.matches("(I[^DM])");
		}
		else {
		*/
		return token.matches("[.,!?:;]+");
	}
}
