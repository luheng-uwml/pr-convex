package experiment;

import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;

public class GraphBuildConfig {
	@Option(name = "-num-neighbors", usage="")
	public int numNeighbors = 20;
	
	@Option(name = "-lowercase", usage="")
	public boolean lowercaseNGrams;
	
	@Option(name = "-minw", usage="")
	public double edgeWeightThreshold = 0.3;
	
	@Option(name = "-num-threads", usage="")
	public int numThreads = 8;
	
	@Option(name = "-use-deva", usage="")
	public boolean useDevA;
	
	@Option(name = "-use-devb", usage="")
	public boolean useDevB;
	
	@Option(name = "-use-word-embedding", usage="")
	public boolean useWordEmbedding;
	
	@Option(name = "-ngram-path", usage="")
	public String ngramFilePath = "temp.ngrams";
	
	@Option(name = "-graph-path", usage="")
	public String graphFilePath = "temp.edges";
	
	public GraphBuildConfig(String[] args) {
		CmdLineParser parser = new CmdLineParser(this);
		parser.setUsageWidth(120);
		try {
			parser.parseArgument(args);
		} catch (CmdLineException e) {
			e.printStackTrace();
		}
	}
}
