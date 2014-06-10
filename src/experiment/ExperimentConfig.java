package experiment;

import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;

public class ExperimentConfig {
	@Option(name = "-num-labels", usage="")
	public int numLabeled = -1;
	
	@Option(name = "-use-deva", usage="")
	public boolean useDevA;
	
	@Option(name = "-use-devb", usage="")
	public boolean useDevB;
	
	@Option(name = "-toy", usage="")
	public boolean useToyData;
	
	@Option(name = "-graph", usage="")
	public boolean useGraph;
	
	@Option(name = "-ssl", usage="")
	public boolean sslTraining;
	
	@Option(name = "-pq", usage="") // the dummy varaible "q" version
	public boolean pqlTraining;
	
	@Option(name = "-min-ff", usage="")
	public int featureFreqCutOff = 1;
	
	@Option(name = "-lambda1", usage="")
	public double lambda1 = 1;
	
	@Option(name = "-lambda2", usage="")
	public double lambda2 = 1;
	
	@Option(name = "-init-eta", usage="")
	public double initialLearningRate = 0.5;
	
	@Option(name = "-max-iters", usage="")
	public int maxNumIterations = 1000;
	
	@Option(name = "-warm-iters", usage="")
	public int warmStartIterations = 100;
	
	@Option(name = "-rand-seed", usage="")
	public int randomSeed = 12345;
	
	@Option(name = "-mat-path", usage="")
	public String matFilePath = "./experiments/temp.mat";
	
	@Option(name = "-pred-path", usage="")
	public String predPath = "./experiments/temp.pred";
	
	@Option(name = "-ngram-path", usage="")
	public String ngramFilePath = "temp.ngrams";
	
	@Option(name = "-graph-path", usage="")
	public String graphFilePath = "temp.edges";

	@Option(name = "-stop", usage="")
	public double stoppingCriterion = 1e-6;
	
	public ExperimentConfig(String[] args) {
		CmdLineParser parser = new CmdLineParser(this);
		parser.setUsageWidth(120);
		try {
			parser.parseArgument(args);
		} catch (CmdLineException e) {
			e.printStackTrace();
		}
	}
}
