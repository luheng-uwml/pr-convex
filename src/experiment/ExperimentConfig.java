package experiment;

import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;

public class ExperimentConfig {
	@Option(name = "-num-labels", usage="")
	public int numLabeled = 1000;
	
	@Option(name = "-graph", usage="")
	public boolean useGraph = false;
	
	@Option(name = "-ssl", usage="")
	public boolean sslTraining = false;
	
	@Option(name = "-pq", usage="") // the dummy varaible "q" version
	public boolean pqlTraining = true;
	
	@Option(name = "-min-ff", usage="")
	public int featureFreqCutOff = 1;
	
	@Option(name = "-lambda1", usage="")
	public double lambda1 = 0.1;
	
	@Option(name = "-lambda2", usage="")
	public double lambda2 = 1;
	
	@Option(name = "-max-iters", usage="")
	public int maxNumIterations = 1000;
	
	@Option(name = "-matpath", usage="")
	public String logFilePath = "./experiments/temp.mat";
	
	@Option(name = "-pred-path", usage="")
	public String predPath = "./experiments/temp.pred";
	
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
