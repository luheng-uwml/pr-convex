package experiment;

import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;

public class ExperimentConfig {
	@Option(name = "-num-labels", usage="")
	public int numLabeled = -1;
	
	@Option(name = "-graph", usage="")
	public boolean useGraph = false;
	
	@Option(name = "-ssl", usage="")
	public boolean sslTraining = false;
	
	@Option(name = "-min-ff", usage="")
	public int featureFreqCutOff = 1;
	
	@Option(name = "-lambda1", usage="")
	public double lambda1 = 10;
	
	@Option(name = "-lambda2", usage="")
	public double lambda2 = 0;
	
	@Option(name = "-matpath", usage="")
	public String logFilePath = "./experiments/temp.mat";
	
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
