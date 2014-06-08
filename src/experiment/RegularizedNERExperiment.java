package experiment;

import java.io.IOException;
import java.util.ArrayList;

import org.kohsuke.args4j.CmdLineParser;

import optimization.*;
import data.CountDictionary;
import data.Evaluator;
import data.IOHelper;
import data.NERCorpus;
import data.NERSequence;
import feature.*;
import graph.*;
import gnu.trove.list.array.TIntArrayList;

public class RegularizedNERExperiment {
	
	private static void runRegularizedExperiment(ExperimentConfig config) {
		/**
		 * Prepare data
		 */
		System.out.println("Train:");
		NERCorpus corpusTrain = new NERCorpus();
		corpusTrain.readFromCoNLL2003("./data/eng.train");
		corpusTrain.printCorpusInfo();
		
		System.out.println("Dev A:");
		NERCorpus corpusDevA = new NERCorpus(corpusTrain, false);
		corpusDevA.readFromCoNLL2003("./data/eng.testa");
		corpusDevA.printCorpusInfo();
		
		System.out.println("Dev B:");
		NERCorpus corpusDevB = new NERCorpus(corpusTrain, false);
		corpusDevB.readFromCoNLL2003("./data/eng.testb");
		corpusDevB.printCorpusInfo();
		
		ArrayList<NERSequence> allInstances = new ArrayList<NERSequence>();
		ArrayList<NERSequence> trainInstances = new ArrayList<NERSequence>();
		int numTrains = 0;
		if (!config.useToyData) {
			allInstances.addAll(corpusTrain.instances);
			numTrains = corpusTrain.instances.size();
		}
		if (config.useDevA) {
			allInstances.addAll(corpusDevA.instances);
			// use devA as part of training here
			if (config.useDevB) {
				numTrains += corpusDevA.instances.size();
			}
		}
		if (config.useDevB) {
			allInstances.addAll(corpusDevB.instances);
		}
		// assign labels and instance IDs
		int numInstances = allInstances.size();
		int[][] labels = new int [numInstances][];
		TIntArrayList trainList = new TIntArrayList(),
				  	  devList = new TIntArrayList();
		for (int i = 0; i < numInstances; i++) {
			if (i < numTrains) {
				trainList.add(i);
				trainInstances.add(allInstances.get(i));
			} else {
				devList.add(i);
			}
			labels[i] = allInstances.get(i).getLabels();
		}
		System.out.println("Number of training instances:\t" + numTrains +
				  		   "\tNumber of test instances:\t" +
				  		   (numInstances - numTrains));
		// extract features
		NERFeatureExtractor extractor = new NERFeatureExtractor(corpusTrain,
				trainInstances, config.featureFreqCutOff);
		extractor.printInfo();
		SequentialFeatures features = extractor.getSequentialFeatures(
				allInstances);
		Evaluator eval = new Evaluator(corpusTrain);
		
		// get graph ready
		GraphRegularizer graph = new DummyGraphRegularizer(
				features.numTargetStates);
		if (config.useGraph) {
			CountDictionary ngramDict = null;
			SparseVector[] edges = null;
			try {
				ngramDict = IOHelper.loadCountDictionary(
						config.ngramFilePath);
				edges = IOHelper.loadSparseVectors(
						config.graphFilePath);
			} catch (IOException e) {
				e.printStackTrace();
			}
			NGramFeatureExtractor ngramExtractor = new NGramFeatureExtractor(
					corpusTrain, allInstances, 3, true, ngramDict);
			double goldPenalty = graph.computeTotalPenalty(labels);
			System.out.println("gold penalty::\t" + goldPenalty);
			graph = new GraphRegularizer(ngramExtractor.getNGramIDs(), edges,
										 features.numTargetStates);
			graph.validate(labels, corpusTrain.nerDict,
					       ngramExtractor.ngramDict);
		}
		AbstractOptimizer optimizer = null;
		if (config.sslTraining) {
			optimizer = new FeatureRescaledEGTrainer(
						features, graph, labels, trainList.toArray(),
						devList.toArray(), eval,
						config.lambda1, config.lambda2,
						config.initialLearningRate,
						config.maxNumIterations,
						config.randomSeed);
		} else if (config.pqlTraining) {
			optimizer = new PQEGTrainer(features,
						graph, labels, trainList.toArray(), devList.toArray(),
						eval, config.lambda1, config.lambda2, 1.0,
						config.initialLearningRate,
						config.maxNumIterations, config.randomSeed);
		}
		else {
			optimizer = new SupervisedEGTrainer(features, graph,
						labels, trainList.toArray(), devList.toArray(), eval,
						config.lambda1, config.lambda2,
						config.initialLearningRate, config.maxNumIterations,
						config.randomSeed);
		}
		optimizer.optimize();
		try {
			IOHelper.saveOptimizationHistory(optimizer.getOptimizationHistory(),
					config.matFilePath);
			IOHelper.savePrediction(corpusTrain, allInstances,
					devList.toArray(), optimizer.getPrediction(),
					config.predPath);
		} catch (IOException e) {
			e.printStackTrace();
		}
		
	}
	
	public static void main(String[] args) {
		ExperimentConfig config = new ExperimentConfig(args);
		runRegularizedExperiment(config);
	}
}
