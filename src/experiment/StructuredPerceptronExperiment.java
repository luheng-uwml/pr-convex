package experiment;

import data.Evaluator;
import data.NERCorpus;
import data.NERSequence;
import feature.NERFeatureExtractor;
import feature.SequentialFeatures;
import gnu.trove.list.array.TIntArrayList;
import optimization.OnlineExponentiatedGradientDescent;
import optimization.StructuredPerceptron;

import java.util.ArrayList;

/**
 * Created by luheng on 5/22/16.
 */
public class StructuredPerceptronExperiment {
    public static void main(String[] args) {
        NERCorpus corpusTrain = new NERCorpus();
        corpusTrain.readFromCoNLL2003("./data/eng.train");

        NERCorpus corpusDev = new NERCorpus(corpusTrain, false);
        corpusDev.readFromCoNLL2003("./data/eng.testa");

        NERCorpus corpusTest = new NERCorpus(corpusTrain, false);
        corpusTest.readFromCoNLL2003("./data/eng.testb");

        corpusTrain.printCorpusInfo();
        corpusDev.printCorpusInfo();
        corpusTest.printCorpusInfo();

        int numAllTokens = 0;
        for (NERSequence instance : corpusTrain.instances) {
            numAllTokens += instance.length;
        }
        System.out.println("Number of all tokens:\t" + numAllTokens);

        ArrayList<NERSequence> allInstances = new ArrayList<>();
        allInstances.addAll(corpusTrain.instances);
        allInstances.addAll(corpusDev.instances);
        allInstances.addAll(corpusTest.instances);

        int numTrains = corpusTrain.instances.size();
        int numDev = corpusTrain.instances.size() + corpusDev.instances.size();
        int[][] labels = new int[allInstances.size()][];
        TIntArrayList trainList = new TIntArrayList(), devList = new TIntArrayList(), testList = new TIntArrayList();
        for (int i = 0; i < allInstances.size(); i++) {
            labels[i] = i < numTrains ? corpusTrain.instances.get(i).getLabels() :
                                i < numDev ? corpusDev.instances.get(i - numTrains).getLabels() :
                                                corpusTest.instances.get(i - numDev).getLabels();
            if (i < numTrains) {
            //if (i < 1000) {
                trainList.add(i);
            } else if (i >= numTrains && i < numDev) {
                devList.add(i);
            } else if (i >= numDev) {
                testList.add(i);
            }
        }
        NERFeatureExtractor extractor = new NERFeatureExtractor(corpusTrain, corpusTrain.instances, 5);
        extractor.printInfo();
        SequentialFeatures features = extractor.getSequentialFeatures(allInstances);
        Evaluator eval = new Evaluator(corpusTrain);

        StructuredPerceptron optimizer = new StructuredPerceptron(features, labels, trainList.toArray(),
                devList.toArray(), eval, 1.0, 20, 12345);
        optimizer.optimize();
        int[][] testPredictions = optimizer.getPredictions(features, testList.toArray());
        eval.runCoNLLEval(corpusTest, testPredictions);
    }
}
