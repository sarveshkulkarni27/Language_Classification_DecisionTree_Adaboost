/**
 * The program trains a model and predicts the output
 * using Decision Tree or Adaboost
 * 
 * 
 * @author Sarvesh Kulkarni
 * 
 */

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Scanner;

public class Classify implements Serializable{
	static int depth = 0;
	int node = 0;

	/**
	 * Loads the file
	 * 
	 * @param 		scanner				Scanner object
	 * 
	 * @param 		xyValues			List to store the value
	 * 
	 * @param		args				Train or Predict
	 * 
	 * @return							List of values
	 */
	
	private List<List<String>> readFile(Scanner scanner, List<List<String>> xyValues, String args) {
		while(scanner.hasNextLine()) {
			String line = ""; 
			line = scanner.nextLine();
			if(line.isEmpty()) {
				continue;
			}
			xyValues.add(getFeatures(line, args));
		}
		return xyValues;
	}

	/**
	 * Features to train the model
	 * 
	 * @param 		line				Input line
	 * 
	 * @param 		args				Train or predict
	 * 
	 * @return							True or False
	 */
	
	public List<String> getFeatures(String line, String args) {
		String lineToProcess;
		String label;
		if(args.equals("train")) {
			lineToProcess = line.substring(3);
			label = line.substring(0, 2);
		}else {
			lineToProcess = line;
			label = "";	
		}
		int count = 0;
		List<String> featureList = new ArrayList<String>();

		//De or Het
		for(String word : lineToProcess.split(" ")) {
			if(word.strip().toLowerCase().equals("de") ||
					word.toLowerCase().equals("het")) {
				count++;
			}
		}
		if(count > 0) {
			featureList.add("True");
		}else {
			featureList.add("False");
		}

		//prepositions
		count = 0;
		for(String word : lineToProcess.split(" ")) {
			if(word.strip().toLowerCase().contains("in") || word.strip().toLowerCase().contains("at")
					|| word.strip().toLowerCase().contains("on") || word.strip().toLowerCase().contains("of")
					|| word.strip().toLowerCase().contains("to")
					) {
				count++;
			}
		}
		if(count > 0) {
			featureList.add("False");
		}else {
			featureList.add("True");
		}

		//The
		count = 0;
		for(String word : lineToProcess.split(" ")) {
			if(word.strip().toLowerCase().equals("the")) {
				count++;
			}
		}
		if(count > 0) {
			featureList.add("False");
		}else {
			featureList.add("True");
		}

		//vowels
		count = 0;
		for(String word : lineToProcess.split(" ")) {
			if(word.strip().toLowerCase().contains("aa") || word.strip().toLowerCase().contains("ee")
					|| word.strip().toLowerCase().contains("ii") || word.strip().toLowerCase().contains("oo")
					|| word.strip().toLowerCase().contains("uu")) {
				count++;
			}
		}
		if(count > 0) {
			featureList.add("True");
		}else {
			featureList.add("False");
		}

		//ij
		count = 0;
		for(String word : lineToProcess.split(" ")) {
			if(word.strip().toLowerCase().contains("ij")) {
				count++;
			}
		}
		if(count > 0) {
			featureList.add("True");
		}else {
			featureList.add("False");
		}


		//conjunctions
		count = 0;
		for(String word : lineToProcess.split(" ")) {
			if(word.strip().toLowerCase().contains("and") || word.strip().toLowerCase().contains("or")
					|| word.strip().toLowerCase().contains("but") || word.strip().toLowerCase().contains("yet")
					) {
				count++;
			}
		}
		if(count > 0) {
			featureList.add("False");
		}else {
			featureList.add("True");
		}

		if(args.equals("train")){
			featureList.add(label);
		}


		return featureList;
	}

	/**
	 * Get the true and false values in a column
	 * 
	 * @param 		remainderList			List of remainder values
	 * 
	 * @param 		xyValues				List of values
	 * 
	 * @return								Map of true and false values
	 */

	public Map<String, List<Integer>> getValues(List<Double> remainderList, List<List<String>> xyValues) {
		int index = remainderList.indexOf(Collections.min(remainderList));

		Map<String, List<Integer>> trueFalse = new HashMap<String, List<Integer>>();

		List<Integer> trueList = new ArrayList<Integer>();
		List<Integer> falseList = new ArrayList<Integer>();		
		for( int xValue = 0; xValue < xyValues.get(0).size(); xValue++) {
			if(xValue != index) {
				continue;
			}
			for(int yValue = 0; yValue < xyValues.size(); yValue++) {
				if((xyValues.get(yValue).get(xValue)).equals("True")) {
					trueList.add(yValue);
				}
				if((xyValues.get(yValue).get(xValue)).equals("False")) {
					falseList.add(yValue);
				}
			}
		}
		trueFalse.put("True", trueList);
		trueFalse.put("False", falseList);
		return trueFalse;
	}

	/**
	 * Calculate the remainder value
	 * 
	 * @param 		total				Total count
	 * 
	 * @param 		countTrue			Total count of True values
	 * 
	 * @param 		countTrueA			Total count of A labels in True values
	 * 
	 * @param 		countFalseA			Total count of A labels in False values
	 * 
	 * @param 		remainderList		List of remainder values
	 * 
	 * @return							Remainder list
	 */

	public List<Double> calculateEntropy(double total, double countTrue, double countTrueA, double countFalseA, List<Double> remainderList) {
		double countFalse = total - countTrue;

		double countTrueB = countTrue - countTrueA;

		double countFalseB = countFalse - countFalseA;

		double remainder = 0;

		if(countTrue == 0 || countFalse == 0) {
			remainderList.add(1.00);
		}else {			
			remainder = ((countTrue/total) 
					* (getEntropy(countTrueA, countTrue) + getEntropy(countTrueB, countTrue))) 
					+ ((countFalse/total) 
							* (getEntropy(countFalseA, countFalse) + getEntropy(countFalseB, countFalse)));

			remainderList.add(remainder);
		}
		return remainderList;

	}

	/**
	 * Calculate the values for remainder
	 * 
	 * @param 		countTrueA			Total count of A labels
	 * 
	 * @param 		countTrue			Total count
	 * 
	 * @return							Value
	 */

	public double getEntropy(double countTrueA, double countTrue) {
		double value = 0;
		if(countTrueA != 0 ) {
			value = (countTrueA/countTrue) * logValue(1/(countTrueA/countTrue));
		}else {
			value = 0;
		}
		return value;
	}

	/**
	 * Count the values in a column
	 * 
	 * @param 		xyValues			List of values
	 * 
	 * @return							List of remainder values
	 */

	public List<Double> countValues(List<List<String>> xyValues) {
		List<Double> remainderList = new ArrayList<Double>();
		for( int xValue = 0; xValue < xyValues.get(0).size() - 1; xValue++) {
			int total = 0;
			int countTrue = 0;
			int countTrueA = 0;
			int countFalseA = 0;
			for(int yValue = 0; yValue < xyValues.size(); yValue++) {
				total++;
				if((xyValues.get(yValue).get(xValue)).equals("True")) {
					countTrue++;
					if((xyValues.get(yValue).get(xyValues.get(0).size() - 1)).equals("nl")){
						countTrueA++;
					}
				}
				if((xyValues.get(yValue).get(xValue)).equals("False")) {
					if((xyValues.get(yValue).get(xyValues.get(0).size() - 1)).equals("nl")) {
						countFalseA++;
					}
				}
			}
			remainderList = calculateEntropy(total, countTrue, countTrueA, countFalseA, remainderList);

		}
		return remainderList;

	}

	/**
	 * Calculate the log value
	 * 
	 * @param 		value			Value to calculate log
	 * 
	 * @return						Log value
	 */

	public static double logValue(double value)
	{
		if(value == 0)
			return 0;
		return (double) (Math.log(value) / Math.log(2));
	}

	/**
	 * Recursively call based on True and False
	 * 
	 * @param 		xyValues			List of values
	 * 
	 * @param 		depth				Depth of the tree
	 * 
	 * @param 		trueList			True values for recursive call
	 * 
	 * @param 		falseList			False values for recursive call
	 * 
	 * @return 
	 */

	public Node recursion(List<List<String>> xyValues, int depth, List<Integer> trueList, List<Integer> falseList, double min, Node node) {

		List<Double> remainderList = new ArrayList<Double>();
		Map<String, List<Integer>> trueFalse = new HashMap<String, List<Integer>>();
		trueList = new ArrayList<Integer>();
		falseList = new ArrayList<Integer>();

		List<List<String>> trueXYList = new ArrayList<List<String>>();
		List<List<String>> falseXYList = new ArrayList<List<String>>();

		remainderList = countValues(xyValues);

		int columnIndex = remainderList.indexOf(Collections.min(remainderList));

		min = Collections.min(remainderList);
		if(min == 1.0) {
			int countA = 0;
			int countB = 0;
			for(int index = 0; index < xyValues.size(); index++) {
				if(xyValues.get(index).get(xyValues.get(0).size() - 1).equals("nl")) {
					countA++;
				}else {
					countB++;
				}
			}
			Map<Integer, Integer> baseMap = new HashMap<Integer, Integer>();
			baseMap.put(1, countA);
			baseMap.put(2, countB);

			return new Node(baseMap);

		}else {
			int countA = 0;
			int countB = 0;
			for(int index = 0; index < xyValues.size(); index++) {
				if(xyValues.get(index).get(xyValues.get(0).size() - 1).equals("nl")) {
					countA++;
				}else {
					countB++;
				}
			}


			trueFalse = getValues(remainderList, xyValues);
			trueList.addAll(trueFalse.get("True"));
			falseList.addAll(trueFalse.get("False"));

			for(int index = 0; index < xyValues.size(); index++) {
				if(trueList.contains(index)) {
					trueXYList.add(xyValues.get(index));
				}
				if(falseList.contains(index)) {
					falseXYList.add(xyValues.get(index));
				}
			}
			depth++;
			if(node == null) {
				node = new Node(columnIndex);
			}
			node.left = recursion(trueXYList, depth, trueList, falseList, min, node.left);
			node.right = recursion(falseXYList, depth, trueList, falseList, min, node.right);

			Map<Integer, Integer> countMap = new HashMap<Integer, Integer>();
			countMap.put(1, countA);
			countMap.put(2, countB);

			return new Node(columnIndex, node.left, node.right, countMap);

		}


	}

	/**
	 * Train a model using Adaboost
	 * 
	 * @param 			xyValues			Store the values
	 * 
	 * @param 			trueList			List of True values
	 * 
	 * @param 			falseList			List of False values
	 * 
	 * @param 			min					Minimum remainder
	 * 
	 * @return								Hypothesis
	 */
	
	public Hypothesis getAdaboost(List<List<String>> xyValues, List<Integer> trueList, List<Integer> falseList, double min) {
		Hypothesis hyp = null;

		for(int k = 0; k < 5; k++) {		
			List<Double> remainderList = new ArrayList<Double>();
			Map<String, List<Integer>> trueFalse = new HashMap<String, List<Integer>>();
			trueList = new ArrayList<Integer>();
			falseList = new ArrayList<Integer>();

			remainderList = countValues(xyValues);

			min = Collections.min(remainderList);


			trueFalse = getValues(remainderList, xyValues);
			trueList.addAll(trueFalse.get("True"));
			falseList.addAll(trueFalse.get("False"));

			List<Integer> errorList = new ArrayList<Integer>();

			double c = 0;
			for( int xValue = 0; xValue < xyValues.get(0).size(); xValue++) {
				if(xValue != xyValues.get(0).size() - 1) {
					continue;
				}
				for(int yValue = 0; yValue < xyValues.size(); yValue++) {
					if(falseList.contains(yValue) && 
							(xyValues.get(yValue).get(xyValues.get(0).size() - 1)).equals("nl")) {
						c++;
						errorList.add(yValue);
					}
					if(trueList.contains(yValue) && 
							(xyValues.get(yValue).get(xyValues.get(0).size() - 1)).equals("en")) {
						c++;
						errorList.add(yValue);
					}
				}

			}
			int size = xyValues.size();
			
			double sampleWeight = 1 / (double)size;

			double incorrectCount = c/xyValues.size();

			double updateRatio = (c/xyValues.size()) / ((xyValues.size() - c) / xyValues.size());

			double hypothesisAlpha = Math.log((1 - incorrectCount) / incorrectCount);

			double updatedCorrectWeight = sampleWeight * updateRatio;

			double sumUpdatedForNormalization = (c * sampleWeight) + ((xyValues.size() - c) * sampleWeight * updatedCorrectWeight);

			double incorrectNormalizedWeight = sampleWeight / sumUpdatedForNormalization;

			double correctNormalizedWeight = updatedCorrectWeight / sumUpdatedForNormalization;

			List<Double> bucketList = new ArrayList<Double>();
			double total = 0;

			for(int index = 0; index < size; index++) {
				if(errorList.contains(index)) {
					total += incorrectNormalizedWeight;
					bucketList.add(total);
				}else {
					total += correctNormalizedWeight;
					bucketList.add(total);
				}
			}

			Random random = new Random();

			List<List<String>> copyList = new ArrayList<List<String>>();

			for(int index = 0; index < size; index++) {
				double randomNumber = random.nextDouble();
				for(int value = 0; value < size; value++) {
					if(bucketList.get(value) >= randomNumber) {
						copyList.add(xyValues.get(value));
						break;
					}
				}

			}

			xyValues = new ArrayList<List<String>>();
			xyValues.addAll(copyList);
			copyList = new ArrayList<List<String>>();
			if(k == 0) {
				hyp = new Hypothesis(hypothesisAlpha);	
			}else {
				hyp.next = new Hypothesis(hypothesisAlpha, hyp.next);
			}
		}

		return hyp;
	}

	/**
	 * Serialize Decision Tree
	 * 
	 * @param 			node				Node object
	 * 
	 * @param 			args				Output file
	 * 
	 * @throws IOException
	 */
	
	public void serializeModel(Node node, String args) throws IOException {
		FileOutputStream outputFile = new FileOutputStream(args);
		ObjectOutputStream out = new ObjectOutputStream(outputFile);
		out.writeObject(node);
		out.close();
		outputFile.close();		
	}

	/**
	 * Deserialize the object
	 * 
	 * @param 		args										Input File
	 * 
	 * @param 		args2										Test File
	 * 
	 * @param 		scanner										Scanner Object
	 * 
	 * @param 		node										Node Object
	 * 
	 * @param 		hyp											Hypothesis Object
	 * 
	 * @param 		args3										Input line
	 * 
	 * @throws 		IOException
	 * 
	 * @throws 		ClassNotFoundException
	 */
	
	public void deserializeModel(String args, String args2, Scanner scanner, Node node, Hypothesis hyp, String args3) throws IOException, ClassNotFoundException {
		FileInputStream fileInput = new FileInputStream(args);
		ObjectInputStream input = new ObjectInputStream(fileInput);
		Object object = (Object) input.readObject();
		if(object instanceof Node) {
			node = (Node) object;
			input.close();
			fileInput.close();
			scanner = new Scanner(new FileInputStream(args2));
			predictModel(scanner, node, args3);
		}
		if(object instanceof Hypothesis) {
			hyp = (Hypothesis) object;
			input.close();
			fileInput.close();			
			scanner = new Scanner(new FileInputStream(args2));
			predictHypothesis(scanner, hyp, args3);
		}

	}

	/**
	 * Predict the model
	 * 
	 * @param 			scanner						Scanner Object
	 * 
	 * @param 			node						Node Object
	 * 
	 * @param 			args						Input line
	 */
	
	public void predictModel(Scanner scanner, Node node, String args) {
		while(scanner.hasNextLine()) {
			String line = ""; 
			line = scanner.nextLine();
			List<String> features = getFeatures(line, args);
			getLanguage(node, features);
		}
	}

	/**
	 * Display the prediction
	 * 
	 * @param 			node						Node Object
	 * 
	 * @param 			features					Get Features
	 */
	
	public void getLanguage(Node node, List<String> features) {

		if(node.getBaseMap() != null ) {
			if(!node.getBaseMap().isEmpty()) {
				if(node.getBaseMap().get(1) > node.getBaseMap().get(2)) {
					System.out.println("nl");
				}
				else {
					System.out.println("en");
				}
			}
		}
		else { 
			if(features.get(node.getNode()).equals("True")) {
				getLanguage(node.getLeft(), features);
			}else {
				getLanguage(node.getRight(), features);
			}
		}
	}

	/**
	 * Predict the Model
	 * 
	 * @param 				scanner					Scanner Object
	 * 
	 * @param 				hyp						Hypothesis Object
	 * 
	 * @param 				args					Input Line
	 */
	
	public void predictHypothesis(Scanner scanner, Hypothesis hyp, String args) {
		while(scanner.hasNextLine()) {
			String line = ""; 
			line = scanner.nextLine();
			List<String> features = getFeatures(line, args);
			getHypLanguage(hyp, features);
		}
	}

	/**
	 * Display the prediction
	 * 
	 * @param 			hyp						Hypothesis Object
	 * 
	 * @param 			features				Get Features
	 */
	
	public void getHypLanguage(Hypothesis hyp, List<String> features) {

		double total = 0.0;
		double trueValue = 1.0;
		double falseValue = -1.0;
		for(int k = 0; k < 5; k++) {
			double aplha = hyp.getAlpha();
			if(features.get(k).equals("True")) {
				total += (trueValue * aplha);
			}else {
				total += (falseValue * aplha);
			}			
			hyp = hyp.next;
		}
		if(total > 0) {
			System.out.println("nl");
		}
		else {
			System.out.println("en");
		}
	}

	/**
	 * Serialize the adaboost Object
	 * 
	 * @param 			hyp					Hypothesis Object
	 * 
	 * @param 			args				Serialize File
	 * 
	 * @throws 			IOException
	 */
	
	public void serializeAdaboost(Hypothesis hyp, String args) throws IOException {
		FileOutputStream outputFile = new FileOutputStream(args);
		ObjectOutputStream out = new ObjectOutputStream(outputFile);
		out.writeObject(hyp);
		out.close();
		outputFile.close();		
	}

	public static void main(String[] args) throws IOException, ClassNotFoundException {
		List<List<String>> xyValues = new ArrayList<List<String>>();
		List<Integer> trueList = new ArrayList<Integer>();
		List<Integer> falseList = new ArrayList<Integer>();

		Classify dTree = new Classify();
		Scanner scanner = null;
		if(args[0].equals("train")) {

			if(args[3].equals("dt")) {
				scanner = new Scanner(new FileInputStream(args[1]));
				xyValues = dTree.readFile(scanner, xyValues, args[0]);

				double min = 100;

				Node node = null;

				node = dTree.recursion(xyValues, depth, trueList, falseList, min, node);

				dTree.serializeModel(node, args[2]);
			}
			if(args[3].equals("ada")) {

				scanner = new Scanner(new FileInputStream(args[1]));
				xyValues = dTree.readFile(scanner, xyValues, args[0]);

				double min = 100;

				Hypothesis hyp = dTree.getAdaboost(xyValues, trueList, falseList, min);
				dTree.serializeAdaboost(hyp, args[2]);
			}


		}
		if(args[0].equals("predict")) {
			Node node = null;
			Hypothesis hyp = null;
			dTree.deserializeModel(args[1], args[2], scanner, node, hyp, args[0]);
		}

	}


}

class Node implements Serializable{

	int node;

	Node left;

	Node right;

	Map<Integer, Integer> countMap;

	Map<Integer, Integer> baseMap;

	public Node(int node, Node left, Node right, Map<Integer, Integer> countMap) {
		this.node = node;
		this.left = left;
		this.right = right;
		this.countMap = countMap;
	}

	public Node(int node) {
		this.node = node;
	}

	public Node(int node, Map<Integer, Integer> countMap) {
		this.node = node;
		this.countMap = countMap;
	}

	public Node(Map<Integer, Integer> baseMap) {
		this.baseMap = baseMap;
	}

	public int getNode() {
		return node;
	}

	public void setNode(int node) {
		this.node = node;
	}

	public Node getLeft() {
		return left;
	}

	public void setLeft(Node left) {
		this.left = left;
	}

	public Node getRight() {
		return right;
	}

	public void setRight(Node right) {
		this.right = right;
	}

	public Map<Integer, Integer> getCountMap() {
		return countMap;
	}

	public void setCountMap(Map<Integer, Integer> countMap) {
		this.countMap = countMap;
	}

	public Map<Integer, Integer> getBaseMap() {
		return baseMap;
	}

	public void setBaseMap(Map<Integer, Integer> baseMap) {
		this.baseMap = baseMap;
	}


}

class Hypothesis implements Serializable{

	double alpha;

	Hypothesis next;

	public Hypothesis(Double alpha, Hypothesis next) {
		this.alpha = alpha;
		this.next = next;
	}

	public Hypothesis(double alpha) {
		this.alpha = alpha;
	}

	public double getAlpha() {
		return alpha;
	}

	public void setAlpha(double alpha) {
		this.alpha = alpha;
	}

	public Hypothesis getNext() {
		return next;
	}

	public void setNext(Hypothesis next) {
		this.next = next;
	}
}
