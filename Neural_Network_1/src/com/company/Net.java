package com.company;

import java.util.Random;

/*
ChickenBroccoli 27-October-2018
Deep-Learning Neural Network
Uses simple gradient descent method to optimize net.
Loss function is the sum over all trials and all neurons, the square of the difference between expected and actual value.
Input and output vector sizes user-configurable
Number of hidden layers user-configurable
Hidden layers the same length
setInput sets the a double[] as the first layer neurons in the net
Method run calculates the value for each neuron for the entire net.
Method train makes instance of Net change itself
Method getOutput returns a double[] of the activations of the last layer of neurons
*/


public class Net {

    private Neuron[][] neurons;
    private Connection[][][] connections;
    private double[][] inputList;
    private double[][] expectedOutputList;
    private double learningRate = 0.001;

    public Net(int inputSize, int outputSize, int layers, int complexity){
        this.neurons = new Neuron[layers][];
        this.neurons[0] = new Neuron[inputSize];    //create inputLayer
        this.neurons[layers -1] = new Neuron[outputSize];   //create outputLayer
        for(int layer = 1; layer < layers-1; layer++){      //creates middle layers
            this.neurons[layer] = new Neuron[complexity];
        }
        this.connections = new Connection[layers-1][][];
        this.connections[0] = new Connection[inputSize][complexity];    //create connections from inputLayer
        this.connections[layers -2] = new Connection[complexity][outputSize];   //create connections to outputLayer
        for(int layer = 1; layer < layers-2; layer++){      //creates connections between middle layers
            this.connections[layer] = new Connection[complexity][complexity];
        }
        this.resetAll();
    }

    public void resetAll() { //resets all connections to (random) and neuron activations to random;
        Random random = new Random();
        for (int layer = 0; layer < neurons.length; layer++) {
            for (int index = 0; index < neurons[layer].length; index++) {
                neurons[layer][index] = new Neuron(random.nextDouble());
            }
        }
        for (int layer = 0; layer < connections.length; layer++) {
            for (int startIndex = 0; startIndex < connections[layer].length; startIndex++) {
                for (int endIndex = 0; endIndex < connections[layer][startIndex].length; endIndex++) {
                    connections[layer][startIndex][endIndex] = new Connection(random.nextDouble()*10-5, random.nextDouble()*10-5);
                }
            }
        }
    }

    public void setExpectedOutput(double[][] expectedOutputList){    //takes expectedOutputList[trial_number][index], gives sum of all
        this.expectedOutputList = expectedOutputList;
    }

    public void setInputList(double[][] inputList) {
        this.inputList = inputList;
    }

    public void train(int iterations){
        for(int i = 0; i<iterations; i++){
            if(i%100 == 0){
                System.out.println(i + "  Error is: " + this.computeError());
            }
            this.runAllTrials();
            this.learnStep();

            //this.printConnectionStates();
        }
    }

    public void setInput(double[] values){
        if(values.length != neurons[0].length){     //makes sure input matches net length
            System.out.println("Input length does not match net.");
        }
        else{
            for(int index = 0; index < values.length; index++){
                neurons[0][index] = new Neuron(values[index]);      //creates new Neurons of set activation values
            }
        }
    }

    public double[] getOutput(){
        double[] output = new double[neurons[neurons.length -1].length];
        for(int index = 0; index < output.length; index++ ){
            output[index] = neurons[neurons.length-1][index].getActivation();
        }
        return output;
    }

    public void run(){
        for (int layer = 1; layer < neurons.length; layer++) {    //for every layer except the first one
            for (int endIndex = 0; endIndex < neurons[layer].length; endIndex++) {   //for every neuron of the layer
                double sum = 0;
                for (int startIndex = 0; startIndex < connections[layer - 1].length; startIndex++) {  //for every connection to the neuron[layer][endIndex], ie connections[layer-1][][endIndex]
                    double weight = connections[layer - 1][startIndex][endIndex].getWeight();
                    double bias = connections[layer - 1][startIndex][endIndex].getBias();
                    double previousActivation = neurons[layer - 1][startIndex].getActivation();
                    sum += previousActivation * weight + bias;        //a(n) = sum(a(n-1)*w+b)
                }
                neurons[layer][endIndex] = new Neuron(sigmoid(sum));
            }
        }
        //this.printNeuronStates();
    }


    //PRIVATE METHODS
    private void runAllTrials(){
        for(int trialIndex = 0; trialIndex< inputList.length; trialIndex++) {
            setInput(inputList[trialIndex]);
            for (int layer = 1; layer < neurons.length; layer++) {    //for every layer except the first one
                for (int endIndex = 0; endIndex < neurons[layer].length; endIndex++) {   //for every neuron of the layer
                    double sum = 0;
                    for (int startIndex = 0; startIndex < connections[layer - 1].length; startIndex++) {  //for every connection to the neuron[layer][endIndex], ie connections[layer-1][][endIndex]
                        double weight = connections[layer - 1][startIndex][endIndex].getWeight();
                        double bias = connections[layer - 1][startIndex][endIndex].getBias();
                        double previousActivation = neurons[layer - 1][startIndex].getActivation();
                        sum += previousActivation * weight + bias;        //a(n) = sum(a(n-1)*w+b)
                    }
                    neurons[layer][endIndex] = new Neuron(sigmoid(sum));
                }
            }
        }
    }

    private double computeError(){     //takes expectedOutputList[trial_number][index], gives sum of all
        double temp = 0;
        for(int trial_Number = 0; trial_Number < expectedOutputList.length; trial_Number++){
            for(int index = 0; index< expectedOutputList[trial_Number].length; index++){
                double difference = expectedOutputList[trial_Number][index] - this.getOutput()[index];
                double error = difference*difference;
                temp += error;
            }
        }
        return temp;
    }

    private void learnStep(){   //modifications must be made after all trials run
        Connection[][][] temp = this.connections;
        for(int trialIndex = 0; trialIndex< inputList.length; trialIndex++) {
            for (int layerIndex = 0; layerIndex < connections.length; layerIndex++) {
                for (int startIndex = 0; startIndex < connections[layerIndex].length; startIndex++) {
                    for (int endIndex = 0; endIndex < connections[layerIndex][startIndex].length; endIndex++) {
                        double weightDerivative = 0.0;
                        double biasDerivative = 0.0;
                        weightDerivative += learningRate* derivativeWeight(layerIndex, startIndex, endIndex, trialIndex);
                        biasDerivative += learningRate * derivativeBias(layerIndex, startIndex, endIndex, trialIndex);
                        double currentWeight = temp[layerIndex][startIndex][endIndex].getWeight();
                        double currentBias = temp[layerIndex][startIndex][endIndex].getBias();
                        temp[layerIndex][startIndex][endIndex] = new Connection(currentWeight - weightDerivative, currentBias - biasDerivative);

                    }
                }
            }
            this.run();
        }
        this.connections = temp;
    }

    //MATH
    private static double sigmoid(double x){  //sigmoid function
        double y = 1/(1+Math.pow(2.718281828459045235360, -x));
        return y;
    }

    private double derivativeWeight(int layerIndex, int startIndex, int endIndex, int trialIndex){
        if(layerIndex == neurons.length-2){
            double x = expectedOutputList[trialIndex][endIndex];
            return 2*(neurons[layerIndex+1][endIndex].getActivation()-x)*(neurons[layerIndex+1][endIndex].getActivation())*(1-neurons[layerIndex+1][endIndex].getActivation())*(neurons[layerIndex][startIndex].getActivation());
        }
        else{
            return derivativeActivation(layerIndex+1, endIndex, trialIndex)*(neurons[layerIndex+1][endIndex].getActivation())*(1-neurons[layerIndex+1][endIndex].getActivation())*neurons[layerIndex][startIndex].getActivation();
        }
    }

    private double derivativeBias(int layerIndex, int startIndex, int endIndex, int trialIndex){
        if(layerIndex == neurons.length-2){
            double x = expectedOutputList[trialIndex][endIndex];
            return 2*(neurons[layerIndex+1][endIndex].getActivation()-x)*(neurons[layerIndex+1][endIndex].getActivation())*(1-neurons[layerIndex+1][endIndex].getActivation());
        }
        else{
            return derivativeActivation(layerIndex+1, endIndex, trialIndex)*(neurons[layerIndex+1][endIndex].getActivation())*(1-neurons[layerIndex+1][endIndex].getActivation());
        }
    }

    private double derivativeActivation(int layerIndex, int index, int trialIndex){
        if(layerIndex == neurons.length-2){
            double sum = 0;
            for(int j =0; j<neurons[layerIndex+1].length; j++){
                double x = expectedOutputList[trialIndex][j];
                sum+= 2*(neurons[layerIndex+1][j].getActivation()-x)*(neurons[layerIndex+1][j].getActivation())*(1-neurons[layerIndex+1][j].getActivation())*(connections[layerIndex][index][j].getWeight());
            }
            return sum;
        }
        else{
            double sum = 0;
            for(int j =0; j<neurons[layerIndex+1].length; j++){
                sum+= derivativeActivation(layerIndex+1, j, trialIndex )*(neurons[layerIndex+1][j].getActivation())*(1-neurons[layerIndex+1][j].getActivation())*(connections[layerIndex][index][j].getWeight());
            }
            return sum;
        }
    }

    public void printNeuronStates(){
        for(Neuron[] layer: neurons ){
            for(Neuron neuron: layer){
                System.out.print(neuron.getActivation() + "  ");
            }
            System.out.println();
        }
        System.out.println();
        System.out.println();
    }

    public void printConnectionStates(){
        for(Connection[][] layer: connections){
            for(Connection[] startIndex: layer){
                for(Connection endIndex: startIndex){
                    System.out.print(endIndex.getWeight()+ ", "+ endIndex.getBias()+ "  ");
                }
                System.out.println();
            }
            System.out.println();
        }
        System.out.println();
        System.out.println();
    }
}
