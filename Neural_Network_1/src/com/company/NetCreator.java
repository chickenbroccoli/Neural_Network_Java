package com.company;

public class NetCreator {
    public static Net netCreator(double[][] inputList, double[][] outputList){
        Net net = new Net(inputList[0].length, outputList[0].length, 3, 3);
        net.setInputList(inputList);
        net.setExpectedOutput(outputList);
        net.train(1000000);
        return net;
    }
}
