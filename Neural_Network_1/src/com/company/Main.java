package com.company;

import java.util.Random;

import static com.company.NetCreator.netCreator;

public class Main {

    public static void main(String[] args) {
	// write your code here
        double[][] input = new double[100000][2];
        double[][] output = new double[100000][1];
        Random j = new Random();
        for(int trialIndex = 0; trialIndex< 100000; trialIndex++) {
            double a = j.nextDouble();
            double b = 0.5;
            double f = a;
            input[trialIndex] = new double[]{a, b};     //creates training inputList
            output[trialIndex] = new double[] {f};
        }



        Net theNet = netCreator(input, output);


        double[] h = {0.2, 0.4};           //create test input
        theNet.setInput(h);                               //sets it

        theNet.run();


        double[] result = theNet.getOutput();
        System.out.println();
        for(int k = 0; k< result.length; k++){
            System.out.println(result[k]);            //gets and prints output
        }
    }
}
