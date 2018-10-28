package com.company;

public class Neuron {

    private double activation;

    public Neuron(){
        setActivation(0);
    }

    public Neuron(double activation){
        setActivation(activation);
    }

    public void setActivation(double activation){
        this.activation = activation;
    }

    public double getActivation(){
        return this.activation;
    }
}
