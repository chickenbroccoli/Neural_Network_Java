package com.company;

public class Connection {

    private double weight;
    private double bias;

    public Connection(){
        this.weight = 1.0;
        this.bias = 0;
    }
    public Connection(double weight, double bias){
        this();
        this.setValues(weight, bias);
    }



    public void setValues(double weight, double bias){
        this.weight = weight;
        this.bias = bias;
    }

    public double getWeight(){
        return this.weight;
    }

    public double getBias(){
        return this.bias;
    }

    public String toString(){
        return this.weight + ", " + this.bias;
    }
}
