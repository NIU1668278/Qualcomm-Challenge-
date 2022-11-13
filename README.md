# Qualcomm-Challenge
Submission to the Qualcomm Challenge in the Datathon FME

Group: Pol Resina, Gerard Grau, Guim Casadellà and Marc Herrero

Problem statement:

To create a chip, we need to deliver the power everywhere. In order to do so, a lot of power switches are placed all over the design. They all have to be connected and routed together in a chain. The faster you turn these power switches on and off, the less power you use.

Here, the challenge begins! We want to create independent groups of power switches so that we reduce the turn off/on time.

A power switch is a type of cell that is inserted inside the design. Each cell has one input pin (conn_out) and one output pin (conn_in). Two power switches are connected through a net from the output pin of the previous switch to the input pin of the next switch.

First Approach:
![image](https://user-images.githubusercontent.com/116063104/201517796-7fd519fd-ea16-4d6e-a3d3-802a30b8f0ff.png)
The travelling salesman problem (also called the travelling salesperson problem or TSP) asks the following question: "Given a list of cities and the distances between each pair of cities, what is the shortest possible route that visits each city exactly once and returns to the origin city?" 

Problem:

We can have several salesmen and huge amount of cities make the computation very slow, because we know it is a NP-hard problem

Solution (Clustering):

Since the complexity is at least O(n^2) we want to reduce the amount of cities per salesman

In order to do that we decided to split the pins in such a way that area is the main factor to take into acount:
![image](https://user-images.githubusercontent.com/116063104/201518362-400bfa2a-7efa-45de-adfb-e59824cffc48.png)
![image](https://user-images.githubusercontent.com/116063104/201518383-e5898d5c-f372-4be0-b290-80b7b4d80265.png)

Solution(TSP):
Now that we have less pins we can apply a TSP algorithm. 

Genetic Algorithm:
The genetic algorithm is a method for solving both constrained and unconstrained optimization problems that is based on natural selection, the process that drives biological evolution.
Problem:
Suboptimal solution, very time consuming

Solution(Simulated annealing):
Simulated annealing (SA) is a probabilistic technique for approximating the global optimum of a given function. Specifically, it is a metaheuristic to approximate global optimization in a large search space for an optimization problem.

This notion of slow cooling implemented in the simulated annealing algorithm is interpreted as a slow decrease in the probability of accepting worse solutions as the solution space is explored. Accepting worse solutions allows for a more extensive search for the global optimal solution.

![image](https://user-images.githubusercontent.com/116063104/201519231-7c3779c8-316d-4cfa-bd4d-38798fed78d8.png)


Further Impovements(Multiprocessing):
As we have splited our problem in diferent clusters the process can be paralelized with no problem.



