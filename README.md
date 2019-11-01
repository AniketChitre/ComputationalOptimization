# Design of Computational Optimization Algorithms

### About Me and Project Scope

Final-year MEng Chemical Engineering undergraduate at Imperial College London interested in optimization and machine learning. Working under the supervision of Dr Antonio Del Rio Chanona on this side-project to introduce myself to Python programming and further explore optimization, especially computational techniques. 

Objective is to have weekly project goals exploring new ideas in the field. 

## Week 1 

Goal to code two local search algorithms in 2d: 
- Simulated annealing 
- Gradient descent 

And implement them on two test-functions: 
- Simple sum-of-squares function (convex)
- Rosenbrock function (non-convex) 

## Week 2 

The week 2 code file builds upon the computational optimization techniques explored in week 1 to: 
- Vectorise the gradient descent and simulated annealing algorithms to higher dimensions (i.e. extending them from taking an input of a 2d vector to being able to process a n-dimension vector/function) 
- Improve the gradient descent algorithm through introduction of "backtracking" - this involves adjusting the "alpha" (learning rate) in the algorithm such that the result from the next iteration is always better than the previous one 
- Visualise some analysis of the algorithms. For example, the figure on the left shows how the output of the simulated annealing algorithm behaves as a function of the number of iterations for a minimisation problem. And the figure to its right shows how a 2d search space is explored by a gradient descent algorithm.


![](Images/Simulated_annealing.png) ![](Images/gradient_descent_search_space.png)

## Week 3

Goal is to implement the Adam optimization algorithm - the key features of this algorithm is: 
- It is a combination of gradient descent with momentum and RMSprop algorithms 


