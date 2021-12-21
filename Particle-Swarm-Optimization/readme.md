PSO: Particle Swarm Optimization is inspired by the social foraging behavior of some animals, such as the flocking behavior of birds. 
The goal of the algorithm is to have all the particles locate the optima in a multi-dimensional hyper-volume.
First, we assign initially random positions to all particles in the space as a population and small initial random velocities. 
The algorithm is executed like a simulation, advancing the location of each particle in turn based on its velocity, the best known global position in the problem space, 
and the best position known to a particle. After each position update, calculate the best global position and the best position of each population. 
Over time, through a combination of exploration and exploitation of known good positions in the search space, 
the particles cluster or converge together around optima or several optima.
