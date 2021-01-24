# physis

Physis (in Greek φύσις) is the first and fundamental reality, the principle and the cause of all things, according to the pre-Socratic philosophers. The term is typically translated as "nature".

The idea behind this package is to provide various modules and tools dealing with natural phenomena being the main subject of physical and chemical sciences.

## Modules

At the moment the package contains a few Python modules, which I coded for the statistics courses I attended during the master's degree course in computational physics at the University of Modena.

I plan to refactor these modules and write the most computationally demanding sections in C/C++. I wil also implement new physical modules, work and time permitting.

### Statistical Physics

Generic statistical methods to be used in solid state physics, e.g. phase transitions. It includes:

- Ising Model
- Gibbs ensemble Monte Carlo for Lennard-Jones gas-liquid transition

### Liquid Crystals

Modules that simulates liquid crystals. It includes:

- Isotropic-Nematic transition for a Liquid Crystal using Lebwohl-Lasher model
