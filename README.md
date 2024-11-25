# Templates for Evolutionary Algorithms
Templates for projects involving evolutionary algorithms in python.

## Desiderata
1. Good logging (using the `logging` module)
2. Automatic creation of a folder with the current date
3. Saving the population state at every generation (or so)
4. Being able to seed the initial population and restart from a previous state
5. Parallelizing fitness evaluations, probably using multi-processing (multi-thread still suffers from GIL)
6. Seeding of all random number generators
