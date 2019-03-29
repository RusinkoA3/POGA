# POGA
Price Optimization via Genetic Algorithm

Generate an optimal set of prices for a product based on discounts for volume purchases. 
Prices should decrease with volume but increase with extended price (volume * price). Prices
must be below a "soft" ceiling (violations are permitted) but MUST be above the floor (no 
violations allowed).

This is an example of code re-use, i.e. I started by using code from another's Github page.

GeneticAlgorithm.py should be loaded into a seperate module for use.
