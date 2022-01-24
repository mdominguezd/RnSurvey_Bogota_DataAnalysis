# Rn Data analysis and visualization

In this repoitory you can find the RAW Rn data and the codes, written in `Python` pogramming language (.ipynb), that where used to analyse the data and visualize it.

## Data distrbution code (Data distribution.ipynb)

This Jupyter notebook reads the RAW data and create graphs for easier visualization and comparison with recommended levels and previous measurements in Latin America and the Caribbean (LAC) region.

## Multivariate analysis code (Multivariate analysis.ipynb)

This Jupyter notebbok uses the information of the RAW data (Dependent variable) and the independent variables to fit two regression models. 
</br>
Subsequently, this notebook uses the data of Bogotá's cadastre to apply the two regression models on all the houses with information of he independent variables. These results were mapped later using `ArcGIS`.

## Household survey code (Bogotá's household survey/ConstructionVariables_Bog.ipynb)

This notebook filters Bogotá´s cadastre data and export a .csv file with age and basement data of houses in Bogotá.