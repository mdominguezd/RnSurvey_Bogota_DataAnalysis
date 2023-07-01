# Rn Data analysis and interactive visualization

In this repository you can find the RAW Rn data collected in Bogotá, Colombia and the codes, written in `Python` pogramming language (.ipynb), that were used to analyse the RC data retrieved in the context of the publication *Indoor 222Rn Modeling in Data-Scarce Regions: An Interactive Dashboard Approach for Bogotá, Colombia*. Additionally, a dashboard was created to make the interaction with the data more user friendly and to facilitate the replicability of this type of studies in other study areas. Further information about the dashboard source code and functionality can be found [here](https://github.com/mdominguezd/IRC_modeling_dashboard).
<br><br>
The repository is divided three jupyter notebooks and four data folders.
- Folders:
    - Dataset for fitting
        
        Folder with the Raw data (`Raw_Results_LR115.xlsx`) used in Data distribution.ipynb and the dataset with dependent and independent variables used for fitting the regression models (`Processed_DataFrame.csv`).
        
    - Dataset for regression
        
        Folder with the cadaster data to which the regression will be applied. This dataset must have the same independent variables than the dataset used for fitting the model. **In the repository this data is zipped for storage purposes**. When the `Multivariate analysis.ipynb` is ran the dataset is unzipped.
        
    - Figures
        
        Folder with all of the figures created in the Data distribution.ipynb and Multivariate analysis.ipynb.
        
    - Regression results
        
        This folder contains the results of regressions created in the Multivariate analysis notebook.
        
- Notebooks
    - Data distribution.ipynb
       
       Jupyter notebook with basic statistical analysis of the raw RC data (Raw_Results_LR115.xlsx).
       
    - Multivariate anlysis.ipynb
        
        Jupyter notebook with: 
        
        - Multivariate analysis of the processed dataset (Processed_DataFrame.csv) [correlation matrix, PCA, etc.] 
        - Fitting of RC data using predictors.
        - Perform feature selection
        - Estimate RC in the Dataset for regression (Cadastre information)
        
    - Dashboard App
        
        <!-- Jupyter notebook that creates a dashboard that can be used to create an interactive app for Residential RC modeling. This app will take the datasets found in **Dataset for fitting** (`Processed_DataFrame.csv`) and **Dataset for regression** (`Houses_for_Rn_estimation_Cataster.txt`). When used for a different study area, this files should be updated with the corresponding data. -->

      An **improved and updated version** of this **dashboard** can be accessed online [**here**](http://ircmodelingdashboard.eu.pythonanywhere.com/). Nevertheless, the datasets presented in this repository can be used as an example in the dashboard.
    

## Publication abstract
Radon ($^{222}$Rn) is a naturally occurring gas that represents a health threat due to its causal relationship with lung cancer. Despite its potential health impacts, several regions have not conducted studies, mainly due to data scarcity and/or economic constraints. This study aims to bridge the baseline information gap by building an interactive [dashboard](http://ircmodelingdashboard.eu.pythonanywhere.com/) that uses inferential statistical methods to estimate indoor radon concentration’s (IRC) spatial distribution for a target area. We demonstrate the functionality of the dashboard by modeling IRC in the city of Bogotá, Colombia, using 30 in situ measurements. The IRC were measured for 35 days using Alpha-track detectors (LR-115). IRC measured were the highest reported in the country, with a geometric mean of 91 ±14 Bq/m$^3$ and a maximum concentration of 407 Bq/m$^3$. In 56.66\% of the residences RC exceeded the WHO's recommendation of 100 Bq/m$^3$.  A prediction map for houses registered in Bogotá’s cadaster was built in the dashboard by using a log-linear regression model fitted with the in situ measurements, together with meteorological, geologic and building specific variables. After feature selection, the log-linear model showed a cross-validation Root Mean Squared Error (RMSE) of 56.5 $\frac{Bq}{m^3}$. Furthermore, the model showed that the age of the house presented a statistically significant positive association with RC. According to the model, IRC measured in houses built before 1980 present a statistically significant increase of 71.60\% compared to those built after 1980 (p-value = 0.045). The prediction map showed higher IRC in older buildings most likely related to cracks in the structure that could enhance gas migration in older houses. This study highlights the importance of expanding $^{222}$Rn studies in countries with a lack of baseline values and provides a cost-effective alternative that could help deal with the scarcity of IRC data and get a better understanding of place-specific IRC spatial distribution.

## Data distrbution code (Data distribution.ipynb)

This Jupyter notebook reads the RAW data (`Raw_Results_LR115.xlsx`) and create graphs for easier visualization and comparison with recommended levels and previous measurements in Latin America and the Caribbean (LAC) region.

## Multivariate analysis code (Multivariate analysis.ipynb)

This Jupyter notebbok uses the information of the RC data (Dependent variable) and the independent variables (`Processed_DataFrame.csv`) to fit one log-linear regression model. 
<br>
Subsequently, this notebook uses the data of Bogotá's cadastre to apply the regression model on all the houses with information of the independent variables (Information taken from Bogotá's cadaster). The data is rasterize using `GDAL` tools. 
<br>
The outputs of this model are:
- Figures:
    - Variable caracterization figure (`Figures/Caracterization.png`)
    - Principal component biplot figure (`Figures/PCA_RC.png`)
    - Percent change calculated for all independent variables (`Figures/Regresión_LogLineal.png`)
    - Percent change calculated for independent variables after feature selection (`Figures/Regresión_LogLineal_withFeatureSel.png`)
    - Residential RC estimated distribution (`Figures/Estimated_Rn_Histogram.png`)
- Files (To Regression results):
    - RC estimated for each house in cadaster information `LinReg_model_results.csv`.
    - Raster with RC regression results (`Log_Linear_estimations.tif`)
    
## Dashboard app 

Refer to the github repository [here](https://github.com/mdominguezd/IRC_modeling_dashboard) to see the source code and the online running version of the dashboard [here](http://ircmodelingdashboard.eu.pythonanywhere.com/) to make use of it.

<!-- This Jupyter notebook creates an interactive app for residential RC modeling:

<center>
    <img src='MarkDown_Assets/RC_modeling_app.jpg' width = '800'>
    
    Initial display of Dashboard app.
</center>

### "Under the hood" data input
The app uses the files in `Dataset for fitting` and `Dataset for regression` folders. These files can be changed for different study areas to perform the same analysis in different study areas or with different RC measurements.

### User inputs
While using the app, the user is able to decide:

<center>
    <img src='MarkDown_Assets/RC_modeling_inputs.jpg' width = '500'>
    
    User inputs in Dashboard app.
</center>

- The reference level to compare its dataset
    - World Health Organization (WHO)
    - US Environmental Protection Agency (EPA)
- The type of visualization to help him decide which features will be selected for the regression
    - Correlation matrix (Low correlation values with RC and highly correlated pairs of independent variables are not suggested)
    - Variance inflation factor (VIFs above 4 are not suggested)
- The type of regression model to be used for fitting and estimating RC
    - Log-linear regression
    - Random Forest regression
- Features to be used in the fitting and estimation of RC
    - The variables here are the same ones in `Processed_DataFrame.csv`.
- Option to perform a high quallity model
    - High quality resolution = 100m ($\approx$ 30 minutes computation time)
    - Low quality ressolution = 300m ($\approx$ 5 minutes computation time)
    
### Data analysis
Two windows display information for the analysis of the data. The first one shows a histogram and a pie plot of the RC measured (`Processed_DataFrame.csv`). The second one shows 

### Modeling results
When models are run with the app. They are displayed in the map and the tables on the left.
<center>
    <img src='MarkDown_Assets/RC_modeling_results.jpg' width = '800'>
    
    Regression results presented in Dashboard.
</center>

#### Advanced modeling settings
If some advanced modelling settings need to be changed, this can be done in the `EstimatingValues` and `rasterize` functions that are defined. Here the user can change:
- Number of cells for cadaster aggregation in the horizontal component (`width_of_cells_for_aggregation`)
- Resolution of cells of raster created (`res`)
- Coordinate reference systerm (`crs`)

#### Reset modeling environment
For optimizing the app performance the estimated values of RC are saved in the `Regression results` folder.


# Important notes
To run all of the codes and deploy the dashboard app all libaries used need to be installed in your modelling envionment. For some Geospatial analysis the `GDAL` library is used. Follow the instructions in [here](https://opensourceoptions.com/blog/how-to-install-gdal-for-python-with-pip-on-windows/) -->

