# House Price Prediction with Linear Regression

House Price Prediction using Linear Regression is a fundamental project in the field of machine learning and data science. It involves building a model to predict the price of a house based on various features such as square footage, number of bedrooms, and number of bathrooms. This type of model is particularly useful for real estate agents, buyers, and sellers to estimate property values accurately.

**Data Description:-**

| Feature       | Description                                                                                                                                               |
|---------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| MSSubClass    | Identifies the type of dwelling involved in the sale.                                                                                                       |
| MSZoning      | Identifies the general zoning classification of the sale.                                                                                                  |
| LotFrontage   | Linear feet of street connected to property.                                                                                                                |
| LotArea       | Lot size in square feet.                                                                                                                                   |
| Street        | Type of road access to property (Gravel or Paved).                                                                                                          |
| Alley         | Type of alley access to property (Gravel, Paved, or No alley access).                                                                                        |
| LotShape      | General shape of property (Regular, Slightly irregular, Moderately Irregular, Irregular).                                                                   |
| LandContour   | Flatness of the property (Near Flat/Level, Banked, Hillside, Depression).                                                                                     |
| Utilities     | Type of utilities available (All public, Electricity, Gas, and Water, Electricity and Gas Only, Electricity only).                                            |
| LotConfig     | Lot configuration (Inside, Corner, Cul-de-sac, Frontage on 2 sides, Frontage on 3 sides).                                                                    |
| LandSlope     | Slope of property (Gentle slope, Moderate Slope, Severe Slope).                                                                                              |
| Neighborhood  | Physical locations within Ames city limits.                                                                                                                |
| Condition1    | Proximity to various conditions (Artery, Feedr, Norm, RRNn, RRAn, PosN, PosA, RRNe, RRAe).                                                                  |
| Condition2    | Proximity to various conditions (if more than one is present).                                                                                              |
| BldgType      | Type of dwelling (Single-family Detached, Two-family Conversion, Duplex, Townhouse End Unit, Townhouse Inside Unit).                                         |
| HouseStyle    | Style of dwelling (One story, One and one-half story, Two story, Split Foyer, Split Level, etc.).                                                           |
| OverallQual   | Rates the overall material and finish of the house (Very Excellent to Very Poor).                                                                           |
| OverallCond   | Rates the overall condition of the house (Very Excellent to Very Poor).                                                                                      |
| YearBuilt     | Original construction date.                                                                                                                                |
| YearRemodAdd  | Remodel date (same as construction date if no remodeling or additions).                                                                                     |
| RoofStyle     | Type of roof (Flat, Gable, Gambrel, Hip, Mansard, Shed).                                                                                                    |
| RoofMatl      | Roof material (Clay or Tile, Standard Shingle, Membrane, Metal, Roll, Gravel & Tar, Wood Shakes, Wood Shingles).                                             |
| Exterior1st   | Exterior covering on house (Asbestos Shingles, Brick, Cement Board, Metal Siding, etc.).                                                                    |
| Exterior2nd   | Exterior covering on house (if more than one material).                                                                                                     |
| MasVnrType    | Masonry veneer type (Brick Common, Brick Face, Cinder Block, None, Stone).                                                                                  |
| MasVnrArea    | Masonry veneer area in square feet.                                                                                                                        |
| ExterQual     | Quality of material on the exterior (Excellent, Good, Average/Typical, Fair, Poor).                                                                         |
| ExterCond     | Present condition of material on the exterior (Excellent, Good, Average/Typical, Fair, Poor).                                                               |
| Foundation    | Type of foundation (Brick & Tile, Cinder Block, Poured Concrete, Slab, Stone, Wood).                                                                        |
| BsmtQual      | Height of the basement (Excellent, Good, Typical, Fair, Poor, No Basement).                                                                                 |
| BsmtCond      | General condition of the basement (Excellent, Good, Typical, Fair, Poor, No Basement).                                                                      |
| BsmtExposure  | Walkout or garden level walls (Good, Average, Minimum, No Exposure, No Basement).                                                                          |
| BsmtFinType1  | Rating of basement finished area (Good Living Quarters, Average Living Quarters, Below Average Living Quarters, etc.).                                      |
| BsmtFinSF1    | Type 1 finished square feet.                                                                                                                                 |
| BsmtFinType2  | Rating of basement finished area (if multiple types).                                                                                                      |
| BsmtFinSF2    | Type 2 finished square feet.                                                                                                                                 |
| BsmtUnfSF     | Unfinished square feet of basement area.                                                                                                                    |
| TotalBsmtSF   | Total square feet of basement area.                                                                                                                         |
| Heating       | Type of heating (Floor Furnace, Gas Forced Warm Air Furnace, Gas Hot Water or Steam Heat, etc.).                                                            |
| HeatingQC     | Heating quality and condition (Excellent, Good, Average/Typical, Fair, Poor).                                                                              |
| CentralAir    | Central air conditioning (Yes or No).                                                                                                                      |
| Electrical    | Electrical system (Standard Circuit Breakers & Romex, Fuse Box over 60 AMP, etc.).                                                                         |
| 1stFlrSF      | First floor square feet.                                                                                                                                   |
| 2ndFlrSF      | Second floor square feet.                                                                                                                                  |
| LowQualFinSF  | Low-quality finished square feet (all floors).                                                                                                             |
| GrLivArea     | Above grade living area square feet.                                                                                                                       |
| BsmtFullBath  | Basement full bathrooms.                                                                                                                                   |
| BsmtHalfBath  | Basement half bathrooms.                                                                                                                                   |
| FullBath      | Full bathrooms above grade.                                                                                                                                |
| HalfBath      | Half baths above grade.                                                                                                                                    |
| Bedroom       | Bedrooms above grade (does NOT include basement bedrooms).                                                                                                  |
| Kitchen       | Kitchens above grade.                                                                                                                                     |
| KitchenQual   | Kitchen quality (Excellent, Good, Typical/Average, Fair, Poor).                                                                                            |
| TotRmsAbvGrd  | Total rooms above grade (does not include bathrooms).                                                                                                      |
| Functional    | Home functionality (Typical, Minor Deductions 1 and 2, Moderate Deductions, Major Deductions 1 and 2, Severely Damaged, Salvage only).                      |
| Fireplaces    | Number of fireplaces.                                                                                                                                     |
| FireplaceQu   | Fireplace quality (Excellent, Good, Average, Fair, Poor, No Fireplace).                                                                                    |
| GarageType    | Garage location (More than one type, Attached, Basement, Built-In, Car Port, Detached, No Garage).                                                         |
| GarageYrBlt   | Year the garage was built.                                                                                                                                 |
| GarageFinish  | Interior finish of the garage (Finished, Rough Finished, Unfinished, No Garage).                                                                            |
| GarageCars    | Size of garage in car capacity.                                                                                                                            |
| GarageArea    | Size of garage in square feet.                                                                                                                             |
| GarageQual    | Garage quality (Excellent, Good, Typical/Average, Fair, Poor, No Garage).                                                                                   |
| GarageCond    | Garage condition (Excellent, Good, Typical/Average, Fair, Poor, No Garage).                                                                                 |
| PavedDrive    | Paved driveway (Paved, Partial Pavement, Dirt/Gravel).                                                                                                     |
| WoodDeckSF    | Wood deck area in square feet.                                                                                                                             |
| OpenPorchSF   | Open porch area in square feet.                                                                                                                            |
| EnclosedPorch | Enclosed porch area in square feet.                                                                                                                        |
| 3SsnPorch     | Three-season porch area in square feet.                                                                                                                    |
| ScreenPorch   | Screen porch area in square feet.                                                                                                                          |
| PoolArea      | Pool area in square feet.                                                                                                                                  |
| PoolQC        | Pool quality (Excellent, Good, Average/Typical, Fair, No Pool).                                                                                             |
| Fence         | Fence quality (Good Privacy, Minimum Privacy, Good Wood, Minimum Wood/Wire, No Fence).                                                                     |
| MiscFeature   | Miscellaneous feature not covered in other categories (Elevator, 2nd Garage, Other, Shed, Tennis Court, None).                                             |
| MiscVal       | Value of miscellaneous feature in dollars.                                                                                                                |
| MoSold        | Month Sold (MM).                                                                                                                                         |
| YrSold        | Year Sold (YYYY).                                                                                                                                        |
| SaleType      | Type of sale (Warranty Deed - Conventional, Warranty Deed - Cash, Warranty Deed - VA Loan, etc.).                                                          |
| SaleCondition | Condition of sale (Normal Sale, Abnormal Sale, Adjoining Land Purchase, Allocation, Sale between family members, Home was not completed, etc.).             |

This table provides a detailed overview of each feature in the dataset, describing the type of information it represents and the possible values it can take.

## Project Overview

### Objective
To Implement a linear regression model to predict house prices based on their square footage, number of bedrooms, and number of bathrooms.

### Techniques Used
- Linear Regression

### Libraries
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn

### Key Features
- Data preprocessing
- Feature selection
- Model training
- Evaluation

### Dataset to downloaded from the below link
https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
