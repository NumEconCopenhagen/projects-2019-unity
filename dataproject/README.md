# Dataproject

For our project we have chosen to analyse historical stock data from the yahoo-database 
We will base our analysis on a book by Lars Tvede aobut the 'golden' and 'dead' cross.

We got inspired by Lars Tvede’s book “Børshandlens psykologi / The psychology of the stock market”, to use python for analyzing stocks. In general, there is two basic analytical method to go around analyzing a stock – Fundamental and technical analysis. Fundamental analysis is what most people think of when analyzing a stock. Basically, you examine the key ratios of a business to determine its financial health. Technical is much different in the sense of only using historical market data to predict future market behavior. Most professional investors combine these to create an edge over the market. In this assignment we will only be focusing on the technical aspect. Since python excel in plotting and attracting data from the internet, it’s a good tool to analyzing historical data. In this assignment we are trying to achieve the basic things when you start out making a technical analysis. Which mean plotting and making it easily for future use to graph for other companies. Furthermore, adding a bit of technical aspects in form of pinpointing golden- and death cross. 

To produce our result one only needs to open the dataProject.ipynb file (located in projects-2019-unity/dataproject) which produces and shows the result, in code made specificly for Jupyter Notebook. 
This code relies heavily on functions defined in the funtions.py file (located in projects-2019-unity/dataproject/dataproject/). In this file we have defined funtion that download the data from yahoo and also that plots the data in an interactive way. The functions are very versatile a can take all the stocks and years located in the Yahoo database. 

Aditionally in making our graph we have relied heavily on the bokeh visualization library, so it will be necesary to 'pip install bokeh' or 'conda install bokeh' before trying to run the files. We have also used a module called mpl-finance so 'pip install mpl-finance' may also be necesary to run first.

Best regards
Unity group
