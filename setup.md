---
title: Setup
---
# Software Packages Required

You will need to have an installation of Python 3 with the matplotlib, pandas, numpy and optionally opencv packages. 

The [Anaconda Distribution](https://www.anaconda.com/products/individual#Downloads) includes all of these except opencv by default.

## Installing OpenCV with Anaconda

* Load the Anaconda Navigator
* Click on "Environments" on the left hand side.
* Choose "Not Installed" from the pull down menu next to the channels button.
* Type "opencv" into the search box.
* Tick the box next to the opencv package and then click apply. 

## Installing from the Anaconda command line

From the Anaconda terminal run the command `conda install -c conda-forge opencv`

# Download the data and code

Please create a directory called "IntroML" on your Desktop. In this directory, create a subfolder called data (e.g., /Users/username/Desktop/IntroML/data)

Download the the following files and place them in the data subfolder you just created:

* [Gapminder Life Expectancy Data](data/gapminder-life-expectancy.csv)
* [World Bank GDP Data](data/worldbank-gdp.csv)
* [World Bank GDP Data with outliers](data/worldbank-gdp-outliers.csv)

If you are using a Mac or Linux system the following commands will download this:

~~~
mkdir data
cd data
wget https://scw-aberystwyth.github.io/machine-learning-novice/data/worldbank-gdp.csv
wget https://scw-aberystwyth.github.io/machine-learning-novice/data/worldbank-gdp-outliers.csv
wget https://scw-aberystwyth.github.io/machine-learning-novice/data/gapminder-life-expectancy.csv
~~~
{: .language-bash}

Download the following python script and place it in the IntroML directory located on your Desktop (e.g., /Users/username/Desktop/IntroML/workshop_helper_functions.py)
* [workshop_helper_functions.py](code/workshop_helper_functions.py)


{% include links.md %}
