# Assignment \#3 Starter

These files are from the 2015 version of the course. They contain Assignment \#2 of that year, and its solution.

Some caveats:

- The paper we were using is Paper 2b in our course (i.e., the one with filename 2\_mathieu.pdf). It has a similar model, but it is not exactly the same notation.
- The solution is not entirely correct, so just use it as a guide for getting unstuck on the problem, or to help you think about the problem.
- Though this approach to solve the problem works and can lead you to fully understand the way the regression model is implemented, you may have an easier time finding a proper solution by using a linear regression package such as the one contained in the *statsmodel* package for Python.

*Note*: If you are using the statsmodel (which you may have to if you want to avoid singular matrix errors), you will benefit from reading:
- [Getting Started](http://www.statsmodels.org/stable/gettingstarted.html?highlight=dummy)
- [The Patsy Documentation](https://patsy.readthedocs.io/en/latest/quickstart.html?highlight=dummy), which is a package for defining Design Matrices more easily (in the style of R).
- And it looks like if you want to this in the R-style, then you want to define the model somewhat like: 
'''
dmatrices('power\_values ~ C(timeOfWeek) + Tc1 + Tc2 + Tc3 + Tc4 + Tc5 + Tc6', data=df, return_type='dataframe')
'''
Here, the Tc1 through Tc6 are the temperature components (the same columns that you had in the design matrix you created), and C(timeOfWeek) is the dummy variable for each of the 480 times of the week (it will automatically do that for you if timeOfWeek is a variable containing the number for the time of the week of every data point in your set).
I'm still exploring this, and some of you more familiar with R would have a better sense of how to use this, so if you do: please chime in.
