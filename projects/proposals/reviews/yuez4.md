# General comments:
Generally, the idea is interesting and should work for a project. The proposed method, however, can be improved. My recommendation would be to set the whole thing up as a multiple-linear regression task with the average daily consumption as the dependent variable, and weather + day-of-the-week indicator variables as the independent variables. You could then add yet another indicator variable for holidays. Finally, you would do a simple t-test for the estimated coefficient corresponding to the holiday indicator variable and this should tell you whether the effect is significant (i.e., whether there is a significant difference in the average response of your model between holidays and non-holidays). Once you establish this (for which, of course, you'll need to know the holidays in France), then you can go ahead and try to cluster the data to see if you can find these holidays without having to know them a-priori. Of course this last task seems a bit unneccessary since it is usually the case that one knows when holidays occur.

# Detailed comments:
Here are some questions I had while reading the proposal:

- I think the above comments should be sufficient. Let me know if you have more questions.
