#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Initialize Otter
import otter
grader = otter.Notebook("project1.ipynb")


# # Project 1: World Progress

# In this project, you'll explore data from [Gapminder.org](http://gapminder.org), a website dedicated to providing a fact-based view of the world and how it has changed. That site includes several data visualizations and presentations, but also publishes the raw data that we will use in this project to recreate and extend some of their most famous visualizations.
# 
# The Gapminder website collects data from many sources and compiles them into tables that describe many countries around the world. All of the data they aggregate are published in the [Systema Globalis](https://github.com/open-numbers/ddf--gapminder--systema_globalis/blob/master/README.md). Their goal is "to compile all public statistics; Social, Economic and Environmental; into a comparable total dataset." All data sets in this project are copied directly from the Systema Globalis without any changes.
# 
# This project is dedicated to [Hans Rosling](https://en.wikipedia.org/wiki/Hans_Rosling) (1948-2017), who championed the use of data to understand and prioritize global development challenges.

# ### Logistics
# 
# **Deadline.**  This project is due at **11:59pm PT on Friday 10/1**. Projects will be accepted up to 2 days (48 hours) late. Projects submitted fewer than 24 hours after the deadline will receive 2/3 credit, and projects submitted between 24 and 48 hours after the deadline will receive 1/3 credit. Projects submitted 48 hours or more after the deadline will receive no credit. It's **much** better to be early than late, so start working now.
# 
# **Checkpoint.**  For full credit on the checkpoint, you must complete the first 8 questions, pass all public autograder tests for those sections, and submit to the Gradescope Project 1 Checkpoint assignment by **11:59pm PT on Friday, 9/24**. **This is worth 5% of your entire project grade**. After you've submitted the checkpoint, you may still change your answers before the project deadline - only your final submission, to the Project 1 assignment, will be graded for correctness. You will have some lab time to work on these questions, but we recommend that you start the project before lab and leave time to finish the checkpoint afterward.
# 
# **Partners.** You may work with one other partner; your partner must be from your assigned lab section. **Only one of you is allowed to submit the project. If both partners submit, you will be docked 10% of your project grade. On Gradescope, the person who submits should also designate their partner so that both of you receive credit.** Once you submit, click into your submission, and there will be an option to Add Group Member in the top right corner. You may also reference [this walkthrough video](https://drive.google.com/file/d/1POtij6KECSBjCUeOC_F0Lt3ZmKN7LKIq/view?usp=sharing) on how to add partners on Gradescope.
# 
# 
# **Rules.** Don't share your code with anybody but your partner. You are welcome to discuss questions with other students, but don't share the answers. The experience of solving the problems in this project will prepare you for exams (and life). If someone asks you for the answer, resist! Instead, you can demonstrate how you would solve a similar problem.
# 
# **Support.** You are not alone! Come to office hours, post on Ed, and talk to your classmates. If you want to ask about the details of your solution to a problem, make a private Ed post and the staff will respond. If you're ever feeling overwhelmed or don't know how to make progress, email your TA or tutor for help. You can find contact information for the staff on the [course website](http://data8.org/fa21/staff.html).
# 
# **Tests.** The tests that are given are **not comprehensive** and passing the tests for a question **does not** mean that you answered the question correctly. Tests usually only check that your table has the correct column labels. However, more tests will be applied to verify the correctness of your submission in order to assign your final score, so be careful and check your work! You might want to create your own checks along the way to see if your answers make sense. Additionally, before you submit, make sure that none of your cells take a very long time to run (several minutes).
# 
# **Free Response Questions:** Make sure that you put the answers to the written questions in the indicated cell we provide. **Every free response question should include an explanation** that adequately answers the question.
# 
# **Tabular Thinking Guide:** Feel free to reference [Tabular Thinking Guide](https://drive.google.com/file/d/1NvbBECCBdI0Ku380oPcTUOcpjH3RI230/view) for extra guidance.
# 
# **Advice.** Develop your answers incrementally. To perform a complicated table manipulation, break it up into steps, perform each step on a different line, give a new name to each result, and check that each intermediate result is what you expect. You can add any additional names or functions you want to the provided cells. Make sure that you are using distinct and meaningful variable names throughout the notebook. Along that line, **DO NOT** reuse the variable names that we use when we grade your answers. For example, in Question 1 of the Global Poverty section we ask you to assign an answer to `latest`. Do not reassign the variable name `latest` to anything else in your notebook, otherwise there is the chance that our tests grade against what `latest` was reassigned to.
# 
# You **never** have to use just one line in this project or any others. Use intermediate variables and multiple lines as much as you would like!
# 
# ---
# 
# To get started, load `datascience`, `numpy`, and `plots`.

# In[2]:


# Run this cell to set up the notebook, but please don't change it.

# These lines import the NumPy and Datascience modules.
from datascience import *
import numpy as np

# These lines do some fancy plotting magic.
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plots
plots.style.use('fivethirtyeight')

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

import d8error


# ## 1. Global Population Growth

# The global population of humans reached 1 billion around 1800, 3 billion around 1960, and 7 billion around 2011. The potential impact of exponential population growth has concerned scientists, economists, and politicians alike.
# 
# The UN Population Division estimates that the world population will likely continue to grow throughout the 21st century, but at a slower rate, perhaps reaching 11 billion by 2100. However, the UN does not rule out scenarios of more extreme growth.
# 
# <a href="http://www.pewresearch.org/fact-tank/2015/06/08/scientists-more-worried-than-public-about-worlds-growing-population/ft_15-06-04_popcount/"> 
#  <img src="pew_population_projection.png"/> 
# </a>
# 
# In this part of the project, we will examine some of the factors that influence population growth and how they have been changing over the years and around the world. There are two main sub-parts of this analysis.
# 
# - First, we will examine the data for one country, Bangladesh. We will see how factors such as life expectancy, fertility rate, and child mortality have changed over time in Bangladesh, and how they are related to the rate of population growth.
# - Next, we will examine whether the changes we have observed for Bangladesh are particular to that country or whether they reflect general patterns observable in other countries too. We will study aspects of world population growth and see how they have been changing.
# 
# The first table we will consider contains the total population of each country over time. Run the cell below.

# In[3]:


population = Table.read_table('population.csv').where("time", are.below(2021))
population.show(3)


# **Note:** The population csv file can also be found [here](https://github.com/open-numbers/ddf--gapminder--systema_globalis/blob/master/countries-etc-datapoints/ddf--datapoints--population_total--by--geo--time.csv). The data for this project was downloaded in February 2017. 

# ### Bangladesh
# 
# The nation of [Bangladesh](https://en.wikipedia.org/wiki/Bangladesh#People's_Republic_of_Bangladesh) was established as a parliamentary democracy after the Bangladesh Liberation War ended in 1971. The war-ravaged fledgling nation was almost immediately faced with floods and famine. In this section of the project, we will examine aspects of the development of Bangladesh since that time.
# 
# In the `population` table, the `geo` column contains three-letter codes established by the [International Organization for Standardization](https://en.wikipedia.org/wiki/International_Organization_for_Standardization) (ISO) in the [Alpha-3](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-3#Current_codes) standard. We will begin by taking a close look at Bangladesh. Use the Alpha-3 link to find the 3-letter code for Bangladesh.

# **Question 1.** Create a table called `b_pop` that has two columns labeled `time` and `population_total`. The first column should contain the years from 1970 through 2020 (including both 1970 and 2020) and the second should contain the population of Bangladesh in each of those years.
# 
# <!--
# BEGIN QUESTION
# name: q1_1
# points:
# - 0
# - 0
# - 0
# - 4
# -->

# In[4]:


b_pop = population.where("geo", "bgd").where("time", are.above_or_equal_to(1970)).drop("geo")
b_pop


# In[5]:


try:
    print(grader.check("q1_1"))
except NameError:
    print("NameError: name 'grader' is not defined")
    print("Please run the cells at the very top of this notebook, from the top down, then run this cell again.")


# Run the following cell to create a table called `b_five` that has the population of Bangladesh every five years. At a glance, it appears that the population of Bangladesh has been growing quickly indeed!

# In[6]:


b_pop.set_format('population_total', NumberFormatter)

fives = np.arange(1970, 2021, 5) # 1970, 1975, 1980, ...
b_five = b_pop.sort('time').where('time', are.contained_in(fives))
b_five.show()


# **Question 2.** Assign `initial` to an array that contains the population for every five year interval from 1970 to 2015 (inclusive). Then, assign `changed` to an array that contains the population for every five year interval from 1975 to 2020 (inclusive). The first array should include both 1970 and 2015, and the second array should include both 1975 and 2020. You should use the `b_five` table to create both arrays, by first filtering the table to only contain the relevant years.
# 
# The annual growth rate for a time period is equal to:
# 
# $$(\frac{\text{Population at end of period}}{\text{Population at start of period}}^{\frac{1}{\text{number of years}}} )-1$$
# 
# We have provided the code below that uses  `initial` and `changed` in order to add a column to `b_five` called `annual_growth`. Don't worry about the calculation of the growth rates; run the test below to test your solution.
# 
# If you are interested in how we came up with the formula for growth rates, consult the [growth rates](https://inferentialthinking.com/chapters/03/2/1/Growth.html) section of the textbook.
# 
# <!--
# BEGIN QUESTION
# name: q1_2
# points:
# - 0
# - 0
# - 0
# - 0
# - 0
# - 4
# -->

# In[7]:


initial = b_five.where("time", are.below_or_equal_to(2015)).column("population_total")
changed = b_five.where("time", are.above_or_equal_to(1975)).column("population_total")

b_1970_through_2015 = b_five.where('time', are.below_or_equal_to(2015)) 
b_five_growth = b_1970_through_2015.with_column('annual_growth', (changed/initial)**0.2-1)
b_five_growth.set_format('annual_growth', PercentFormatter)


# In[8]:


grader.check("q1_2")


# While the population has grown every five years since 1970, the annual growth rate decreased dramatically from 1985 to 2015. Let's look at some other information in order to develop a possible explanation. Run the next cell to load three additional tables of measurements about countries over time.

# In[9]:


life_expectancy = Table.read_table('life_expectancy.csv').where('time', are.below(2021))
child_mortality = Table.read_table('child_mortality.csv').relabel(2, 'child_mortality_under_5_per_1000_born').where('time', are.below(2021))
fertility = Table.read_table('fertility.csv').where('time', are.below(2021))


# The `life_expectancy` table contains a statistic that is often used to measure how long people live, called *life expectancy at birth*. This number, for a country in a given year, [does not measure how long babies born in that year are expected to live](http://blogs.worldbank.org/opendata/what-does-life-expectancy-birth-really-mean). Instead, it measures how long someone would live, on average, if the *mortality conditions* in that year persisted throughout their lifetime. These "mortality conditions" describe what fraction of people at each age survived the year. So, it is a way of measuring the proportion of people that are staying alive, aggregated over different age groups in the population.

# Run the following cells below to see `life_expectancy`, `child_mortality`, and `fertility`. Refer back to these tables as they will be helpful for answering further questions!

# In[10]:


life_expectancy.show(3)


# In[11]:


child_mortality.show(3)


# In[12]:


fertility.show(3)


# <!-- BEGIN QUESTION -->
# 
# **Question 3.** Perhaps population is growing more slowly because people aren't living as long. Use the `life_expectancy` table to draw a line graph with the years 1970 and later on the horizontal axis that shows how the *life expectancy at birth* has changed in Bangladesh.
# 
# <!--
# BEGIN QUESTION
# name: q1_3
# manual: True
# -->

# In[13]:


#Fill in code here
life_expectancy.where("geo", "bgd").where("time", are.above_or_equal_to(1970)).plot("time", "life_expectancy_years")


# <!-- END QUESTION -->
# 
# <!-- BEGIN QUESTION -->
# 
# **Question 4.** Assuming everything else stays the same, do the trends in life expectancy in the graph above directly explain why the population growth rate decreased from 1985 to 2015 in Bangladesh? Why or why not? 
# 
# Hint: What happened in Bangladesh in 1991, and does that event explain the overall change in population growth rate? This [webpage](https://en.wikipedia.org/wiki/1991_Bangladesh_cyclone) provides relevant context.
# 
# <!--
# BEGIN QUESTION
# name: q1_4
# manual: True
# -->

# _The trends in life expectancy do not directly explain why population growth rate is decreasing. Life expectancy has steadily increased from the 1970s until today, so citizens of Bangladesh aren't living short enough lives to drive down the growth rate. The Bangladesh cyclone briefly dropped the growth rate in 1991, but it was a single event whose conditions were quickly reversed and the growth rate recovered very quickly._

# <!-- END QUESTION -->
# 
# 
# 
# The `fertility` table contains a statistic that is often used to measure how many babies are being born, the *total fertility rate*. This number describes the [number of children a woman would have in her lifetime](https://www.measureevaluation.org/prh/rh_indicators/specific/fertility/total-fertility-rate), on average, if the current rates of birth by age of the mother persisted throughout her child bearing years, assuming she survived through age 49. 

# **Question 5.** Complete the function `fertility_over_time`. It takes the Alpha-3 code of a country as `country_code` and a `start` year. It returns a two-column table with labels `Year` and `Children per woman` that can be used to generate a line chart of the country's fertility rate each year, starting at the `start` year. The plot should include the `start` year and all later years that appear in the `fertility` table. 
# 
# Then, determine the Alpha-3 code for Bangladesh. The code for Bangladesh and the year 1970 are used in the call to your `fertility_over_time` function in order to plot how Bangladesh's fertility rate has changed since 1970. Note that the function `fertility_over_time` should not return the plot itself. **The expression that draws the line plot is provided for you; please don't change it.**
# 
# <!--
# BEGIN QUESTION
# name: q1_5
# points:
# - 0
# - 0
# - 0
# - 0
# - 0
# - 4
# -->

# In[14]:


def fertility_over_time(country_code, start):
    """Create a two-column table that describes a country's total fertility rate each year."""
    country_fertility = fertility.where("geo", country_code)
    country_fertility_after_start = country_fertility.where("time", are.above_or_equal_to(start))
    cleaned_table = country_fertility_after_start.drop("geo").relabeled(
        "time", "Year").relabeled("children_per_woman_total_fertility", "Children per woman")
    return cleaned_table

bangladesh_code = "bgd"
fertility_over_time(bangladesh_code, 1970).plot(0, 1) # You should *not* change this line.


# In[15]:


grader.check("q1_5")


# <!-- BEGIN QUESTION -->
# 
# **Question 6.** Assuming everything else is constant, do the trends in fertility in the graph above help directly explain why the population growth rate decreased from 1980 to 2020 in Bangladesh? Why or why not?
# 
# <!--
# BEGIN QUESTION
# name: q1_6
# manual: True
# -->

# _Yes, this graph shows a sharp decrease in fertility between 1980 and 2020. Prior to the decline, Bangladeshi women were having more than 6 children each on average, but by 2020 they were only having two. This means fewer babies are being added to the population every year, significantly slowing population growth even with longer lifespans for those actually born._

# <!-- END QUESTION -->
# 
# 
# 
# It has been [observed](https://www.ageing.ox.ac.uk/download/143) that lower fertility rates are often associated with lower child mortality rates. The link has been attributed to family planning: if parents can expect that their children will all survive into adulthood, then they will choose to have fewer children. In the reverse direction, having fewer children may allow families to devote more resources to each child, reducing child mortality. We can see if this association is evident in Bangladesh by plotting the relationship between total fertility rate and [child mortality rate per 1000 children](https://en.wikipedia.org/wiki/Child_mortality).

# **Question 7.** Using both the `fertility` and `child_mortality` tables, draw a scatter diagram that has Bangladesh's total fertility on the horizontal axis and its child mortality on the vertical axis with one point for each year, starting with 1970.
# 
# **The code that draws the scatter diagram is provided for you; please don't change it.** Instead, create a table called `post_1969_fertility_and_child_mortality` with the appropriate column labels and data in order to generate the chart correctly. Use the label `Children per woman` to describe total fertility and the label `Child deaths per 1000 born` to describe child mortality.
# 
# *Hint*: Do not drop the `time` column or you will get an error in the scatterplot in the next cell!
# 
# <!--
# BEGIN QUESTION
# name: q1_7
# manual: false
# points:
# - 0
# - 0
# - 0
# - 0
# - 0
# - 0
# - 0
# - 4
# -->

# In[16]:


bgd_fertility = fertility.where("geo", "bgd").relabeled("children_per_woman_total_fertility", "Children per woman")
bgd_child_mortality = child_mortality.where("geo", "bgd").relabeled("child_mortality_under_5_per_1000_born", 
                                                                    "Child deaths per 1000 born")
fertility_and_child_mortality = bgd_fertility.join("time", bgd_child_mortality)
post_1969_fertility_and_child_mortality = fertility_and_child_mortality.where("time", are.above_or_equal_to(1970))

# Don't change anything below this line!
x_births = post_1969_fertility_and_child_mortality.column("Children per woman")
y_deaths = post_1969_fertility_and_child_mortality.column("Child deaths per 1000 born")
time_colors = post_1969_fertility_and_child_mortality.column("time")

plots.figure(figsize=(6,6))
plots.scatter(x_births, y_deaths, c=time_colors, cmap="Blues_r")
plots.colorbar()                  
plots.xlabel("Children per woman")
plots.ylabel("Child deaths per 1000 born");


# In[17]:


grader.check("q1_7")


# The plot above uses **color** to encode data about the `time` column from the table `post_1969_fertility_and_child_mortality`. The colors, ranging from dark blue to white, represent the passing of time between the 1970s to the 2020s. For example, a point on the scatter plot representing data from the 1970s would appear as **dark blue** and a point from the 2010s would appear as **light blue**.
# 

# <!-- BEGIN QUESTION -->
# 
# **Question 8.** In one or two sentences, describe the association (if any) that is illustrated by this scatter diagram. Does the diagram show that reduced child mortality *causes* parents to choose to have fewer children?
# 
# <!--
# BEGIN QUESTION
# name: q1_8
# manual: True
# -->

# _This plot shows a very strong positive association between the number of children a woman has and the child mortality rate, but it alone does not show that the reduced rate is the cause of the choice women make to have fewer children. To prove causality, one would have to show that with every other condition in Bangladesh the same, any change in the mortality rate corresponded to a change in the number of children women had to the same degree illustrated in the chart above._

# <!-- END QUESTION -->
# 
# 
# 
# ### Checkpoint (due Friday 9/24 by 11:59 PM PT)
# **Congratulations, you have reached the checkpoint! Run the following cells and submit to the Gradescope assignment corresponding to the checkpoint: <u>Project 1 Checkpoint</u>**

# --- 
# 
# To double check your work, the cell below will rerun all of the autograder tests for Section 1. 

# In[18]:


checkpoint_tests = ["q1_1", "q1_2", "q1_5", "q1_7"]
for test in checkpoint_tests:
    display(grader.check(test))


# ## Submission
# 
# Make sure you have run all cells in your notebook in order before running the cell below, so that all images/graphs appear in the output. The cell below will generate a zip file for you to submit. **Please save before exporting!**

# In[19]:


# Save your notebook first, then run this cell to export your submission.
grader.export(pdf=False)


# ### The World
# 
# The change observed in Bangladesh since 1970 can also be observed in many other developing countries: health services improve, life expectancy increases, and child mortality decreases. At the same time, the fertility rate often plummets, and so the population growth rate decreases despite increasing longevity.

# Run the cell below to generate two overlaid histograms, one for 1962 and one for 2010, that show the distributions of total fertility rates for these two years among all 201 countries in the `fertility` table.

# In[20]:


Table().with_columns(
    '1962', fertility.where('time', 1962).column(2),
    '2010', fertility.where('time', 2010).column(2)
).hist(bins=np.arange(0, 10, 0.5), unit='child per woman')
_ = plots.xlabel('Children per woman')
_ = plots.ylabel('Percent per children per woman')
_ = plots.xticks(np.arange(10))


# **Question 9.** Assign `fertility_statements` to an array of the numbers of each statement below that can be correctly inferred from these histograms.
# 
# 1. About the same number of countries had a fertility rate between 3.5 and 4.5 in both 1962 and 2010.
# 1. In 1962, less than 20% of countries had a fertility rate below 3.
# 1. At least half of countries had a fertility rate between 5 and 8 in 1962.
# 1. In 2010, about 40% of countries had a fertility rate between 1.5 and 2.
# 1. At least half of countries had a fertility rate below 3 in 2010.
# 1. More countries had a fertility rate above 3 in 1962 than in 2010.
# 
# <!--
# BEGIN QUESTION
# name: q1_9
# points:
# - 0
# - 4
# -->

# In[21]:


fertility_statements = make_array(1, 2, 3, 5, 6)


# In[22]:


grader.check("q1_9")


# <!-- BEGIN QUESTION -->
# 
# **Question 10.** Draw a line plot of the world population from 1800 through 2020 (inclusive of both endpoints). The world population is the sum of all of the countries' populations. You should use the `population` table defined earlier in the project. 
# 
# <!--
# BEGIN QUESTION
# name: q1_10
# manual: True
# -->

# In[23]:


#Fill in code here
population.where("time", are.below_or_equal_to(2020)).drop("geo").group("time", sum).plot("time", "population_total sum")


# <!-- END QUESTION -->
# 
# **Question 11.** Create a function `stats_for_year` that takes a `year` and returns a table of statistics. The table it returns should have four columns: `geo`, `population_total`, `children_per_woman_total_fertility`, and `child_mortality_under_5_per_1000_born`. Each row should contain one unique Alpha-3 country code and three statistics: population, fertility rate, and child mortality for that `year` from the `population`, `fertility` and `child_mortality` tables. Only include rows for which all three statistics are available for the country and year.
# 
# In addition, restrict the result to country codes that appears in `big_50`, an array of the 50 most populous countries in 2020. This restriction will speed up computations later in the project.
# 
# After you write `stats_for_year`, try calling `stats_for_year` on any year between 1960 and 2020. Try to understand the output of stats_for_year.
# 
# *Hint*: The tests for this question are quite comprehensive, so if you pass the tests, your function is probably correct. However, without calling your function yourself and looking at the output, it will be very difficult to understand any problems you have, so try your best to write the function correctly and check that it works before you rely on the `grader` tests to confirm your work.
# 
# *Hint*: What do all three tables have in common (pay attention to column names)?
# 
# <!--
# BEGIN QUESTION
# name: q1_11
# manual: false
# points:
# - 0
# - 0
# - 0
# - 4
# -->

# In[24]:


# We first create a population table that only includes the 
# 50 countries with the largest 2020 populations. We focus on 
# these 50 countries only so that plotting later will run faster.
big_50 = population.where('time', are.equal_to(2020)).sort("population_total", descending=True).take(np.arange(50)).column('geo')
population_of_big_50 = population.where('time', are.above(1959)).where('geo', are.contained_in(big_50))

def stats_for_year(year):
    """Return a table of the stats for each country that year."""
    p = population_of_big_50.where('time', are.equal_to(year)).drop('time')
    f = fertility.where('time', are.equal_to(year)).drop('time')
    c = child_mortality.where('time', are.equal_to(year)).drop('time')
    f_50 = f.where("geo", are.contained_in(big_50))
    c_50 = c.where("geo", are.contained_in(big_50))
    stats = p.join("geo", f_50)
    stats = stats.join("geo", c_50)
    return stats


# In[25]:


grader.check("q1_11")


# **Question 12.** Create a table called `pop_by_decade` with two columns called `decade` and `population`, in this order. It has a row for each year that starts a decade, in increasing order starting with 1960 and ending with 2020. For example, 1960 is the start of the 1960's decade. The `population` column contains the total population of all countries included in the result of `stats_for_year(year)` for the first `year` of the decade. You should see that these countries contain most of the world's population.
# 
# *Hint:* One approach is to define a function `pop_for_year` that computes this total population, then `apply` it to the `decade` column.  The `stats_for_year` function from the previous question may be useful here.
# 
# This first test is just a sanity check for your helper function if you choose to use it. You will not lose points for not implementing the function `pop_for_year`.
# 
# **Note:** The cell where you will generate the `pop_by_decade` table is below the cell where you can choose to define the helper function `pop_for_year`. You should define your `pop_by_decade` table in the cell that starts with the table `decades` being defined. 
# 
# <!--
# BEGIN QUESTION
# name: q1_12_0
# manual: false
# points: 
# - 0
# - 0
# - 0
# - 0
# -->

# In[26]:


def pop_for_year(year):
    """Return the total population for the specified year."""
    return sum(stats_for_year(year).column("population_total"))


# In[27]:


grader.check("q1_12_0")


# Now that you've defined your helper function (if you've chosen to do so), define the `pop_by_decade` table.
# 
# <!--
# BEGIN QUESTION
# name: q1_12
# manual: false
# points:
# - 0
# - 0
# - 0
# - 0
# - 0
# - 0
# - 4
# -->

# In[28]:


decades = Table().with_column('decade', np.arange(1960, 2021, 10))

pop_by_decade = decades.with_column("population",decades.apply(pop_for_year, "decade"))
pop_by_decade.set_format(1, NumberFormatter)


# In[29]:


grader.check("q1_12")


# The `countries` table describes various characteristics of countries. The `country` column contains the same codes as the `geo` column in each of the other data tables (`population`, `fertility`, and `child_mortality`). The `world_6region` column classifies each country into a region of the world. Run the cell below to inspect the data.

# In[30]:


countries = Table.read_table('countries.csv').where('country', are.contained_in(population.group('geo').column('geo')))
countries.select('country', 'name', 'world_6region')


# **Question 13.** Create a table called `region_counts`. It should contain two columns called `region` and `count`. The `region` column should contain regions of the world, and the `count` column should contain the number of countries in each region that appears in the result of `stats_for_year(2020)`. 
# 
# For example, one row would have `south_asia` as its `region` value and an integer as its `count` value: the number of large South Asian countries for which we have population, fertility, and child mortality numbers from 2020.
# 
# *Hint*: You may have to relabel a column to name it `region`. 
# 
# <!--
# BEGIN QUESTION
# name: q1_13
# points:
# - 0
# - 0
# - 0
# - 4
# -->

# In[31]:


stats_for_2020 = stats_for_year(2020)
region_counts = stats_for_2020.join("geo", countries, "country").group("world_6region").select("world_6region",
                    "count").relabeled("world_6region", "region")
region_counts


# In[32]:


grader.check("q1_13")


# The following scatter diagram compares total fertility rate and child mortality rate for each country in 1960. The area of each dot represents the population of the country, and the color represents its region of the world. Run the cell. Do you think you can identify any of the dots?

# In[33]:


from functools import lru_cache as cache

# This cache annotation makes sure that if the same year
# is passed as an argument twice, the work of computing
# the result is only carried out once.
@cache(None)
def stats_relabeled(year):
    """Relabeled and cached version of stats_for_year."""
    return stats_for_year(year).relabel(2, 'Children per woman').relabel(3, 'Child deaths per 1000 born')

def fertility_vs_child_mortality(year):
    """Draw a color scatter diagram comparing child mortality and fertility."""
    with_region = stats_relabeled(year).join('geo', countries.select('country', 'world_6region'), 'country')
    with_region.scatter(2, 3, sizes=1, group=4, s=500)
    plots.xlim(0,10)
    plots.ylim(-50, 500)
    plots.title(year)
    plots.show()

fertility_vs_child_mortality(1960)


# **Question 14.** Assign `scatter_statements` to an array of the numbers of each statement below that can be inferred from this scatter diagram for 1960. 
# 1. As a whole, the `europe_central_asia` region had the lowest child mortality rate.
# 1. The lowest child mortality rate of any country was from an `east_asia_pacific` country.
# 1. Most countries had a fertility rate above 5.
# 1. There was an association between child mortality and fertility.
# 1. The two largest countries by population also had the two highest child mortality rates.
# 
# <!--
# BEGIN QUESTION
# name: q1_14
# points:
# - 0
# - 4
# -->

# In[34]:


scatter_statements = make_array(1, 3, 4)


# In[35]:


grader.check("q1_14")


# The result of the cell below is interactive. Drag the slider to the right to see how countries have changed over time. You'll find that the great divide between so-called "Western" and "developing" countries that existed in the 1960's has nearly disappeared. This shift in fertility rates is the reason that the global population is expected to grow more slowly in the 21st century than it did in the 19th and 20th centuries.
# 
# **Note:** Don't worry if a red warning pops up when running the cell below. You'll still be able to run the cell!

# In[36]:


import ipywidgets as widgets

_ = widgets.interact(fertility_vs_child_mortality, 
                     year=widgets.IntSlider(min=1960, max=2020, value=1960))


# Now is a great time to take a break and watch the same data presented by [Hans Rosling in a 2010 TEDx talk](https://www.gapminder.org/videos/reducing-child-mortality-a-moral-and-environmental-imperative) with smoother animation and witty commentary.

# ## 2. Global Poverty

# In 1800, 85% of the world's 1 billion people lived in [extreme poverty](https://en.wikipedia.org/wiki/Extreme_poverty), defined by the United Nations as "a condition characterized by severe deprivation of basic human needs, including food, safe drinking water, sanitation facilities, health, shelter, education and information." At the time when the data in this project were gathered, a common definition of extreme poverty was a person living on less than \$1.25 a day.
# 
# In 2018, the proportion of people living in extreme poverty was estimated to be [about 9\%](https://unstats.un.org/sdgs/report/2019/goal-01/). Although the world rate of extreme poverty has declined consistently for hundreds of years, the number of people living in extreme poverty is still over 600 million. The United Nations adopted an [ambitious goal](http://www.un.org/sustainabledevelopment/poverty/): "By 2030, eradicate extreme poverty for all people everywhere." In this part of the project we will examine aspects of global poverty that might affect whether the goal is achievable.

# First, load the population and poverty rate by country and year and the country descriptions. While the `population` table has values for every recent year for many countries, the `poverty` table only includes certain years for each country in which a measurement of the rate of extreme poverty was available.

# In[37]:


population = Table.read_table('population.csv')
countries = Table.read_table('countries.csv').where('country', are.contained_in(population.group('geo').column('geo')))
poverty = Table.read_table('poverty.csv')
poverty.show(3)


# **Question 1.** Assign `latest_poverty` to a three-column table with one row for each country that appears in the `poverty` table. The first column should contain the 3-letter code for the country. The second column should contain the most recent year for which an extreme poverty rate is available for the country. The third column should contain the poverty rate in that year. **Do not change the last line, so that the labels of your table are set correctly.**
# 
# *Hint*: think about how ```group``` works: it does a sequential search of the table (from top to bottom) and collects values in the array in the order in which they appear, and then applies a function to that array. The `first` function may be helpful, but you are not required to use it.
# 
# <!--
# BEGIN QUESTION
# name: q2_1
# points:
# - 0
# - 0
# - 4
# -->

# In[38]:


def first(values):
    return values.item(0)

latest_poverty = poverty.sort("time", descending=True).group("geo", first)
latest_poverty = latest_poverty.relabeled(0, 'geo').relabeled(1, 'time').relabeled(2, 'poverty_percent') # You should *not* change this line.
latest_poverty


# In[39]:


grader.check("q2_1")


# **Question 2.** Using both `latest_poverty` and `population`, create a four-column table called `recent_poverty_total` with one row for each country in `latest_poverty`. The four columns should have the following labels and contents:
# 1. `geo` contains the 3-letter country code,
# 1. `poverty_percent` contains the most recent poverty percent,
# 1. `population_total` contains the population of the country in 2010,
# 1. `poverty_total` contains the number of people in poverty **rounded to the nearest integer**, based on the 2010 population and most recent poverty rate.
# 
# *Hint*: You are not required to use `poverty_and_pop`, and you are always welcome to add any additional names. 
# <!--
# BEGIN QUESTION
# name: q2_2
# points:
# - 0
# - 0
# - 0
# - 4
# -->

# In[40]:


poverty_and_pop = latest_poverty.drop("time").join("geo", population.where("geo", are.contained_in(latest_poverty.column(
    "geo"))).where("time", are.equal_to(2010)).drop("time"))
recent_poverty_total = poverty_and_pop.with_column("poverty_total", np.round(poverty_and_pop.column("poverty_percent")/100 * 
    poverty_and_pop.column("population_total")))
recent_poverty_total


# In[41]:


grader.check("q2_2")


# **Question 3.** Assign the name `poverty_percent` to the known percentage of the world’s 2010 population that were living in extreme poverty. Assume that the `poverty_total` numbers in the `recent_poverty_total` table describe **all** people in 2010 living in extreme poverty. You should get a number that is above the 2018 global estimate of 9%, since many country-specific poverty rates are older than 2018.
# 
# *Hint*: The sum of the `population_total` column in the `recent_poverty_total` table is not the world population, because only a subset of the world's countries are included in the `recent_poverty_total` table (only some countries have known poverty rates). Use the `population` table to compute the world's 2010 total population.
# 
# *Hint*: We are computing a percentage (value between 0 and 100), not a proportion (value between 0 and 1).
# 
# <!--
# BEGIN QUESTION
# name: q2_3
# points:
# - 0
# - 4
# -->

# In[42]:


poverty_percent = sum(recent_poverty_total.column("poverty_total"))/sum(population.where("time", are.equal_to(
    2010)).column("population_total")) * 100
poverty_percent


# In[43]:


grader.check("q2_3")


# The `countries` table includes not only the name and region of countries, but also their positions on the globe.

# In[44]:


countries.select('country', 'name', 'world_4region', 'latitude', 'longitude')


# **Question 4.** Using both `countries` and `recent_poverty_total`, create a five-column table called `poverty_map` with one row for every country in `recent_poverty_total`.  The five columns should have the following labels and contents:
# 1. `latitude` contains the country's latitude,
# 1. `longitude` contains the country's longitude,
# 1. `name` contains the country's name,
# 1. `region` contains the country's region from the `world_4region` column of `countries`,
# 1. `poverty_total` contains the country's poverty total.
# 
# <!--
# BEGIN QUESTION
# name: q2_4
# points:
# - 0
# - 0
# - 0
# - 4
# -->

# In[45]:


poverty_map = countries.where("country", are.contained_in(recent_poverty_total.column("geo"))).join("country", 
    recent_poverty_total, "geo").select("latitude", "longitude", "name", "world_4region", "poverty_total").relabeled(
    "world_4region", "region")
poverty_map


# In[46]:


grader.check("q2_4")


# Run the cell below to draw a map of the world in which the areas of circles represent the number of people living in extreme poverty. Double-click on the map to zoom in.

# In[47]:


# It may take a few seconds to generate this map.
colors = {'africa': 'blue', 'europe': 'black', 'asia': 'red', 'americas': 'green'}
scaled = poverty_map.with_columns(
    'labels', poverty_map.column('name'),
    'colors', poverty_map.apply(colors.get, 'region'),
    'areas', 1e-4 * poverty_map.column('poverty_total')
).drop('name', 'region', 'poverty_total')

Circle.map_table(scaled)


# Although people lived in extreme poverty throughout the world in 2010 (with more than 5 million in the United States), the largest numbers were in Asia and Africa.

# **Question 5.** Assign `largest` to a two-column table with the `name` (not the 3-letter code) and `poverty_total` of the 10 countries with the largest number of people living in extreme poverty.
# 
# *Hint*: How can we use `take` and `np.arange` in conjunction with each other?
# 
# <!--
# BEGIN QUESTION
# name: q2_5
# points:
# - 0
# - 0
# - 0
# - 4
# -->

# In[48]:


largest = poverty_map.select("name", "poverty_total").sort("poverty_total", descending=True).take(np.arange(10))
largest.set_format('poverty_total', NumberFormatter)


# In[49]:


grader.check("q2_5")


# <!-- BEGIN QUESTION -->
# 
# **Question 6.** It is important to study the absolute number of people living in poverty, not just the percent. The absolute number is an important factor in determining the amount of resources needed to support people living in poverty. 
# 
# In Question 7, you will be asked to write a function called `poverty_timeline` that takes **the name of a country** as its argument  (not the Alpha-3 country code). It should draw a line plot of the number of people living in poverty in that country with time on the horizontal axis. The line plot should have a point for each row in the `poverty` table for that country. To compute the population living in poverty from a poverty percentage, multiply by the population of the country **in that year**.
# 
# For this question, write out a generalized process for Question 7. What should this function output, and what steps will you take within the function?
# 
# <!--
# BEGIN QUESTION
# name: q2_6
# manual: True
# -->

# _To get to the desired line plot output for each specified country argument, I must first construct a table for each country that the plot method can extract data from. This table will not be created until the return statement of the function, however--its data will come from a master table with all of "Poverty"'s countries' necessary poverty data compiled into one. To make that table, I will first join "Countries" and "Population" such that the "names" column values in "Countries" appear next to their corresponding "geo" values. I will then select only the relevant columns from this new table, and join them with "Poverty" via rows which share the same "name" AND "time" values in each table. Dropping the "country" column, adding the total poverty population column via column multiplication, and then selecting just that column, "name" and "time" will leave me with a table listing the total number of people in poverty in a given country and year. The function's return statement will then create a subtable with only its argument's data displayed, and then create a line plot showing the change in total poverty population over time._

# <!-- END QUESTION -->
# 
# 
# 
# **Question 7.** Now, we'll actually write the function called `poverty_timeline`. Recall that `poverty_timeline` takes **the name of a country** as its argument (not the Alpha-3 country code). It should draw a line plot of the number of people living in poverty in that country with time on the horizontal axis. The line plot should have a point for each row in the `poverty` table for that country. To compute the population living in poverty from a poverty percentage, multiply by the population of the country **in that year**.
# 
# *Hint:* This question is long. Feel free to create cells and experiment. You can create cells by going to the toolbar and hitting the `+` button, or by going to the `Insert` tab.

# In[50]:


def poverty_timeline(country):
    '''Draw a timeline of people living in extreme poverty in a country.'''
    geo = countries.join("country", population, "geo").select("country", "name", "time", "population_total").join(["country", 
        "time"], poverty, ["geo", "time"]).drop(0)
    geo = geo.with_column("poverty_total", geo.column(2) * geo.column(3)).select(1, 0, 4)
    # This solution will take multiple lines of code. Use as many as you need
    return geo.where(0, country).plot(1, 2)
    # Don't change anything below this line. 
    plots.title(country)
    plots.ylim(bottom=0)
    plots.show() # This should be the last line of your function. 


# <!-- BEGIN QUESTION -->
# 
# Finally, draw the line plots below to see how the world is changing. Pay attention to the axes! You can check your work by comparing your graphs to the ones on [gapminder.org](https://www.gapminder.org/tools/#$state$entities$show$country$/$in@=ind;;;;&marker$axis_y$which=number_of_people_in_poverty&scaleType=linear&spaceRef:null;;;&chart-type=linechart).
# 
# <!--
# BEGIN QUESTION
# name: q2_7
# manual: True
# -->

# In[51]:


poverty_timeline('India')
poverty_timeline('Nigeria')
poverty_timeline('China')
poverty_timeline('Colombia')
poverty_timeline('United States')


# <!-- END QUESTION -->
# 
# 
# 
# Although the number of people living in extreme poverty increased in some countries including Nigeria and the United States, the decreases in other countries, most notably the massive decreases in China and India, have shaped the overall trend that extreme poverty is decreasing worldwide, both in percentage and in absolute number. 
# 
# To learn more, watch [Hans Rosling in a 2015 film](https://www.gapminder.org/videos/dont-panic-end-poverty/) about the UN goal of eradicating extreme poverty from the world. 
# 
# Below, we've also added an interactive dropdown menu for you to visualize `poverty_timeline` graphs for other countries. Note that each dropdown menu selection may take a few seconds to run.

# In[57]:


# Just run this cell

all_countries = poverty_map.column('name')
_ = widgets.interact(poverty_timeline, country=list(all_countries))


# **Question 8.** Please fill out the [Official Project 1 Partner Form](https://docs.google.com/forms/d/e/1FAIpQLSe-nEnLYK4rCCQOu7kKNlmDynXUVMqUaoXWMAT9eK07M0WRnA/viewform). This will be used to assign grades to both partners. Note that this form is different from the form you should have received from your GSI for project partner matching. **This form is mandatory. However, if you are working with a partner, only one person needs to submit the form.** Assign `secret_word` to the secret word given at the end of the form.
# 
# <!--
# BEGIN QUESTION
# name: q2_8
# points:
# - 4
# -->

# In[59]:


secret_word = "penguin"


# In[60]:


grader.check("q2_8")


# **You're finished!** Congratulations on discovering many important facts about global poverty and demonstrating your mastery of table manipulation and data visualization. Time to submit.

# ---
# 
# To double-check your work, the cell below will rerun all of the autograder tests.

# In[61]:


grader.check_all()


# ## Submission
# 
# Make sure you have run all cells in your notebook in order before running the cell below, so that all images/graphs appear in the output. The cell below will generate a zip file for you to submit. **Please save before exporting!**

# In[62]:


# Save your notebook first, then run this cell to export your submission.
grader.export(pdf=False, force_save=True)


#  
