#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Initialize Otter
import otter
grader = otter.Notebook("hw07.ipynb")


# # Homework 7: Testing Hypotheses

# **Helpful Resource:**
# - [Python Reference](http://data8.org/fa21/python-reference.html): Cheat sheet of helpful array & table methods used in Data 8!
# 
# **Recommended Readings**: 
# * [Random Functions guide](https://drive.google.com/file/d/1_kkBJhbS_OPEhUTSogL4syaVHkbyzKSf/view)
# * [Testing Hypotheses](https://www.inferentialthinking.com/chapters/11/Testing_Hypotheses.html)
# * [A/B Testing](https://inferentialthinking.com/chapters/12/1/AB_Testing.html)

# Please complete this notebook by filling in the cells provided. Before you begin, execute the following cell to setup the notebook by importing some helpful libraries. Each time you start your server, you will need to execute this cell again.
# 
# For all problems that you must write explanations and sentences for, you **must** provide your answer in the designated space. **Moreover, throughout this homework and all future ones, please be sure to not re-assign variables throughout the notebook!** For example, if you use `max_temperature` in your answer to one question, do not reassign it later on. Otherwise, you will fail tests that you thought you were passing previously!
# 
# **Deadline:**
# 
# This assignment is due Thursday, 10/14 at 11:59pm PT. Turn it in by Wednesday 11:59pm PT for 5 extra credit points. Late work will not be accepted as per the [policies](http://data8.org/fa21/policies.html) page.
# 
# 
# **Note: This homework has hidden tests on it. That means even though tests may say 100% passed, it doesn't mean your final grade will be 100%. We will be running more tests for correctness once everyone turns in the homework.**
# 
# 
# Directly sharing answers is not okay, but discussing problems with the course staff or with other students is encouraged. Refer to the policies page to learn more about how to learn cooperatively.
# 
# You should start early so that you have time to get help if you're stuck. Office hours are held Monday-Friday. The schedule appears on [http://data8.org/fa21/office-hours.html](http://data8.org/fa21/office-hours.html).

# In[2]:


# Run this cell to set up the notebook, but please don't change it.

# These lines import the Numpy and Datascience modules.
import numpy as np
from datascience import *
import d8error

# These lines do some fancy plotting magic.
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import warnings
warnings.simplefilter('ignore', FutureWarning)


# ## 1. Vaccinations Across The Nation
# A vaccination clinic has two types of vaccines against a disease. Each person who comes in to be vaccinated gets either Vaccine 1 or Vaccine 2. One week, everyone who came in on Monday, Wednesday, and Friday was given Vaccine 1. Everyone who came in on Tuesday and Thursday was given Vaccine 2. The clinic is closed on weekends.
# 
# Doctor Adhikari at the clinic said, "Oh wow, it's just like tossing a coin that lands heads with chance $\frac{3}{5}$. Heads you get Vaccine 1 and Tails you get Vaccine 2."
# 
# But Doctor Wagner said, "No, it's not. We're not doing anything like tossing a coin."
# 
# That week, the clinic gave Vaccine 1 to 211 people and Vaccine 2 to 107 people. Conduct a test of hypotheses to see which doctor's position is better supported by the data.

# **Question 1.1.** Given the information above, what was the sample size, and what was the percentage of people who got **Vaccine 1?** **(4 points)**
# 
# *Note*: Your percent should be a number between 0 and 100.
# 
# <!--
# BEGIN QUESTION
# name: q1_1
# manual: false
# points:
#  - 0
#  - 0
#  - 0
#  - 0
#  - 2
#  - 2
# -->

# In[3]:


sample_size = 211 + 107
percent_V1 = 211/sample_size * 100

print(f"Sample Size: {sample_size}")
print(f"Vaccine 1 Percent: {percent_V1}")


# In[4]:


grader.check("q1_1")


# <!-- BEGIN QUESTION -->
# 
# **Question 1.2.** State the null hypothesis. It should reflect the position of either Dr. Adhikari or Dr. Wagner. **(4 points)**
# 
# <!--
# BEGIN QUESTION
# name: q1_2
# manual: true
# -->

# _The distribution of vaccine frequencies follows a model that yields V1 ~60% of the time._

# <!-- END QUESTION -->
# 
# <!-- BEGIN QUESTION -->
# 
# **Question 3.** State the alternative hypothesis. It should reflect the position of the doctor you did not choose to represent in Question 1.2. **(4 points)**
# 
# <!--
# BEGIN QUESTION
# name: q1_3
# manual: true
# -->

# _The distribution of vaccine frequencies does not follow a ~60% V1 model._

# <!-- END QUESTION -->
# 
# **Question 1.4.** One of the test statistics below is appropriate for testing these hypotheses. Assign the variable `valid_test_stat` to the number corresponding to the correct test statistic. **(4 points)**
# 
# 1. percent of heads - 60
# 2. percent of heads - 50
# 3. |percent of heads - 60|
# 4. |percent of heads - 50|
# 
# <!--
# BEGIN QUESTION
# name: q1_4
# manual: false
# points:
#  - 0
#  - 0
#  - 4
# -->

# In[5]:


valid_test_stat = 3
valid_test_stat


# In[6]:


grader.check("q1_4")


# **Question 1.5.** Using your answer from Questions 1.1 and 1.4, find the observed value of the test statistic and assign it to the variable `observed_statistic`. **(4 points)**
# 
# <!--
# BEGIN QUESTION
# name: q1_5
# manual: false
# points:
#  - 0
#  - 0
#  - 4
# -->

# In[7]:


observed_statistic = abs(percent_V1 - 60)
observed_statistic


# In[8]:


grader.check("q1_5")


# **Question 1.6.** In order to perform this hypothesis test, you must simulate the test statistic. From the three options below, pick the assumption that is needed for this simulation. Assign `assumption_needed` to an integer corresponding to the assumption. **(4 points)**
# 
# 1. The statistic must be simulated under the null hypothesis.
# 2. The statistic must be simulated under the alternative hypothesis.
# 3. No assumptions are needed. We can just simulate the statistic.
# 
# <!--
# BEGIN QUESTION
# name: q1_6
# manual: false
# points:
#  - 0
#  - 0
#  - 4
# -->

# In[9]:


assumption_needed = 1
assumption_needed


# In[10]:


grader.check("q1_6")


# <!-- BEGIN QUESTION -->
# 
# **Question 1.7.** Simulate 20,000 values of the test statistic under the assumption you picked in Question 1.6. **(4 points)** 
# 
# As usual, start by defining a function that simulates one value of the statistic. Your function should use `sample_proportions`. Then write a `for` loop to simulate multiple values, and collect them in the array `simulated_statistics`. 
# 
# Use as many lines of code as you need. We have included the code that visualizes the distribution of the simulated values. The red dot represents the observed statistic you found in Question 1.5.
# 
# <!--
# BEGIN QUESTION
# name: q1_7
# manual: true
# -->

# In[12]:


def one_simulated_statistic():
    null_props = make_array(3/5, 2/5)
    sim_props = sample_proportions(sample_size, null_props)
    test_stat = abs(sim_props.item(0) * 100 - 60)
    return test_stat

num_simulations = 20000

simulated_statistics = make_array()

for i in np.arange(num_simulations):
    simulated_statistics = np.append(simulated_statistics, one_simulated_statistic())


# In[13]:


# Run this cell to produce a histogram of the simulated statistics

Table().with_columns('Simulated Statistic', simulated_statistics).hist()
plt.scatter(observed_statistic, -0.002, color='red', s=40);


# **Question 1.8.** Using `simulated_statistics`, `observed_statistic`, and `num_simulations`, find the empirical p-value based on the simulation. **(4 points)**
# 
# <!--
# BEGIN QUESTION
# name: q1_8
# manual: false
# points:
#  - 0
#  - 0
#  - 4
# -->

# In[77]:


p_value = np.count_nonzero(simulated_statistics >= observed_statistic)/num_simulations
p_value


# In[31]:


grader.check("q1_8")


# **Question 1.9.** Assign `correct_doctor` to the number corresponding to the correct statement below. Use the 5% cutoff for the p-value. **(4 points)**
# 
# 1. The data support Dr. Adhikari's position more than they support Dr. Wagner's.
# 2. The data support Dr. Wagner's position more than they support Dr. Adhikari's.
# 
# As a reminder, here are the two claims made by Dr. Adhikari and Dr. Wagner:
# > **Doctor Adhikari:** "Oh wow, it's just like tossing a coin that lands heads with chance $\frac{3}{5}$. Heads you get Vaccine 1 and Tails you get Vaccine 2."
# 
# >**Doctor Wagner:** "No, it's not. We're not doing anything like tossing a coin."
# 
# <!--
# BEGIN QUESTION
# name: q1_9
# manual: false
# points:
#  - 0
#  - 0
#  - 4
# -->

# In[16]:


correct_doctor = 2
correct_doctor


# In[17]:


grader.check("q1_9")


# ## 2. Using TVD as a Test Statistic
# 
# Before beginning this section, please read [this section](https://inferentialthinking.com/chapters/11/2/Multiple_Categories.html#a-new-statistic-the-distance-between-two-distributions) of the textbook on TVD!

# Total variation distance (TVD) is a special type of test statistic that we use when we want to compare two distributions of categorical data. It is often used when we observe that a set of observed proportions/probabilities is different than what we expect under the null model. 
# 
# Consider a six-sided die that we roll 6,000 times. If the die is fair, we would expect that each face comes up $\frac{1}{6}$ of the time. By random chance, a fair die won't always result in equal proportions (that is, we won't get exactly 1000 of each face). However, if we suspect that the die might be unfair based on the data, we can conduct a hypothesis test using TVD to compare the expected [$\frac{1}{6}$, $\frac{1}{6}$, $\frac{1}{6}$, $\frac{1}{6}$, $\frac{1}{6}$, $\frac{1}{6}$] distribution to what is actually observed.
# 
# In this part of the homework, we'll look at how we can use TVD to determine the effect that different factors have on happiness. 
# 
# We will be working with data from the [Gallup World Poll](https://www.gallup.com/analytics/349487/gallup-global-happiness-center.aspx#:~:text=World%20Happiness%20Report&text=Using%20the%20Gallup%20World%20Poll,about%20the%20World%20Happiness%20Report.) that is presented in the World Happiness Report, a survey of the state of global happiness. The survey ranked 155 countries by overall happiness and estimated the influence that economic production, social support, life expectancy, freedom, absence of corruption, and generosity had on population happiness. The study has been repeated for several years - we'll be looking at data from the 2016 survey. **Run the cell below to load in the `happiness_scores` table.**

# In[18]:


happiness_scores = Table.read_table("happiness_scores.csv")
happiness_scores.show(5)


# Participants in the study were asked to evaluate their life satisfaction from a scale of 0 (worst possible life) to 10 (best possible life). The responses for each country were averaged to create the `Happiness Score`.
# 
# The columns `Economy (GDP per Capita)`, `Family`, `Health (Life Expectancy)`, `Freedom`, `Trust (Government Corruption)`, and `Generosity` estimate the extent to which each factor influences happiness, both for better or for worse. The higher the value, the more influential that factor was in calculating the country's happiness score. If you add up all the factors (in addition to a "Difference from Dystopia" value we excluded in the dataset), you get the happiness score.
# 
# Let's look at the different factors that affect happiness in the United States. Run the cell below to view the row in `us_happiness` that contains data for the United States.

# In[19]:


us_happiness = happiness_scores.where("Country", "United States")
us_happiness


# To compare the different factors, we'll look at the proportion of the happiness score that is attributed to each variable. 
# You can find these proportions in the table `us_happiness_factors` after running the cell below.
# 
# *Note:* The factors shown in `us_happiness` don't add up exactly to the happiness score, so we adjusted the proportions to  only account for the data we have access to. The proportions were found by dividing each Happiness Factor value by the sum of all Happiness Factor values in `us_happiness`.

# In[20]:


us_happiness_factors = Table().read_table("us_happiness_factors.csv")
us_happiness_factors


# <!-- BEGIN QUESTION -->
# 
# **Question 2.1.** Suppose we want to test whether or not each factor contributes the same amount to the overall Happiness Score. Define the null hypothesis, alternative hypothesis, and test statistic in the cell below. Check your work with a neighbor, LA, or uGSI. **(4 points)**
# 
# *Note:* Please format your answer as follows:
# - Null Hypothesis: ...  
# - Alternative Hypothesis: ...  
# - Test Statistic: ...  
# 
# <!--
# BEGIN QUESTION
# name: q2_1
# manual: true
# -->

# _Null Hypothesis: Each given factor of the happiness scale contributes the same to the overall Happiness Score.
# Alternative Hypothesis: The factors do not contribute equally.
# Test Statistic: The TVD between null distribution and sample distribution._

# <!-- END QUESTION -->
# 
# **Question 2.2.** Write a function `calculate_tvd` that takes in the observed distribution (`obs_dist`) and expected distribution under the null hypothesis (`null_dist`) and calculates the total variation distance. Use this function to set `observed_tvd` to be equal to the observed test statistic. **(4 points)**
# 
# <!--
# BEGIN QUESTION
# name: q2_2
# manual: false
# points:
#  - 2
#  - 2
# -->

# In[23]:


null_distribution = np.ones(6) * 1/6

def calculate_tvd(obs_dist, null_dist):
    distances = abs(obs_dist - null_dist)
    tvd = sum(distances)/2
    return tvd
    
observed_tvd = calculate_tvd(us_happiness_factors.column(1), null_distribution)
observed_tvd


# In[24]:


grader.check("q2_2")


# **Question 2.3.** Create an array called `simulated_tvds` that contains 10,000 simulated values under the null hypothesis. Assume that the original sample consisted of 1,000 individuals. **(4 points)**
# 
# *Hint:* The `sample_proportions` function may be helpful to you. Refer to the [Python Reference sheet](http://data8.org/fa21/python-reference.html#:~:text=sample_proportions(sample_size%2C%20model_proportions)) to read up on it!
# 
# <!--
# BEGIN QUESTION
# name: q2_3
# manual: false
# points:
#  - 1
#  - 1
#  - 1
#  - 1
# -->

# In[25]:


simulated_tvds = make_array()

for i in np.arange(10000):
    trial_dist = sample_proportions(1000, null_distribution)
    simulated_tvds = np.append(simulated_tvds, calculate_tvd(trial_dist, null_distribution))
    
simulated_tvds


# In[26]:


grader.check("q2_3")


# Run the cell below to plot a histogram of your simulated test statistics, as well as the observed value of the test statistic.

# In[27]:


Table().with_column("Simulated TVDs", simulated_tvds).hist()
plt.scatter(observed_tvd, 0, color='red', s=70, zorder=2);
plt.show();


# **Question 2.4.** Use your simulated statistics to calculate the p-value of your test. Make sure that this number is consistent with what you observed in the histogram above. **(4 points)**
# 
# <!--
# BEGIN QUESTION
# name: q2_4
# manual: false
# points:
#  - 4
# -->

# In[29]:


p_value_tvd = np.count_nonzero(observed_tvd <= simulated_tvds)/10000
p_value_tvd


# In[32]:


grader.check("q2_4")


# <!-- BEGIN QUESTION -->
# 
# **Question 2.5.** What can you conclude about how each factor contributes to the overall happiness score in the US? Explain your answer using the results of your hypothesis test. Assume a p-value cutoff of 5%. **(4 points)**
# 
# <!--
# BEGIN QUESTION
# name: q2_5
# manual: true
# -->

# _Each factor does not contribute evenly, as the observed statistic is well outside the p-cutoff for acceptable likelihood of getting said statistic under the null hypothesis._

# <!-- END QUESTION -->

# ## 3. Who is Older?
# 
# Data scientists have drawn a simple random sample of size 500 from a large population of adults. Each member of the population happened to identify as either "male" or "female". Data was collected on several attributes of the sampled people, including age. The table `sampled_ages` contains one row for each person in the sample, with columns containing the individual's gender identity.

# In[33]:


sampled_ages = Table.read_table('age.csv')
sampled_ages.show(5)


# **Question 3.1.** How many females were there in our sample? Keep in mind that `group` sorts categories in alphabetical order. **(4 points)**
# 
# <!--
# BEGIN QUESTION
# name: q3_1
# manual: false
# points:
#  - 0
#  - 4
# -->

# In[34]:


num_females = sampled_ages.group("Gender").sort("Gender").column("count").item(0)
num_females


# In[35]:


grader.check("q3_1")


# **Question 3.2.** Complete the cell below so that `avg_male_vs_female` evaluates to `True` if the sampled males are older than the sampled females on average, and `False` otherwise. Use Python code to achieve this. **(4 points)**
# 
# <!--
# BEGIN QUESTION
# name: q3_2
# manual: false
# points:
#  - 0
#  - 4
# -->

# In[37]:


group_mean_tbl = sampled_ages.group("Gender", np.average).sort("Gender")
group_means = group_mean_tbl.column(1)       # array of mean ages
avg_male_vs_female = group_means.item(0) < group_means.item(1)
avg_male_vs_female


# In[38]:


grader.check("q3_2")


# **Question 3.3.** The data scientists want to use the data to test whether males are older than females or, in other words, whether the ages of the two groups have the same distribution. One of the following statements is their null hypothesis and another is their alternative hypothesis. Assign `null_statement_number` and `alternative_statement_number` to the numbers of the correct statements in the code cell below. **(4 points)**
# 
# 1. In the sample, the males and females have the same distribution of ages; the sample averages of the two groups are different due to chance.
# 2. In the population, the males and females have the same distribution of ages; the sample averages of the two groups are different due to chance.
# 3. The age distributions of males and females in the population are different due to chance.
# 4. The males in the sample are older than the females, on average.
# 5. The males in the population are older than the females, on average.
# 6. The average ages of the males and females in the population are different.
# 
# <!--
# BEGIN QUESTION
# name: q3_3
# manual: false
# points:
#  - 0
#  - 0
#  - 0
#  - 0
#  - 0
#  - 2
#  - 2
# -->

# In[39]:


null_statement_number = 2
alternative_statement_number = 5


# In[40]:


grader.check("q3_3")


# **Question 3.4.** The data scientists have decided to use a permutation test. Assign `permutation_test_reason` to the number corresponding to the reason they made this choice. **(4 points)**
# 
# 1. Since a person's age can't be related to their gender under the null hypothesis, it doesn't matter who is labeled "male" and who is labeled "female", so you can use permutations.
# 2. Under the null hypothesis, permuting the labels in the `sampled_ages` table is equivalent to drawing a new random sample with the same number of males and females as in the original sample.
# 3. Under the null hypothesis, permuting the rows of `sampled_ages` table is equivalent to drawing a new random sample with the same number of males and females as in the original sample.
# 
# <!--
# BEGIN QUESTION
# name: q3_4
# manual: false
# points:
#  - 0
#  - 0
#  - 4
# -->

# In[41]:


permutation_test_reason = 2
permutation_test_reason


# In[42]:


grader.check("q3_4")


# **Question 3.5.** To test their hypotheses, the data scientists have followed our textbook's advice and chosen a test statistic where the following statement is true: Large values of the test statistic favor the alternative hypothesis.
# 
# The data scientists' test statistic is one of the two options below. Which one is it? Assign the appropriate number to the variable `correct_test_stat`. **(4 points)**
# 
# 1. "male age average - female age average" in a sample created by randomly shuffling the male/female labels
# 2. "|male age average - female age average|" in a sample created by randomly shuffling the male/female labels
# 
# <!--
# BEGIN QUESTION
# name: q3_5
# manual: false
# points:
#  - 0
#  - 0
#  - 4
# -->

# In[43]:


correct_test_stat = 1
correct_test_stat


# In[44]:


grader.check("q3_5")


# **Question 3.6.** Complete the cell below so that `observed_statistic_ab` evaluates to the observed value of the data scientists' test statistic. Use as many lines of code as you need, and remember that you can use any quantity, table, or array that you created earlier. **(4 points)**
# 
# <!--
# BEGIN QUESTION
# name: q3_6
# manual: false
# points:
#  - 0
#  - 0
#  - 4
# -->

# In[78]:


observed_statistic_ab = group_means.item(1) - group_means.item(0)
observed_statistic_ab


# In[79]:


grader.check("q3_6")


# **Question 3.7.** Assign `shuffled_labels` to an array of shuffled male/female labels. The rest of the code puts the array in a table along with the data in `sampled_ages`. **(4 points)**
# 
# <!--
# BEGIN QUESTION
# name: q3_7
# manual: false
# points:
#  - 0
#  - 0
#  - 0
#  - 1
#  - 1
#  - 1
#  - 1
# -->

# In[91]:


shuffled_labels = sampled_ages.select("Gender").sample(with_replacement=False).column(0)
original_with_shuffled_labels = sampled_ages.with_column("Shuffled Label", shuffled_labels)
original_with_shuffled_labels


# In[92]:


grader.check("q3_7")


# **Question 3.8.** [Pretend this is a midterm problem and solve it without doing the calculation in a code cell.] The comparison below uses the array `shuffled_labels` from Question 3.7 and the count `num_females` from Question 3.1. **(4 points)**
# 
# For this comparison, assign the correct number from one of the following options to the variable `correct_q8`.
# 
# `comp = np.count_nonzero(shuffled_labels == 'female') == num_females`
# 
# 1. `comp` is set to `True`.
# 2. `comp` is set to `False`.
# 3. `comp` is set to `True`, or `False`, depending on how the shuffle came out.
# 
# <!--
# BEGIN QUESTION
# name: q3_8
# manual: false
# points:
#  - 0
#  - 0
#  - 4
# -->

# In[93]:


correct_q8 = 1
correct_q8


# In[94]:


grader.check("q3_8")


# **Question 3.9.** Define a function `simulate_one_statistic` that takes no arguments and returns one simulated value of the test statistic. We've given you a skeleton, but feel free to approach this question in a way that makes sense to you. Use as many lines of code as you need. Refer to the code you have previously written in this problem, as you might be able to re-use some of it. **(4 points)**
# 
# <!--
# BEGIN QUESTION
# name: q3_9
# manual: false
# points:
#  - 4
# -->

# In[95]:


def simulate_one_statistic():
    "Returns one value of our simulated test statistic"
    shuffles = sampled_ages.select("Gender").sample(with_replacement=False)
    with_shuffles = sampled_ages.with_column("Shuffled", shuffles).drop(0)
    grouped_shuffle_avgs = with_shuffles.group(1, np.average).sort(0).column(1)
    statistic = grouped_shuffle_avgs.item(1) - grouped_shuffle_avgs.item(0)
    return statistic


# In[96]:


grader.check("q3_9")


# After you have defined your function, run the following cell a few times to see how the statistic varies.

# In[85]:


simulate_one_statistic()


# **Question 3.10.** Complete the cell to simulate 4,000 values of the statistic. We have included the code that draws the empirical distribution of the statistic and shows the value of `observed_statistic_ab` from Question 3.6. **(4 points)**
# 
# *Note:* This cell will take around a minute to run.
# 
# <!--
# BEGIN QUESTION
# name: q3_10
# manual: false
# points:
#  - 4
# -->

# In[86]:


repetitions = 4000

simulated_statistics_ab = make_array()

for i in np.arange(repetitions):
    simulated_statistics_ab = np.append(simulated_statistics_ab, simulate_one_statistic())

Table().with_columns('Simulated Statistic', simulated_statistics_ab).hist()
plt.scatter(observed_statistic_ab, -0.002, color='red', s=70);


# In[87]:


grader.check("q3_10")


# **Question 3.11.** Use the simulation to find an empirical approximation to the p-value. Assign `p_val` to the appropriate p-value from this simulation. Then, assign `conclusion` to either `null_hyp` or `alt_hyp`. **(4 points)** 
# 
# *Note:* Assume that we use the 5% cutoff for the p-value.
# 
# <!--
# BEGIN QUESTION
# name: q3_11
# manual: false
# points:
#  - 0
#  - 0
#  - 0
#  - 2
#  - 2
# -->

# In[88]:


# These are variables provided for you to use.
null_hyp = 'The data are consistent with the null hypothesis.'
alt_hyp = 'The data support the alternative more than the null.'

p_val = np.count_nonzero(observed_statistic_ab <= simulated_statistics_ab)/repetitions
conclusion = null_hyp

p_val, conclusion # Do not change this line


# In[89]:


grader.check("q3_11")


# You're done with Homework 7!  
# 
# **Important submission steps:** 
# 1. Run the tests and verify that they all pass.
# 2. Choose **Save and Checkpoint** from the **File** menu, then **run the final cell**. 
# 3. Click the link to download the zip file.
# 4. Go to [Gradescope](https://www.gradescope.com/courses/288788) and submit the zip file to the corresponding assignment. The name of this assignment is "Homework 7 Autograder". 
# 
# **It is your responsibility to make sure your work is saved before running the last cell.**

# ---
# 
# To double-check your work, the cell below will rerun all of the autograder tests.

# In[ ]:


grader.check_all()


# ## Submission
# 
# Make sure you have run all cells in your notebook in order before running the cell below, so that all images/graphs appear in the output. The cell below will generate a zip file for you to submit. **Please save before exporting!**

# In[ ]:


# Save your notebook first, then run this cell to export your submission.
grader.export(pdf=False, force_save=True)


#  
