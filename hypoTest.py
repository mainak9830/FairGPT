# Team Members - Aishik Deb, Mainak Adak, Hao Lin, Yuqing Wang
# Description - Python file to use hypothesis testing for bias evaluation purposes
# Concept Used -  Hypothesis Testing
# System Used - Google Cloud VM Instance using Ubuntu

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, shapiro, levene, wilcoxon
from scipy import stats

def visualize_bar_dist(x, y, title, x_label, y_label):
    plt.bar(x,y)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

# def remove_all_0_pair(arr):
#     # find columns with all zeros
#     zero_cols = np.all(arr == 0, axis=0)
#     # remove columns with all zeros
#     arr = arr[:, ~zero_cols]
#     zero_cols_indices = np.where(zero_cols)[0]
#     # print(zero_cols_indices)
#     return arr, zero_cols_indices
        
        
# Read the csv file and return a dataframe
def read_profession_csv(gender, race):
    df = pd.read_csv('/Users/miku/Downloads/Final/'+race+'_'+gender+'_profession_corrected.csv')
    # Drop the columns that are not needed
    df.drop(['race', 'gender', 'story_id', 'profession', 'response'], axis=1, inplace=True)
    return df
    

#  get mean value for each profession grouped by name
def get_profession_list(df):
    # group by name and compute the mean of the remaining columns for each group.
    grouped = df.groupby('name').mean()
    # Generate a list for each column, except for the 'name' column
    lists = [grouped[col].tolist() for col in grouped.columns if col != 'name']
    return lists


def two_sample_ttest(sample1, sample2, alpha = 0.05):
    # check variance equivalence
    _, p = levene(sample1, sample2)
    if p > alpha:
        # val equal
        # print('equal variance')
        print(type(sample1))
        print(len(sample1))
        _, p = stats.ttest_ind(sample1, sample2, equal_var=True)
    else:
        # val diff
        # print('unequal variance')
        _, p = stats.ttest_ind(sample1, sample2, equal_var=False)
    return p

def two_sample_mean_compare(sample1, sample2, alpha = 0.05):
    # Check normality for two sample
    t_statistic = 0
    # perform Shapiro-Wilk test for test group
    _, p1 = shapiro(sample1)
    # perform Shapiro-Wilk test for refer group
    _, p2 = shapiro(sample2)
    bothnormal = False
    # two sample both normal distribution
    if p1 > alpha and p2 > alpha:
        bothnormal = True
        # print('normal distribution')
        p = two_sample_ttest(sample1, sample2, alpha)
    else:
        # one of them not normal distribution
        # print('not normal distribution')
        _, p = wilcoxon(sample1, sample2)
    return t_statistic, p, bothnormal


def wlicoxon_alternative(sample1, sample2):
    statistic, pvalue = wilcoxon(sample1, sample2, alternative='less')
    return statistic, pvalue


def parse_prof_list(list, job_df):
    parsed_list = [] * len(list)
    for i in range(len(list)):
        parsed_list.append((job_df.columns[i], list[i][1]))
    return parsed_list


def parse_prof_lists(lists, job_df, sample1_name, sample2_name):
    for i in range(len(lists)):
        lists[i] = parse_prof_list(lists[i], job_df)
    diff_df = pd.DataFrame(lists[0], columns=['profession', 'p_value'])
    equal_df = pd.DataFrame(lists[1], columns=['profession', 'p_value'])
    sample1_df = pd.DataFrame(lists[2], columns=['profession', 'p_value'])
    sample2_df = pd.DataFrame(lists[3], columns=['profession', 'p_value'])
    print(f"Professions that are different between {sample1_name} and {sample2_name}:")
    print(diff_df.to_string(index=False))
    print(f"Professions that are equal between {sample1_name} and {sample2_name}:")
    print(equal_df.to_string(index=False))
    print(f"Professions that are dominant in {sample1_name}:")
    print(sample1_df.to_string(index=False))
    print(f"Professions that are dominant in {sample2_name}:")
    print(sample2_df.to_string(index=False))
    

def two_group_ttest(test_group, refer_group, alpha = 0.05):
    diff_professions_index = []
    equal_professions_index = []
    sample1_dominant_professions = []
    sample2_dominant_professions = []
    for i in range(len(test_group)):
        # print(i)
        sample1 = test_group[i]
        sample2 = refer_group[i]
        t_statistic, p_value, ttest = two_sample_mean_compare(sample1, sample2, alpha)
        if p_value < alpha:
            diff_professions_index.append((i, p_value))
            if t_statistic > 0:
                sample1_dominant_professions.append((i, p_value))
            elif t_statistic < 0 and ttest:
                sample2_dominant_professions.append((i, p_value))
            if not ttest:
                _, p_val = wlicoxon_alternative(sample1, sample2)
                if p_val < alpha:
                    sample1_dominant_professions.append((i, p_value))
                else:
                    sample2_dominant_professions.append((i, p_value))
            
        else:
            equal_professions_index.append((i, p_value))
        
    return [diff_professions_index, equal_professions_index, sample1_dominant_professions, sample2_dominant_professions]


def get_sum(df):
    return df.iloc[:, 1:].sum().tolist()



white_male_sample = read_profession_csv('male', 'white')
asian_male_sample = read_profession_csv('male','asian')
black_male_sample = read_profession_csv('male', 'black')
hispanic_male_sample = read_profession_csv('male','hispanics')


white_female_sample = read_profession_csv('female', 'white')
asain_female_sample = read_profession_csv('female','asian')
black_female_sample = read_profession_csv('female', 'black')
hispanic_female_sample = read_profession_csv('female','hispanics')


# Hypothesis 1:
# For each gender there is (4 races * 50 names * 100 stories) = 20,000 jobs
male_profession = pd.concat([white_male_sample, asian_male_sample, black_male_sample, hispanic_male_sample])
female_profession = pd.concat([white_female_sample, asain_female_sample, black_female_sample, hispanic_female_sample])

male_chi = get_sum(male_profession)
female_chi = get_sum(female_profession)
# Conduct a k-sample proportion test
gender_table = np.array([male_chi, female_chi])
# gender_table = remove_all_0_pair(gender_table)

# perform chi-square test for independence
chi2_stat, p_val, dof, expected = chi2_contingency(gender_table)

# print results
print("Chi-square statistic: ", chi2_stat)
print("P-value: ", p_val)
print("Degrees of freedom: ", dof)
print("Expected frequencies: ", expected)
if p_val < 0.05:
    print("Reject null hypothesis, gender and profession are dependent")
else:
    print("Accept null hypothesis, gender and profession are independent.")


# Two Sample T-test
# Null hypothsis: The gender and profession category are independent
male_profession_lst = get_profession_list(male_profession)
female_profession_lst = get_profession_list(female_profession)

job_df = male_profession.drop(['name'], axis=1)

# conduct two sample t-test for male and female
ttest_M_F = two_group_ttest(male_profession_lst, female_profession_lst)
parse_prof_lists(ttest_M_F, job_df, "Male", "Female")




# Hypothesis 2
# For each race, there is 2 genders * 50 names * 100 stories = 10,000
white_profession = pd.concat([white_male_sample, white_female_sample])
white_pro_chi = get_sum(white_profession)


asain_profession = pd.concat([asian_male_sample, asain_female_sample])
asain_pro_chi = get_sum(asain_profession)

black_profession = pd.concat([black_male_sample, black_female_sample])
black_pro_chi = get_sum(black_profession)

hispanics_profession = pd.concat([hispanic_male_sample, hispanic_female_sample])
hispanic_pro_chi = get_sum(hispanics_profession)

race_table = np.array([white_pro_chi, asain_pro_chi, black_pro_chi, hispanic_pro_chi])
# race_table, _ = remove_all_0_pair(race_table)

# perform chi-square test for independence
chi2_stat, p_val, dof, expected = chi2_contingency(race_table)

# print results
print("Chi-square statistic: ", chi2_stat)
print("P-value: ", p_val)
print("Degrees of freedom: ", dof)
print("Expected frequencies: ", expected)

if p_val < 0.05:
    print("Reject null hypothesis, race and profession are dependent")
else:
    print("Accept null hypothesis, race and profession are independent.")


white_profession = get_profession_list(white_profession)
asain_profession = get_profession_list(asain_profession)
black_profession = get_profession_list(black_profession)
hispanics_profession = get_profession_list(hispanics_profession)

# conduct two sample t-test for white and asain
ttest_W_A = two_group_ttest(white_profession, asain_profession)
parse_prof_lists(ttest_W_A, job_df, "White", "Asian")

# conduct two sample t-test for white and black
ttest_W_B = two_group_ttest(white_profession, black_profession)
parse_prof_lists(ttest_W_B, job_df, "White", "Black")

# conduct two sample t-test for white and hispanic
ttest_W_H = two_group_ttest(white_profession, hispanics_profession)
parse_prof_lists(ttest_W_H, job_df, "White", "Hispanic")


# Hypothesis 3
# prepare data
white_male_profession = get_profession_list(white_male_sample)
white_female_profession = get_profession_list(white_female_sample)
asian_male_profession = get_profession_list(asian_male_sample)
asian_female_profession = get_profession_list(asain_female_sample)
black_male_profession = get_profession_list(black_male_sample)
black_female_profession = get_profession_list(black_female_sample)
hispanic_male_profession = get_profession_list(hispanic_male_sample)
hispanic_female_profession = get_profession_list(hispanic_female_sample)

white_male_chi = get_sum(white_male_sample)
white_female_chi = get_sum(white_female_sample)
asian_male_chi = get_sum(asian_male_sample)
asian_female_chi = get_sum(asain_female_sample)
black_male_chi = get_sum(black_male_sample)
black_female_chi = get_sum(black_female_sample)
hispanic_male_chi = get_sum(hispanic_male_sample)
hispanic_female_chi = get_sum(hispanic_female_sample)

# Social group chi-square test
group_table = np.array([white_male_chi, white_female_chi,
                        asian_male_chi, asian_female_chi,
                        black_male_chi,black_female_chi,
                        hispanic_male_chi, hispanic_female_chi
                        ])

# perform chi-square test for independence
chi2_stat, p_val, dof, expected = chi2_contingency(group_table)

# print results
print("Chi-square statistic: ", chi2_stat)
print("P-value: ", p_val)
print("Degrees of freedom: ", dof)
print("Expected frequencies: ", expected)

if p_val < 0.05:
    print("Reject null hypothesis, intersection group and profession are dependent")
else:
    print("Accept null hypothesis, intersection group and profession are independent.")




# 3.1 Compare api_male, black_male and hispanic_male with white_male
# White Male and Asain Male
ttest_WM_AM = two_group_ttest(white_male_profession, asian_male_profession)
parse_prof_lists(ttest_WM_AM, job_df, "White Male", "Asian Male")


# White Male and Black Male
ttest_WM_BM = two_group_ttest(white_male_profession, black_male_profession)
parse_prof_lists(ttest_WM_BM, job_df, "White Male", "Black Male")


# White Male and Hispanic Male
ttest_WM_HM = two_group_ttest(white_male_profession, hispanic_male_profession)
parse_prof_lists(ttest_WM_HM, job_df, "White Male", "Hispanic Male")

# 3.2 Compare api_female, black_female and hispanic_female with white_female
# White Female and Asain Female
ttest_WF_AF = two_group_ttest(white_female_profession, asian_female_profession)
parse_prof_lists(ttest_WM_HM, job_df, "White Female", "Asian Female")

# White Female and Black Female
ttest_WF_BF = two_group_ttest(white_female_profession, black_female_profession)
parse_prof_lists(ttest_WF_BF, job_df, "White Female", "Black Female")

# White Female and Hispanic Female
ttest_WF_HF = two_group_ttest(white_female_profession, hispanic_female_profession)
parse_prof_lists(ttest_WF_HF, job_df, "White Female", "Hispanic Female")


# 3.3 Compare within same race
# Asain Male and Asain Female
ttest_AM_AF = two_group_ttest(asian_male_profession, asian_female_profession)
parse_prof_lists(ttest_AM_AF, job_df, "Asian Male", "Asian Female")

# Black Male and Black Female
ttest_BM_BF = two_group_ttest(black_male_profession, black_female_profession)
parse_prof_lists(ttest_BM_BF, job_df, "Black Male", "Black Female")

# Hispanic Male and Hispanic Female
ttest_HM_HF = two_group_ttest(hispanic_male_profession, hispanic_female_profession)
parse_prof_lists(ttest_HM_HF, job_df, "Hispanic Male", "Hispanic Female")

