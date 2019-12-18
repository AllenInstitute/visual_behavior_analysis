import ternary  # triangle plotting package
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats

import matplotlib.pyplot as plt


def relevant_df_only(dataframe, 
                    hierarchy_columns_list, 
                    metric_column):
    """takes an input dataframe  and reduces it to just the columns that are relevant to the the current analysis
    
    Arguments:
        dataframe {pandas dataframe} -- dataframe that at minimum contains all of the columns and the metric you'd like to compute variance over and for. 
        hierarchy_columns_list {list of strings} -- List with the column names to compute variance for in order. 
                                        First item in the list will be the "bottom" or the first thing to be averaged over
        metric_column {str} -- the column name for metric you'd like to compute the variance of
    
    Returns:
        dataframe -- a dataframe that is only the columns that are relevant to the current analysis
    """
    relevant_columns = hierarchy_columns_list + metric_column
    relevant_df = dataframe[relevant_columns].copy()
    return relevant_df



def compute_level_variance_mean(level_df, 
                                level_column):
    """uses pandas .var to compute the variance of a given level. Then uses numpy.mean to compute the mean of that variance
    
    Arguments:
        level_df {pandas dataframe} -- dataframe consisting of the level column and the metric
        level_column {dataframe column} -- the column of data that variance is currently being calculated for
    
    Returns:
        float -- the mean of the variance for the current level(summarizing the variance of the level with 1 number)
    """
    
    level_variance_mean = np.mean(level_df.groupby(level_column).var())[0]
    return level_variance_mean



def compute_level_variance_std(level_df,
                                level_column):
    """uses pandas .var to compute the variance of a given leve. Then uses numpy.std to compute the standard deviation of that variance
    
    Arguments:
        level_df {pandas dataframe} -- dataframe consisting of the level column and the metric
        level_column {dataframe column} -- the column of data that variance is currently being calculated for
    
    Returns:
        float -- the standard deviation of the variance for the current level(summarizing the variance of the level with 1 number)
    """
    level_variance_std = np.std(level_df.groupby(level_column).var())[0]
    return level_variance_std



def gen_level_mean_df(level_df, 
                    level_column):
    """computes the mean of the variance for each unique item in a level and returns a dataframe
    
    Arguments:
       level_df {pandas dataframe} -- dataframe consisting of the level column and the metric
       level_column {dataframe column} -- the column of data that variance is currently being calculated for
    
    Returns:
        dataframe -- a dataframe with a column that is each unique item at that level, and a column which is the mean of the variance of the metric
    """
    
    level_mean_df=level_df.groupby(level_column).mean()
    level_mean_df[level_column[0]] = level_mean_df.index
    level_mean_df = level_mean_df.reset_index(drop=True)
    return level_mean_df



def get_previous_level(hierarchy_columns_list, 
                        level):

    current_level_index = hierarchy_columns_list.index(level)
    previous_level = hierarchy_columns_list[(current_level_index -1)]
    return previous_level



def collapse_relevant_df(relevant_df, 
                        metric_column,  
                        hierarchy_columns_list,  
                        level):
    """ "collapses" the dataframe on a specific level. The metric column is now the mean of the 
        metric for each unique item in the level rather than being all the individual metric measurements. 
        The previous level is removed from the relevant dataframe because it is no longer relevant. 
    
    Arguments:
        relevant_df {pandas dataframe} -- the input dataframe- or previous iteration of 
        metric_column {[type]} -- [description]
        hierarchy_columns_list {[type]} -- [description]
        level {str} -- the string of the level 
    
    Returns:
        [type] -- [description]
    """
    
    relevant_df = relevant_df.drop(metric_column, axis=1).drop_duplicates()
    if hierarchy_columns_list.index(level) > 0:
        previous_level = get_previous_level(hierarchy_columns_list, level)
        relevant_df = relevant_df.drop(previous_level, axis=1).drop_duplicates()
    return relevant_df



def compute_total_variance(variance_df):
    total_variance = variance_df["variance_mean"].sum()
    return total_variance



def add_total_variance_to_variance_df(variance_df):
    total_variance = compute_total_variance(variance_df)
    total_var_df = pd.DataFrame({"level":"total_variance", "variance_mean": total_variance}, index = [0])
    variance_df = variance_df.append(total_var_df, sort = True)
    return variance_df
    


def compute_variance_hierarchy(dataframe, 
                                hierarchy_columns_list, 
                                metric_column_name, 
                                print_stmts):

    metric_column = [metric_column_name]
    variance_df = pd.DataFrame()
    relevant_df = relevant_df_only(dataframe, 
                                    hierarchy_columns_list, 
                                    metric_column)
  
    for level in hierarchy_columns_list:
        if print_stmts == True:
            print("calculating " + level + " variance")

        level_column = [level]
        level_df = relevant_df_only(relevant_df, level_column, metric_column)
        
        level_variance_mean = compute_level_variance_mean(level_df, level_column)
        level_variance_std = compute_level_variance_std(level_df, level_column)
        level_variance_df = pd.DataFrame({"level":level, "variance_mean": level_variance_mean, "variance_std":level_variance_std}, index = [0])
        variance_df = variance_df.append(level_variance_df,  sort = True)

        level_mean_df = gen_level_mean_df(level_df, level_column)
        relevant_df = collapse_relevant_df(relevant_df, metric_column, hierarchy_columns_list, level)
        relevant_df = pd.merge(level_mean_df, relevant_df, how = "left", on = level)
    
    variance_df = variance_df.reset_index(drop=True)
    return variance_df



def compute_variance_ratios(variance_df):
    total_variance = compute_total_variance(variance_df)
    variance_df = add_total_variance_to_variance_df(variance_df)
    variance_df = variance_df.reset_index(drop=True)
    variance_df["variance_ratio"] = variance_df["variance_mean"] / total_variance
    return variance_df



def generate_variance_df(dataframe, 
                        hierarchy_columns_list, 
                        metric_column_name, 
                        print_stmts = False):

    variance_df = compute_variance_hierarchy(dataframe, 
                                            hierarchy_columns_list, 
                                            metric_column_name,
                                             print_stmts= print_stmts)
    
    variance_df = compute_variance_ratios(variance_df)
    return variance_df



def plot_variance_ratios(variance_df):
    sns.pointplot(x="level", y= "variance_ratio", ci =None, data = variance_df.iloc[:-1])
    yerr = variance_df['variance_std']
    plt.errorbar(x=variance_df["level"], y=variance_df["variance_ratio"], yerr=yerr, fmt= "none", c ="k" )
    plt.ylim(-.05, 1)

    plt.show()








