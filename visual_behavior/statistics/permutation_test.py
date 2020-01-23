import time
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt

############## NULL DISTRIBUTION ##############
def shuffle_subject_data(subject_df, 
                        sample_size, 
                        num_null_shuffles):

    subject_data = subject_df["measurement"].values
    num_sub_measures = len(subject_data)
    shuffled_data_indexes = np.random.choice(num_sub_measures, size = (sample_size, num_null_shuffles))
    shuffled_subject_data = subject_data[shuffled_data_indexes]
    return shuffled_subject_data


def shuffle_comparator_groups(subject_df, 
                            sample_size, 
                            num_null_shuffles):

    comparator_groups = np.asarray[True, False]
    shuffled_comparator_groups = np.random.choice(comparator_groups, size = (sample_size, num_null_shuffles))
    return shuffled_comparator_groups


def split_shuffled_data_by_shuffled_comp_groups(shuffled_subject_data, shuffled_comparator_groups):
    shuffled_grp1_assignments = shuffled_comparator_groups== True
    shuffled_grp0_assignments = shuffled_comparator_groups  == False

    grp1_data_points_per_shuffle = np.sum(shuffled_grp1_assignments, axis=0).astype(float)
    grp0_data_points_per_shuffle = np.sum(shuffled_grp0_assignments, axis=0).astype(float)
    
    shuffled_grp1_data = shuffled_subject_data * shuffled_grp1_assignments
    shuffled_grp0_data = shuffled_subject_data * shuffled_grp0_assignments

    shuffled_dict = {"grp1_data_points_per_shuffle":grp1_data_points_per_shuffle, 
                    "grp0_data_points_per_shuffle": grp0_data_points_per_shuffle, 
                    "shuffled_grp1_data":shuffled_grp1_data, 
                    "shuffled_grp0_data":shuffled_grp0_data}

    return shuffled_dict


def create_null_distribution(subject_id, 
                            power_df, 
                            sample_size, 
                            num_null_shuffles):

    subject_df = power_df[power_df["subject_id"]==subject_id]
    shuffled_subject_data = shuffle_subject_data(subject_df, null_dist_params)
    shuffled_comparator_groups = shuffle_comparator_groups(subject_df, null_dist_params)
    shuffled_dict = split_shuffled_data_by_shuffled_comp_groups(shuffled_subject_data, shuffled_comparator_groups)

    grp1_data_mean= np.sum(shuffled_dict["shuffled_grp1_data"], axis=0) / shuffled_dict["grp1_data_points_per_shuffle"]
    grp0_data_mean= np.sum(shuffled_dict["shuffled_grp0_data"], axis=0) / shuffled_dict["grp0_data_points_per_shuffle"]

    null_distribution = np.log2(grp1_sample_means / grp0_sample_means)
    null_distribution  = np.abs(null_dist)

    return null_distribution



############## ALTERNATE DISTRIBUTION ##############

def create_alt_distribution(null_distribution, 
                            effect_size):
    alt_distribution = null_distribution + np.log2(1.0 + effect_size)
    return alt_distribution



##############  POWER TO DISTINGUISH ALTERNATE FROM NULL ##############

def find_significance_threshold(alpha_sig, 
                                num_null_shuffles, 
                                null_distribution):

    num_null_significant = int(alpha_sig * num_null_shuffles)
    if num_null_significant <30:
        print ("too few shuffles for reliable estimation!")
    sorted_null_distribution = np.sort(null_distribution)
    significance_threshold = sorted_null_distirbution[-num_null_significant]
    return significance_threshold


def plot_power_of_alt_from_null(null_distribution, 
                                alt_distribution, 
                                significance_threshold,
                                power, 
                                subject_id, 
                                sample_size, 
                                effect_size):

    #plot null & alternate distributions
    sns.kdeplot(null_dist, color ="dimgray", label = "null distribution")
    plt.axvline(x =np.mean(null_dist),linestyle= "solid", color ="dimgray", ymax=.05) 
    g = sns.kdeplot(alt_dist,  color = "mediumseagreen", label = "alternative distribution")
        
    #shade area in alt distribution above significance threshold
    line = g.get_lines()[-1]
    x,y = line.get_data()
    mask = x > significance_threshold
    x,y, = x[mask], y[mask]
    g.fill_between(x, y1=y, alpha=0.5, facecolor = "mediumseagreen")
        
    #plot means of each distribution
    plt.axvline(x =np.mean(alt_dist),linestyle= "solid",  color = "mediumseagreen", ymax=.05)
    plt.axvline(x =significance_threshold, linestyle = "dashed", ymax = 1.7*(np.max(null_dist)), color = "tomato", 
        label = "sig threshold")
        
    #legends and titles
    plt.legend(loc = "upper right", bbox_to_anchor= (1.4, 1), fancybox= True)
    plt.title('subject: '+str(subject_id)+', sample size: '+str(sample_size)+', effect size: '+str(effect_size) +", power: " + str(power))
        
    plt.show()



############## MULTIPLE POWER

def multi_power(power_df, 
                sample_sizes, 
                effect_sizes, 
                num_null_shuffles = 1000, 
                alpha_sig = 0.05, 
                plot_single_power_dists = False, 
                plot_ave_heatmap = False):


    start_time = time.time()#see how long it takes to run
    counter = 1 #used to keep track of current subject out of total subjects it's currently calculating

    ##get a list of the unique subjects, since subjects can be listed more than once
    subjects_list = power_df["subject_id"].unique()
    num_subjects = len(subject_list)

    #create an empty numpy array the size of the final matrix
    #shape of array= rows number of sample trials, columns = effect sizes, z = subjects
    power_matrix = np.zeros((len(sample_sizes), len(effect_sizes), num_subjects))

    for index_subject, subject_id in enumerate(subjects_list):
        print(subject)
        print(str(counter) + " of " + str(num_subjects))
        counter = counter + 1
        
        for index_sample_size, sample_size in enumerate(sample_sizes):
            null_dist = create_null_distribution(subject_id, 
                                                power_df, 
                                                sample_size, 
                                                num_null_shuffles)

            for index_effect_size, effect_size in enumerate(effect_sizes):
                power_of_alt_from_null(null_dist, 
                                effect_size, 
                                alpha_sig, 
                                num_null_shuffles, 
                                subject, 
                                sample_size, 
                                plot_single_power_dists)
    
                power_matrix[index_sample_size, index_effect_size, index_subject] = power_of_alt_from_null(null_dist, 
                                                                                                            effect_size, 
                                                                                                            alpha_sig, 
                                                                                                            num_null_shuffles, 
                                                                                                            subject, 
                                                                                                            sample_size, 
                                                                                                            plot_single_power_dists = plot_single_power_dists)
    
    if plot_ave_heatmap ==True:
        plot_average_heatmap(power_matrix)
    
    power_dict = {"subjects":subjects, "input_data":power_df, "sample_sizes":[sample_sizes], "effect_sizes":[effect_sizes], "effect_size_type": effect_size_type, 
    "num_null_shuffles":num_null_shuffles, "alpha_sig":alpha_sig}
    
    print("creating this power matrix array took: " + str(time.time() - start_time))    
    return power_matrix, power_dict
    
def plot_average_heatmap(power_matrix):
    mean_of_subjects = power_matrix.mean(axis=2)
    mean_power_matrix = pd.DataFrame(mean_of_subjects, index =sample_sizes, columns = effect_sizes)
    mean_power_matrix = mean_power_matrix.rename_axis("sample_sizes")
    mean_power_matrix = mean_power_matrix.rename_axis("effect_sizes", axis = "columns")
    sns.heatmap(mean_power_matrix, annot= True)
    plt.show()
 