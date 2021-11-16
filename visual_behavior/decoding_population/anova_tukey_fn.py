# Perform Anova (1 way) and pairwise Tukey HSD

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)

def anova_tukey(svm_df, values_stat, label_stat='experience_levels'):
    
#     svm_df is a df which contains the following columns: 
#     'cre': identifies mouse cre line
#     label_stat: a column that shows the labels which categorize values_stat column
#     values_stat: a column with values which we want to do statistics on.
#     e.g. use of the code:
    '''
    label_stat='experience_levels'
    values_stat = 'decoding_magnitude'
    anova_all, tukey_all = anova_tukey(svm_df, values_stat, label_stat)
    '''

    anova_all = [] # each index is for 1 cre line, and shows the results of Anova (1 way) across experience levels.
    tukey_all = [] # each index is for 1 cre line, and shows the results of pairwise tukey test for experience levels.

    cres = svm_df['cre'].unique() 
    
    for cre in cres: 

        print(f'\n\n----------- Perfoming ANOVA/TUKEY on {cre} -----------\n')

        thiscre = svm_df[svm_df['cre']==cre]
        thiscre = thiscre[['cre', label_stat, values_stat]] # only take the relevant columns       

        print(thiscre.shape)


        ############ create dataframe "stats_df", which is suitable for doing anova ############

        # rename the column that is used for doing anova to "value"
        stats_df = thiscre.rename(columns={'decoding_magnitude': 'value'})

        # only take valid values
        stats_df = stats_df[~np.isnan(stats_df['value'])]
        print(stats_df.shape)

        # replace Familiar, Novel 1, and Novel >1 in the df with 0, 1, and 2
        cnt = -1
        dfall = pd.DataFrame()
        for expl in exp_level_all:
            cnt = cnt+1
            dfnow = stats_df[stats_df[label_stat]==expl]
            dfnow[label_stat] = [cnt for x in dfnow[label_stat]]
            dfall = pd.concat([dfall, dfnow])
        stats_df = dfall
    #     stats_df


        ############ ANOVA, 1-way : compare stats_df['value'] across stats_df['experience_levels]' ############
        # model = ols('value ~ C(experience_levels)', data=stats_df).fit()
        model = ols('value ~ C(eval(label_stat))', data=stats_df).fit() 
        anova_table = sm.stats.anova_lm(model, typ=2)
        print(anova_table)
        print('\n')

        anova_all.append(anova_table)


        ############ TUKEY HSD : compare stats_df['value'] across pairs of stats_df['experience_levels]'  ############        
        v = stats_df['value']
        f = stats_df[label_stat]

        MultiComp = MultiComparison(v, f)
        tukey_table = MultiComp.tukeyhsd().summary()    

        print(tukey_table) # Show all pair-wise comparisons

        tukey_all.append(tukey_table) # cres x 2 (test-shfl ; test) x tukey_table (ie 4 x7)


    return anova_all, tukey_all






############### Add tukey lines in the plot: if a pariwaise tukey comparison is significant add a line and an asterisk
## NOTE: the code below needs further refinement!

def add_tukey_lines_new(ax, df, tukey_all, icre):

    tukey = tukey_all[icre]
    y_new = df['test_av']+df['test_sd'] 
    mn = np.nanmin(df['shfl_av']-df['shfl_sd'])
    mx = np.nanmax(df['test_av']+df['test_sd'])                


    t = np.array(tukey.data)
    # print(t.shape)

    g1inds = np.unique(np.array([t[i][[0,1]] for i in np.arange(1,t.shape[0])]).astype(int)[:,0])
    g2inds = np.unique(np.array([t[i][[0,1]] for i in np.arange(1,t.shape[0])]).astype(int)[:,1])
    # print(g1inds, g2inds)

    cnt = 0
    cntr = 0    
    for group1_ind in g1inds: #range(3):
        for group2_ind in g2inds[g2inds > group1_ind]: #np.arange(group1_ind+1, 4):
    #         print(group1_ind, group2_ind)
            cnt = cnt+1

            if tukey.data[cnt][-1] == False:
                txtsig = "ns" 
            else:
                txtsig = "*"
                cntr = cntr+1
                r = cntr*((mx-mn)/10)

                x1, x2 = x_new[group1_ind], x_new[group2_ind]   
                y, h, col = np.max([y_new[group1_ind], y_new[group2_ind]]) + r, (mx-mn)/20, 'k'

                ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col, clip_on=True) #, transform=trans)
                ax.text((x1+x2)*.5, y+h, txtsig, ha='center', va='bottom', color=col)

                # plot the line outside, but it didnt work:
    #             https://stackoverflow.com/questions/47597534/how-to-add-horizontal-lines-as-annotations-outside-of-axes-in-matplotlib
