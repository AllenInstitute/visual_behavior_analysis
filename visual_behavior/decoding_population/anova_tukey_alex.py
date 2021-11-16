
anova_all = [] # each index is for 1 cre line, and shows the results of Anova (1 way) across experience levels.
tukey_all = [] # each index is for 1 cre line, and shows the results of pairwise tukey test for experience levels.

for cre in cres: 

    print(f'\n\n----------- Perfoming ANOVA/TUKEY on {cre} -----------\n')
    
    thiscre = svm_df[svm_df['cre']==cre]
    thiscre = thiscre[['cre', 'experience_levels', 'decoding_magnitude']]        
        
        
    ############ create dataframe "c", which is suitable for doing anova ############
    thiscre_anova = thiscre.rename(columns={'decoding_magnitude': 'value'})

    c = thiscre_anova.copy()
    print(c.shape)
    

    # only take valid values
    c = c[~np.isnan(c['value'])]
    print(c.shape)


    # replace Familiar, Novel 1, and Novel >1 in the df with 0, 1, and 2
    cnt = -1
    b = pd.DataFrame()
    for expl in exp_level_all:
        cnt = cnt+1
        a = c[c['experience_levels']==expl]
        a['experience_levels'] = [cnt for x in a['experience_levels']]
        b = pd.concat([b,a])
    c = b
#     c

    
    ############ ANOVA, 1-way ############
    model = ols('value ~ C(experience_levels)', data=c).fit() 
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)
    print('\n')
    
    anova_all.append(anova_table)


    ############ TUKEY HSD ############        
    v = c['value']
    f = c['experience_levels']

    MultiComp = MultiComparison(v, f)
    print(MultiComp.tukeyhsd().summary()) # Show all pair-wise comparisons


    tukey_all_ts_tsSh = MultiComp.tukeyhsd().summary()

    tukey_all.append(tukey_all_ts_tsSh) # cres x 2 (test-shfl ; test) x tukey_table (ie 4 x7)


