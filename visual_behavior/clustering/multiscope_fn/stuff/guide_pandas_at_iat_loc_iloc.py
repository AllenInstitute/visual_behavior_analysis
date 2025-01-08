#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 16:29:24 2019

@author: farzaneh
"""

#%% For each mouse_id, show the cre line and the number of sessions

num_sessions_per_mouse = np.array([sum(all_sess.mouse_id == all_mice_id[i]) for i in range(len(all_mice_id))]) / 8.

mouseCre_numSess = pd.DataFrame([], columns=['cre', 'num_sessions'])
for i in range(len(all_mice_id)):
    mouse_id = all_mice_id[i]
    mouseCre_numSess.at[mouse_id, ['cre','num_sessions']] = all_sess[all_sess.mouse_id == all_mice_id[i]].cre.iloc[0] , num_sessions_per_mouse[i]

#print(mouseCre_numSess)
print(mouseCre_numSess.sort_values(by='cre'))
# show total number of sessions per cre line
print(mouseCre_numSess.groupby('cre').num_sessions.sum())



#%% compare iat, at, iloc, loc
"""
mouseCre_numSess
Out[68]: 
                      cre num_sessions
392241  Slc17a7-IRES2-Cre            1
409296            at_test            3
411922               test            1
412035       Vip-IRES-Cre            1
414146       Vip-IRES-Cre            4
429956       Vip-IRES-Cre            7
431151       Vip-IRES-Cre            6
438912       Vip-IRES-Cre            5
440631       Sst-IRES-Cre            8
447934       Sst-IRES-Cre            4
448366       Sst-IRES-Cre            6
449653       Vip-IRES-Cre            6
"""

labs = mouseCre_numSess.index
cols = mouseCre_numSess.columns 

# position based : iloc and iat are similar except with iat the indexers have to be scalar.
mouseCre_numSess.iloc[2,0] # we can get the index of column "cre" from: mouseCre_numSess.columns.get_loc('cre')
mouseCre_numSess.iloc[:,0]
mouseCre_numSess.iloc[2]

mouseCre_numSess.iat[2,0]
# mouseCre_numSess.iat[2] # note this gives error.
# mouseCre_numSess.iat[2] # note this gives error.


# label based : loc and at are similar except with "at" the indexers have to be scalar.
mouseCre_numSess.loc[labs[2],'cre']
mouseCre_numSess.loc[:,'cre']
mouseCre_numSess.loc[labs[2]] # same as mouseCre_numSess.loc[labs[2],:]

mouseCre_numSess.at[labs[2], 'cre']
# mouseCre_numSess.at[:, 'cre'] # note this gives error.
# mouseCre_numSess.at[labs[2]] # note this gives error.


# assign new values: loc and at can be used, and they are exactly the same
# assign new values: change existing values
mouseCre_numSess.loc[labs[2], 'cre'] = 'test'
mouseCre_numSess.at[labs[1], 'cre'] = 'at_test'


# create new rows
mouseCre_numSess.loc['123', 'cre'] = 'newrow'
mouseCre_numSess.at['456', 'cre'] = 'newrow_at'
mouseCre_numSess.loc['789', ['cre','num_sessions']] = 'newrow', 12
mouseCre_numSess.at['89', ['cre','num_sessions']] = 'newrow', 12
mouseCre_numSess.loc['789', ['cre','num_sessions']] = 'newrow', 12
mouseCre_numSess.at['9', :] = 'newrow', 12
mouseCre_numSess.loc['10', :] = 'newrow', 120


# assign some indeces of a pandas table to another pandas table: (note we dont use loc or at here!)
all_sess_thisCre[cols0] = all_sess_now.iloc[cre_all==cre]    
