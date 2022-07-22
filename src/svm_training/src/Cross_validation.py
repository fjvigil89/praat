from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

#####  GroupKFold #############
# GroupKFold is a variation of k-fold which ensures that the same group is not represented in both
# testing and training sets. For example if the data is obtained from different subjects with
# several samples per-subject and if the model is flexible enough to learn from highly person
# specific features it could fail to generalize to new subjects. GroupKFold makes it possible to
# detect this kind of overfitting situations.
#X = [0.1, 0.2, 2.2, 2.4, 2.3, 4.55, 5.8, 8.8, 9, 10]
#y = ["a", "b", "b", "b", "c", "c", "c", "d", "d", "d"]
#groups = [1, 1, 1, 2, 2, 2, 3, 3, 3, 3]

def GroupKFold_G(X, y, groups, kfold = 5):
    gkf = GroupKFold(n_splits = kfold)
    dict_fold = {}; i = 1
    for train, test in gkf.split(X, y, groups=groups):
        dict_fold['fold' + str(i)] = {'train': train, 'test': test}
        i = i + 1
        #print("%s %s" % (train, test))
    return dict_fold

##### StratifiedGroupKFold #####
# StratifiedGroupKFold is a cross-validation scheme that combines both StratifiedKFold and GroupKFold.
# The idea is to try to preserve the distribution of classes in each split while keeping each group
# within a single split. That might be useful when you have an unbalanced dataset so that using just
# GroupKFold might produce skewed splits.
#X = list(range(18))
#y = [1] * 6 + [0] * 12
#groups = [1, 2, 3, 3, 4, 4, 1, 1, 2, 2, 3, 4, 5, 5, 5, 6, 6, 6]

def StratifiedGroupKFold_G(X, y, groups, kfold = 5):
    sgkf = StratifiedGroupKFold(n_splits = kfold)
    dict_fold = {};
    i = 0
    for train, test in sgkf.split(X, y, groups=groups):
        dict_fold['fold' + str(i)] = {'train': train, 'test': test}
        i = i + 1
        #print("%s %s" % (train, test))
    return dict_fold


if __name__ == '__main__':
    X = list(range(18))
    y = [1] * 6 + [0] * 12
    groups = [1, 2, 3, 3, 4, 4, 1, 1, 2, 2, 3, 4, 5, 5, 5, 6, 6, 6]
    GroupKFold_G(X, y, groups, 3)
    print('----------------------------------')
    StratifiedGroupKFold_G(X, y, groups, 3)