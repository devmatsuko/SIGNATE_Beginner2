import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import pandas_profiling as pdp

# CSVファイルの読み込み
test = pd.read_csv('../data/test.csv')
train = pd.read_csv('../data/train.csv')

# pandas profiling
# profile = pdp.ProfileReport(traiｎ)
# profile.to_file(output_file="../number.html")

# 目的変数の分離(x:目的変数以外、y:目的変数)
train_x = train.drop(['disease'], axis=1)
train_y = train['disease']

# テストデータのバックアップ
test_x = test.copy()

# データクレンジング
# 学習データとテストデータを結合
data = pd.concat([train_x, test_x], sort=False, ignore_index=True)

# データの標準化
# 年代列の追加
train_age = pd.Series(
    data=data['Age'] - np.mod(data['Age'], 10), name='Generation')
data = pd.concat([data, train_age], axis=1)

# 検査項目列のみ抽出（年代別）
data_norm_gene = pd.DataFrame([], index=data.index, columns=[
                              'T_Bil_norm_gene', 'D_Bil_norm_gene', 'ALP_norm_gene', 'ALT_GPT_norm_gene', 'AST_GOT_norm_gene'])
data_norm_gene = data_norm_gene.astype('float')
# 年代別に標準化
for idx in data.index:
    data_norm_gene['T_Bil_norm_gene'][idx] = (data['T_Bil'][idx] - data.groupby('Generation').mean(
    )['T_Bil'][data['Generation'][idx]])/data.groupby('Generation').var()['T_Bil'][data['Generation'][idx]]
for idx in data.index:
    data_norm_gene['D_Bil_norm_gene'][idx] = (data['D_Bil'][idx] - data.groupby('Generation').mean(
    )['D_Bil'][data['Generation'][idx]])/data.groupby('Generation').var()['D_Bil'][data['Generation'][idx]]
for idx in data.index:
    data_norm_gene['ALP_norm_gene'][idx] = (data['ALP'][idx] - data.groupby('Generation').mean(
    )['ALP'][data['Generation'][idx]])/data.groupby('Generation').var()['ALP'][data['Generation'][idx]]
for idx in data.index:
    data_norm_gene['ALT_GPT_norm_gene'][idx] = (data['ALT_GPT'][idx] - data.groupby('Generation').mean(
    )['ALT_GPT'][data['Generation'][idx]])/data.groupby('Generation').var()['ALT_GPT'][data['Generation'][idx]]
for idx in data.index:
    data_norm_gene['AST_GOT_norm_gene'][idx] = (data['AST_GOT'][idx] - data.groupby('Generation').mean(
    )['AST_GOT'][data['Generation'][idx]])/data.groupby('Generation').var()['AST_GOT'][data['Generation'][idx]]

# 検査項目列のみ抽出（性別別）
data_norm_gend = pd.DataFrame([], index=data.index, columns=[
                              'T_Bil_norm_gend', 'D_Bil_norm_gend', 'ALT_GPT_norm_gend', 'AST_GOT_norm_gend'])
data_norm_gend = data_norm_gend.astype('float')
for idx in data.index:
    data_norm_gend['T_Bil_norm_gend'][idx] = (data['T_Bil'][idx] - data.groupby('Gender').mean(
    )['T_Bil'][data['Gender'][idx]])/data.groupby('Gender').var()['T_Bil'][data['Gender'][idx]]
for idx in data.index:
    data_norm_gend['D_Bil_norm_gend'][idx] = (data['D_Bil'][idx] - data.groupby('Gender').mean(
    )['D_Bil'][data['Gender'][idx]])/data.groupby('Gender').var()['D_Bil'][data['Gender'][idx]]
for idx in data.index:
    data_norm_gend['ALT_GPT_norm_gend'][idx] = (data['ALT_GPT'][idx] - data.groupby('Gender').mean(
    )['ALT_GPT'][data['Gender'][idx]])/data.groupby('Gender').var()['ALT_GPT'][data['Gender'][idx]]
for idx in data.index:
    data_norm_gend['AST_GOT_norm_gend'][idx] = (data['AST_GOT'][idx] - data.groupby('Gender').mean(
    )['AST_GOT'][data['Gender'][idx]])/data.groupby('Gender').var()['AST_GOT'][data['Gender'][idx]]
data = pd.concat([data, data_norm_gene], axis=1)
data = pd.concat([data, data_norm_gend], axis=1)

# print(data.head(10))
# 欠損値補完
#　カテゴリ変数をループしてlabel encoding
for c in ['Gender']:
    le = LabelEncoder()
    le.fit(data[c])
    data[c] = le.transform(data[c])


# trainとtestの分離
train_x = data[:len(train_x)]
test_x = data[len(train_x):].reset_index(drop=True)

# 年代（Generaiton）の削除
train_x = train_x.drop(['Generation'], axis=1)
test_x = test_x.drop(['Generation'], axis=1)


# クロスバリデーション
scores = []
params = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'eta': 0.1,
    'gamma': 0.0,
    'alpha': 0.0,
    'lambda': 1.0,
    'min_child_weight': 1,
    'max_depth': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 71,
}
num_round = 500  # 決定木の本数？
va_pred_temp = [0] * len(test_x.index)
dtest = xgb.DMatrix(test_x)
kf = KFold(n_splits=4, shuffle=True, random_state=71)
for tr_idx, va_idx in kf.split(train_x):
    tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]
    # 特徴量と目的変数をxgboostのデータ構造に変換する
    dtrain = xgb.DMatrix(tr_x, label=tr_y)
    dvaild = xgb.DMatrix(va_x, label=va_y)
    watchlist = [(dtrain, 'train'), (dvaild, 'eval')]
    model = xgb.train(params, dtrain, num_round,
                      evals=watchlist, early_stopping_rounds=25)
    va_pred = model.predict(dvaild, ntree_limit=model.best_ntree_limit)
    # va_pred_label = np.where(va_pred > 0.5, 1, 0)
    # score = roc_auc_score(va_y, va_pred_label)
    score = roc_auc_score(va_y, va_pred)
    scores.append(score)
    va_pred_temp = va_pred_temp + \
        model.predict(dtest, ntree_limit=model.best_ntree_limit)
print(scores)

# モデル学習
#dtrain = xgb.DMatrix(train_x, label=train_y)
#model = xgb.train(params, dtrain, num_round)

# テストデータでの予測
# dtest = xgb.DMatrix(test_x)
# pred = model.predict(dtest)
# pred = np.wehre(va_pred_temp/4 > 0.5, 1, 0)
pred = va_pred_temp/4
print(pred)

# CSV出力
sub = pd.DataFrame(index=test.index, columns=['disease'])
sub['disease'] = list(pred)
sub.to_csv('../submission.csv', header=False)
