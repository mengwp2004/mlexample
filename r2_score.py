from sklearn.metrics import r2_score

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

score =r2_score(y_true, y_pred)  

print(score)


y_true = [1,2,3]

y_pred = [3,2,1]

score = r2_score(y_true, y_pred)

print(score)


