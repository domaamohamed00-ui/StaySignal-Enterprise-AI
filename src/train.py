import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# الموديلات الـ 6
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier

def train_and_export_all():
    # 1. تحميل البيانات المنظفة
    data_path = 'data/processed/cleaned_hotel_bookings.csv'
    print(f"Loading processed data from {data_path}...")
    df = pd.read_csv(data_path)

    target = 'is_canceled'
    drop_cols = [target, 'reservation_status', 'reservation_status_date']
    
    X = df.drop(columns=drop_cols)
    y = df[target]

    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # 2. تقسيم البيانات (Train / Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. الـ Preprocessor المحسن مع StandardScaler
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )

    # حساب نسبة الفئات لـ XGBoost لرفع الـ Recall
    ratio = float(y_train.value_counts()[0] / y_train.value_counts()[1])

    # 4. تعريف الموديلات الـ 6
    models = {
        'Tuned XGBoost': XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=10,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=ratio,
            random_state=42,
            eval_metric='logloss'
        ),
        'Tuned Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=150, max_depth=6, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=15, random_state=42),
        'AdaBoost': AdaBoostClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }

    results = []
    best_score = 0.0
    best_pipeline = None
    best_model_name = ""
    best_metrics = {}

    print("\nStarting evaluation for 6 Models...\n" + "="*65)

    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        # تدريب
        pipeline.fit(X_train, y_train)
        
        # تنبؤ
        y_pred = pipeline.predict(X_test)
        
        if hasattr(pipeline, "predict_proba"):
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
        else:
            auc = 0.0

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        model_res = {
            'Model': name,
            'Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1-Score': round(f1, 4),
            'ROC-AUC': round(auc, 4)
        }
        results.append(model_res)

        if auc > best_score:
            best_score = auc
            best_pipeline = pipeline
            best_model_name = name
            best_metrics = model_res

    # عرض جدول المقارنة
    results_df = pd.DataFrame(results).sort_values(by='ROC-AUC', ascending=False)
    print("\nOptimized Model Comparison Results:")
    print("="*65)
    print(results_df.to_string(index=False))
    print("="*65)
    print(f"\nBest Model: {best_model_name} with ROC-AUC = {best_score:.4f}")

    # 5. حفظ كل الـ Artifacts في مجلد artifacts/
    os.makedirs('artifacts', exist_ok=True)
    
    # حفظ الموديل الفائز
    joblib.dump(best_pipeline, 'artifacts/model.joblib')

    # حفظ metrics.json للموديل الفائز
    with open('artifacts/metrics.json', 'w') as f:
        json.dump(best_metrics, f, indent=4)

    # حفظ insights.json
    insights = {
        "total_records": len(df),
        "cancellation_rate_pct": round(float(df[target].mean() * 100), 2),
        "top_country": str(df['country'].mode()[0]),
        "avg_lead_time_days": round(float(df['lead_time'].mean()), 2)
    }
    with open('artifacts/insights.json', 'w') as f:
        json.dump(insights, f, indent=4)

    print("\nSuccessfully saved best pipeline, metrics, and insights to artifacts/")

if __name__ == "__main__":
    train_and_export_all()