import pandas as pd
import numpy as np

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw hotel booking dataset by handling missing values,
    removing duplicates, and adding engineered features.
    """
    df = df.copy()

    # 1. إزالة الصفوف المتكررة
    df = df.drop_duplicates()

    # 2. معالجة القيم المفقودة
    df['company'] = df['company'].fillna(0)
    df['agent'] = df['agent'].fillna(0)
    df['country'] = df['country'].fillna('Unknown')
    df['children'] = df['children'].fillna(0)

    # 3. Feature Engineering (إنشاء ميزات جديدة)
    df['total_stays'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
    df['total_guests'] = df['adults'] + df['children'] + df['babies']

    # إزالة الحجوزات التي لا تحتوي على أي ضيوف (Data Anomaly)
    df = df[df['total_guests'] > 0]

    return df


if __name__ == "__main__":
    # المسارات مظبوطة بالنسبة لمجلد المشروع الرئيسي (D:\bosch)
    raw_path = 'data/raw/hotel_bookings.csv'
    processed_path = 'data/processed/cleaned_hotel_bookings.csv'

    print("Loading raw data...")
    raw_df = pd.read_csv(raw_path)

    print(f"Original shape: {raw_df.shape}")
    cleaned_df = clean_data(raw_df)

    print(f"Cleaned shape: {cleaned_df.shape}")

    # حفظ البيانات المنظفة
    cleaned_df.to_csv(processed_path, index=False)
    print(f"Successfully saved cleaned data to {processed_path}")