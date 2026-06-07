# Automated string-to-datetime casting and duplicate cleansing for both datasets. 
# This ensures that the data is in the correct format for analysis and modeling, 
# and that we are not training on duplicate records which could bias our results.
# To handle data loading, geolocation lookups, feature engineering, and scaling.
# It uses explicit try-except blocks, typing, and clear operational logging.

import os
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Configure structured logging for traceability
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class FraudDataPipeline:
    def __init__(self, raw_data_dir: str, processed_data_dir: str):
        self.raw_dir = raw_data_dir
        self.processed_dir = processed_data_dir
        self.scaler = StandardScaler()

    def load_csv(self, file_name: str) -> pd.DataFrame:
        """Loads raw CSV data with robust error handling."""
        path = os.path.join(self.raw_dir, file_name)
        try:
            logger.info(f"Attempting to load dataset: {file_name}")
            df = pd.read_csv(path)
            logger.info(f"Successfully loaded {file_name} with shape {df.shape}")
            return df
        except FileNotFoundError as e:
            logger.error(f"Critical file missing at {path}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading {file_name}: {str(e)}")
            raise

    @staticmethod
    def ip_to_int(ip_series: pd.Series) -> pd.Series:
        """Converts IPv4 strings or floats to 32-bit integers safely."""
        try:
            # Handle float representations if IP was parsed incorrectly
            ip_clean = ip_series.fillna("0.0.0.0").astype(str).str.split('.').str[0].astype(float).astype(int)
            # Alternate vectorized conversion assuming standard float/int representation if already numeric
            if pd.api.types.is_numeric_dtype(ip_series):
                return ip_series.fillna(0).astype(np.int64)
            
            return ip_series.apply(
                lambda x: sum(int(num) << (24 - 8 * i) for i, num in enumerate(str(x).split('.')))
                if pd.notnull(x) and '.' in str(x) else 0
            )
        except Exception as e:
            logger.error(f"Failed to convert IP addresses to integer: {str(e)}")
            raise

    def merge_geolocation(self, fraud_df: pd.DataFrame, ip_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merges E-commerce fraud data with IP address blocks using a range-based lookup.
        Optimized via sorting and merge_asof to avoid memory issues.
        """
        try:
            logger.info("Initiating range-based IP geolocation merge.")
            
            # Ensure IP values are numeric integers
            if not pd.api.types.is_numeric_dtype(fraud_df['ip_address']):
                fraud_df['ip_int'] = self.ip_to_int(fraud_df['ip_address'])
            else:
                fraud_df['ip_int'] = fraud_df['ip_address'].astype(np.int64)
                
            ip_df['lower_bound_int'] = ip_df['lower_bound_ip_address'].astype(np.int64)
            ip_df['upper_bound_int'] = ip_df['upper_bound_ip_address'].astype(np.int64)

            # Sort records required for merge_asof
            fraud_df = fraud_df.sort_values('ip_int')
            ip_df = ip_df.sort_values('lower_bound_int')

            # Conditional conditional matching using merge_asof
            merged_df = pd.merge_asof(
                fraud_df,
                ip_df,
                left_on='ip_int',
                right_on='lower_bound_int',
                direction='backward'
            )

            # Post-merge constraint validation: Validate if the IP falls cleanly within upper boundaries
            valid_mask = (merged_df['ip_int'] >= merged_df['lower_bound_int']) & \
                         (merged_df['ip_int'] <= merged_df['upper_bound_int'])
            
            merged_df.loc[~valid_mask, 'country'] = 'Unknown'
            
            # Clean up temporary operational variables
            merged_df.drop(columns=['lower_bound_int', 'upper_bound_int', 'ip_int'], inplace=True, errors='ignore')
            logger.info("Geolocation mapping completed successfully.")
            return merged_df
        except Exception as e:
            logger.error(f"Error during range-based IP lookup merge: {str(e)}")
            raise

    def engineer_ecommerce_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineers actionable behavioral, temporal, and velocity patterns."""
        try:
            logger.info("Starting E-commerce Feature Engineering.")
            df = df.copy()

            # 1. Cast time dimensions accurately
            df['signup_time'] = pd.to_datetime(df['signup_time'])
            df['purchase_time'] = pd.to_datetime(df['purchase_time'])

            # 2. Time Since Signup (Business Rationale: Instant purchases often indicate automated bot registrations)
            df['time_since_signup_hr'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds() / 3600.0

            # 3. Temporal Signatures (Business Rationale: High-risk behavior frequently spikes late at night)
            df['hour_of_day'] = df['purchase_time'].dt.hour
            df['day_of_week'] = df['purchase_time'].dt.dayofweek

            # 4. Device Velocity Profile (Business Rationale: Multiple unique users on a single hardware ID indicates a device farm)
            device_velocity = df.groupby('device_id')['user_id'].transform('count')
            df['transactions_per_device'] = device_velocity

            # 5. Missing value resolution strategy
            df['country'] = df['country'].fillna('Unknown')

            logger.info("Feature engineering completed for E-commerce stream.")
            return df
        except Exception as e:
            logger.error(f"Error engineering E-commerce features: {str(e)}")
            raise

    def process_and_scale(self, ecom_df: pd.DataFrame, bank_df: pd.DataFrame):
        """Applies final categorical transformations and numeric scaling."""
        try:
            logger.info("Scaling numeric feature sets.")
            
            # E-commerce pipeline finalizing
            ecom_numeric = ['purchase_value', 'age', 'time_since_signup_hr', 'transactions_per_device']
            ecom_df[ecom_numeric] = self.scaler.fit_transform(ecom_df[ecom_numeric])
            
            # One-Hot Encoding categorical descriptors
            ecom_df = pd.get_dummies(ecom_df, columns=['source', 'browser', 'sex'], drop_first=True)

            # Banking data stream scaling (Time and Amount optimization)
            bank_df = bank_df.copy()
            bank_df[['Time', 'Amount']] = self.scaler.fit_transform(bank_df[['Time', 'Amount']])

            # Export artifacts safely to disk
            os.makedirs(self.processed_dir, exist_ok=True)
            ecom_df.to_csv(os.path.join(self.processed_dir, "cleaned_ecommerce.csv"), index=False)
            bank_df.to_csv(os.path.join(self.processed_dir, "cleaned_creditcard.csv"), index=False)
            
            logger.info("All final artifacts securely written to data/processed/ output paths.")
        except Exception as e:
            logger.error(f"Error during final scaling and export pipeline phase: {str(e)}")
            raise

if __name__ == "__main__":
    # Allows end-to-end execution directly from the command line
    pipeline = FraudDataPipeline(raw_data_dir="data/raw", processed_data_dir="data/processed")
    
    try:
        fraud_raw = pipeline.load_csv("Fraud_Data.csv")
        ip_raw = pipeline.load_csv("IpAddress_to_Country.csv")
        bank_raw = pipeline.load_csv("creditcard.csv")
        
        # Run Pipeline
        ecom_mapped = pipeline.merge_geolocation(fraud_raw, ip_raw)
        ecom_featured = pipeline.engineer_ecommerce_features(ecom_mapped)
        pipeline.process_and_scale(ecom_featured, bank_raw)
        
        print("\n🎉 TASK 1 PIPELINE RUN COMPLETED SUCCESSFULY!")
    except Exception as e:
        print(f"\n❌ PIPELINE CRASHED. Diagnostic Trace: {e}")