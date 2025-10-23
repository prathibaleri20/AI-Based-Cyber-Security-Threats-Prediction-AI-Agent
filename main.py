import pandas as pd
import numpy as np
import sqlite3
import time
import os
import io
import threading # Needed for running uvicorn in a separate thread if testing locally

# Import FastAPI and related components
from fastapi import FastAPI, File, UploadFile, Form, HTTPException

# Import scikit-learn components for preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Import TensorFlow and Keras components for the model
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Bidirectional, GRU, MultiHeadAttention, Dense, Dropout, Input, Concatenate, GlobalAveragePooling1D, LayerNormalization, Reshape, Permute, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Import metrics (optional, mainly for evaluation during development)
from sklearn.metrics import classification_report, roc_auc_score, f1_score

# --- Configuration ---
# Define database path (can be made configurable via environment variable)
DATABASE_PATH = os.environ.get("DATABASE_PATH", "network_flows.db")
# Define threat threshold (can be made configurable via environment variable)
THREAT_THRESHOLD = float(os.environ.get("THREAT_THRESHOLD", "0.7")) # Default threshold

# --- Data Loading and Preprocessing (Integrated) ---
# In a real deployment, you might load the preprocessed data or the raw data
# and apply the saved preprocessor. For simplicity in this combined script,
# we'll assume the original unsw_df and preprocessor/model are available
# after initial setup or loaded from saved files.

# For a deployable application, you would typically:
# 1. Load the original dataset (or a representative sample).
# 2. Fit the preprocessor and train the model ONCE during setup or use saved versions.
# 3. Save the fitted preprocessor and the trained model.
# 4. In the deployed application, load the SAVED preprocessor and model.

# --- Assuming unsw_df, preprocessor, model, and X are available from a setup script ---
# For demonstration purposes in a single file, we'll include a simplified setup.
# In a real application, this setup would run separately or be part of a build process.

# Load the dataset (assuming it's available locally or mounted)
try:
    # Assuming the parquet files are in a directory accessible at runtime
    # In a Render deployment with persistent storage, this path would be the mount path
    # If using kagglehub in deployment, you'd need to handle download there.
    # For simplicity here, assume files are present.
    parquet_files_deployment = ['UNSW_NB15_testing-set.parquet', 'UNSW_NB15_training-set.parquet']
    df_list_deployment = []
    # Assuming files are in the same directory as the script or a specified data directory
    data_dir = os.environ.get("DATA_DIR", ".") # Allow data directory to be configurable

    for file in parquet_files_deployment:
        file_path = os.path.join(data_dir, file)
        if os.path.exists(file_path):
            try:
                df_list_deployment.append(pd.read_parquet(file_path))
                print(f"Loaded {file} for setup.")
            except Exception as e:
                print(f"Error loading {file} during setup: {e}")
        else:
             print(f"Warning: Data file not found during setup: {file_path}")


    if df_list_deployment:
        unsw_df = pd.concat(df_list_deployment, ignore_index=True)
        print("Dataset loaded for preprocessing setup.")
    else:
        print("Error: No dataset files loaded for preprocessing setup.")
        unsw_df = None # Ensure unsw_df is None if loading failed

except Exception as e:
    print(f"An error occurred during dataset loading for setup: {e}")
    unsw_df = None


# Perform preprocessing setup (if dataset loaded)
preprocessor = None
model = None
X = None # Store original feature names
if unsw_df is not None:
    try:
        # Identify categorical and numerical features (same logic as before)
        categorical_features = ['proto', 'service', 'state']
        numerical_features = unsw_df.select_dtypes(include=np.number).columns.tolist()
        numerical_features = [f for f in numerical_features if f not in ['label', 'attack_cat']]

        # Handle missing values (same logic as before)
        for col in numerical_features:
            unsw_df.loc[:, col] = unsw_df.loc[:, col].fillna(0)
        for col in categorical_features:
             if pd.api.types.is_categorical_dtype(unsw_df[col]):
                 if 'Unknown' not in unsw_df[col].cat.categories:
                     unsw_df[col] = unsw_df[col].cat.add_categories('Unknown')
             unsw_df.loc[:, col] = unsw_df.loc[:, col].fillna('Unknown')


        # Create and fit the preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ],
            remainder='passthrough'
        )

        # Separate features (X) and target (y) for fitting preprocessor
        X = unsw_df.drop(['label', 'attack_cat'], axis=1)
        y = unsw_df['label'].values # Target is needed for stratify split, not for preprocessor fitting on X

        preprocessor.fit(X) # Fit the preprocessor

        print("Preprocessor fitted successfully.")

        # --- Model Building and (Simulated) Loading ---
        # In a real deployment, you would load a SAVED model.
        # For simplicity here, we'll rebuild and train a minimal model if none is loaded/saved.
        # Ideally, save and load the model and preprocessor.

        # Determine the input shape for the model based on preprocessed data shape
        # Need to transform a dummy sample to get the shape after preprocessing
        dummy_input = X.head(1)
        processed_dummy = preprocessor.transform(dummy_input)
        processed_dummy_dense = processed_dummy.toarray() if not isinstance(processed_dummy, np.ndarray) else processed_dummy
        input_shape = (processed_dummy_dense.shape[1], 1) # Shape is (number_of_features, 1) after reshaping

        # Build the model architecture (same as before)
        def build_hybrid_model(input_shape, num_classes=2):
            inputs = Input(shape=input_shape)
            cnn = Conv1D(filters=64, kernel_size=3, activation='relu', padding='causal')(inputs)
            cnn = MaxPooling1D(pool_size=2)(cnn)
            cnn = Conv1D(filters=128, kernel_size=3, activation='relu', padding='causal')(cnn)
            cnn = GlobalAveragePooling1D()(cnn)
            bigru = Bidirectional(GRU(64, return_sequences=True))(inputs)
            bigru = Bidirectional(GRU(32))(bigru)
            attention = MultiHeadAttention(num_heads=4, key_dim=32)(inputs, inputs)
            transformer = GlobalAveragePooling1D()(attention)
            combined = Concatenate()([cnn, bigru, transformer])
            dense = Dense(128, activation='relu')(combined)
            dense = Dropout(0.3)(dense)
            outputs = Dense(num_classes, activation='softmax')(dense)
            model = Model(inputs, outputs)
            return model

        # Determine number of classes (assuming binary classification 0 or 1)
        num_classes = 2 # Assuming binary classification based on 'label' column

        # Build the model
        model = build_hybrid_model(input_shape, num_classes)

        # Compile the model (using a dummy optimizer and loss if not training here)
        # In a real scenario, load a trained model with weights.
        model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        print("Model architecture built.")

        # --- Simulate loading a trained model ---
        # In a real application, load model weights from a file:
        # model.load_weights('path/to/your/model_weights.h5')
        # Or load the entire model:
        # model = tf.keras.models.load_model('path/to/your/saved_model')
        print("Simulating loading of a trained model (architecture is built).") # Acknowledge that a trained model is expected


    except Exception as e:
        print(f"An error occurred during preprocessing setup or model building: {e}")
        preprocessor = None
        model = None
        X = None # Reset X if setup failed


# --- Database Setup (Integrated) ---
# Ensure the database and table are created on startup if they don't exist.

def setup_database(db_path=DATABASE_PATH, dataframe=unsw_df):
    """Sets up the SQLite database and creates the flows table if it doesn't exist."""
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        if dataframe is None:
             print("Error: Cannot set up database schema, unsw_df is not available.")
             return # Exit if dataframe is not available

        # Get column names from the DataFrame, excluding 'label' and 'attack_cat'
        feature_columns_db = [col for col in dataframe.columns if col not in ['label', 'attack_cat']]

        # Define the SQL CREATE TABLE statement
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS flows (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
        """
        for col_name in feature_columns_db:
            dtype = dataframe[col_name].dtype
            if pd.api.types.is_float_dtype(dtype) or pd.api.types.is_datetime64_any_dtype(dtype):
                sqlite_dtype = 'REAL'
            elif pd.api.types.is_integer_dtype(dtype):
                sqlite_dtype = 'INTEGER'
            elif pd.api.types.is_object_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype):
                sqlite_dtype = 'TEXT'
            else:
                sqlite_dtype = 'TEXT'
            create_table_sql += f"    {col_name} {sqlite_dtype},\n"

        create_table_sql += "    timestamp REAL\n"
        create_table_sql = create_table_sql.rstrip(",\n") + "\n);"

        cursor.execute(create_table_sql)
        conn.commit()
        print(f"Database '{db_path}' and 'flows' table created successfully (if they did not exist).")

    except sqlite3.Error as se:
        print(f"Database setup error: {se}")
    except Exception as e:
        print(f"An unexpected error occurred during database setup: {e}")
    finally:
        if conn:
            conn.close()

# Run database setup on application startup
setup_database()


# --- Core Prediction Function ---
# This function remains similar to the one defined previously

def process_and_predict(input_df: pd.DataFrame, threshold: float = THREAT_THRESHOLD):
    """
    Preprocesses input data (single sample or batch) and performs threat prediction.

    Args:
        input_df (pd.DataFrame): DataFrame containing the input data.
                                 Expected columns should match the features used
                                 during model training (excluding label/attack_cat).
        threshold (float): Probability threshold for triggering an alert.

    Returns:
        list: A list of dictionaries, where each dictionary contains
              information about a triggered alert for the input samples
              (e.g., original index, probability, predicted_label, message).
              Returns an empty list if no alerts are triggered, or if preprocessor/model not available.
    """
    alerts = []

    if input_df.empty:
        print("Input DataFrame is empty.")
        return alerts

    # Ensure preprocessor and model are available
    if preprocessor is None or model is None or X is None:
        print("Error: Preprocessor, model, or original feature names (X) not available. Cannot process and predict.")
        return alerts

    try:
        # Apply preprocessing
        feature_columns = X.columns.tolist()
        # Ensure input_df only contains feature columns or preprocess will fail
        # Select only the columns that are in our feature list and create a copy
        input_df_features = input_df.filter(items=feature_columns).copy()

        # Handle potential missing columns in the input_df that were in X during training
        # Add missing columns with a default value (e.g., 0 for numerical, 'Unknown' for categorical)
        if 'unsw_df' in globals(): # Assuming unsw_df has the original dtypes for reference
            original_dtypes = unsw_df[feature_columns].dtypes
            for col in feature_columns:
                if col not in input_df_features.columns:
                    # print(f"Warning: Missing feature column '{col}' in input data. Adding with default value.")
                    if pd.api.types.is_numeric_dtype(original_dtypes[col]):
                         input_df_features[col] = 0
                    elif pd.api.types.is_object_dtype(original_dtypes[col]) or pd.api.types.is_categorical_dtype(original_dtypes[col]):
                         input_df_features[col] = 'Unknown'
                    else:
                         input_df_features[col] = 0
        else:
             print("Warning: Original unsw_df not found for handling missing input columns.")

        # Reorder columns to match the order used during preprocessor fitting
        # Use reindex to ensure order and handle missing if not added above
        input_df_features = input_df_features.reindex(columns=feature_columns, fill_value=0)


        processed_data = preprocessor.transform(input_df_features)

        # Convert to dense numpy array if it's a sparse matrix
        if isinstance(processed_data, np.ndarray):
             processed_data_dense = processed_data
        else:
             processed_data_dense = processed_data.toarray()

        # Reshape the processed data to match model input shape (samples, features, 1)
        processed_data_reshaped = processed_data_dense.reshape(processed_data_dense.shape[0], processed_data_dense.shape[1], 1)

        # Get prediction probabilities
        pred_probs = model.predict(processed_data_reshaped, verbose=0)

        # Process predictions for each sample
        for i in range(len(pred_probs)):
            anomaly_prob = pred_probs[i][1] # Probability for the positive class (threat)
            predicted_label = np.argmax(pred_probs[i])

            # Trigger alert if probability exceeds threshold for the positive class
            if anomaly_prob > threshold:
                # Get the original index from the input_df for reporting
                original_index = input_df.index[i]
                alerts.append({
                    'original_index': int(original_index) if isinstance(original_index, np.int64) else original_index, # Ensure index is serializable
                    'probability': float(anomaly_prob),
                    'predicted_label': int(predicted_label),
                    'message': "Threat detected"
                    })

    except Exception as e:
        print(f"Error during processing and prediction: {e}")
        # Depending on the error, you might want to log or handle differently
        pass # Continue processing other samples/batches


    return alerts


# --- FastAPI Application and Endpoints ---

app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Cyber Threat Detection Bot API"}


@app.post("/upload-csv/")
async def upload_csv(file: UploadFile = File(...)):
    """
    Reads a CSV file into a pandas DataFrame, processes it, and returns alerts.
    """
    print(f"Received file: {file.filename}, Content Type: {file.content_type}")

    # Check if preprocessor and model are loaded
    if preprocessor is None or model is None or X is None:
         raise HTTPException(status_code=500, detail="Model or preprocessor not loaded. Application is not ready.")

    content = await file.read()

    try:
        # Read the content into a pandas DataFrame
        try:
            # Use dtype=False initially to let pandas infer, or specify if known
            df = pd.read_csv(io.StringIO(content.decode('utf-8')), dtype=False)
            if df.empty:
                raise ValueError("Uploaded CSV is empty.")
        except Exception as csv_read_error:
            raise ValueError(f"Could not read CSV: {csv_read_error}")


        print(f"Successfully read CSV into DataFrame. Shape: {df.shape}")

        # Process the DataFrame and get predictions
        alerts = process_and_predict(df, threshold=THREAT_THRESHOLD)

        # Return the results in the defined JSON structure
        return {"filename": file.filename, "shape": list(df.shape), "alerts": alerts}

    except ValueError as ve:
         print(f"Value error processing CSV file: {ve}")
         raise HTTPException(status_code=400, detail=f"Value error processing CSV file '{file.filename}': {ve}")
    except Exception as e:
        print(f"Error processing CSV file: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing CSV file '{file.filename}': {e}")


@app.post("/predict-text/")
async def predict_text(text_input: str = Form(...)):
    """
    Receives text input (expected to be a single row of comma-separated values
    matching the feature columns), parses it, processes it, and returns alerts.
    """
    print(f"Received text input: {text_input}")

    # Check if preprocessor and model are loaded
    if preprocessor is None or model is None or X is None:
         raise HTTPException(status_code=500, detail="Model or preprocessor not loaded. Application is not ready.")

    feature_columns = X.columns.tolist()

    try:
        # Parse the text input into a pandas DataFrame
        try:
             input_io = io.StringIO(text_input + "\n")
             # Use dtype=False initially to let pandas infer
             # Provide column names to ensure they match feature_columns
             input_df = pd.read_csv(input_io, header=None, names=feature_columns, dtype=False)
             if input_df.shape[1] != len(feature_columns):
                  raise ValueError(f"Number of values in text input ({input_df.shape[1]}) does not match expected features ({len(feature_columns)}).")
             if input_df.empty:
                  raise ValueError("Parsed text input resulted in an empty DataFrame.")

             # Convert dtypes to match the original DataFrame's feature columns
             if 'unsw_df' in globals(): # Use unsw_df for dtype reference
                  original_dtypes = unsw_df[feature_columns].dtypes
                  for col in feature_columns:
                      if col in original_dtypes:
                           try:
                                if pd.api.types.is_numeric_dtype(original_dtypes[col]):
                                     input_df.loc[:, col] = pd.to_numeric(input_df.loc[:, col], errors='coerce')
                                     input_df.loc[:, col] = input_df.loc[:, col].fillna(0)
                                else:
                                     input_df.loc[:, col] = input_df.loc[:, col].astype(original_dtypes[col])
                           except Exception as e:
                                print(f"Warning: Could not cast column '{col}' to original dtype from text input: {e}. Value was: {input_df.loc[0, col]}")
                                if pd.api.types.is_numeric_dtype(original_dtypes[col]):
                                    input_df.loc[:, col] = 0
                                else:
                                    input_df.loc[:, col] = 'Unknown'
                      else:
                           print(f"Warning: Column '{col}' not found in original unsw_df dtypes for text input casting.")
             else:
                  print("Warning: Original unsw_df not found for dtype casting of text input.")


        except Exception as parse_error:
             raise ValueError(f"Could not parse text input: {parse_error}")


        print(f"Successfully parsed text input into DataFrame. Shape: {input_df.shape}")

        # Process the DataFrame and get predictions
        alerts = process_and_predict(input_df, threshold=THREAT_THRESHOLD)


        # Return the results in the defined JSON structure
        return {"received_text": text_input, "alerts": alerts}

    except ValueError as ve:
         print(f"Value error processing text input: {ve}")
         raise HTTPException(status_code=400, detail=f"Value error processing text input: {ve}. Please check the format.")
    except Exception as e:
        print(f"Error processing text input: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing text input: {e}")


@app.post("/simulate-live/")
async def simulate_live_detection(batch_size: int = 50):
    """
    Simulates real-time threat detection by reading data from a SQLite database
    in batches, preprocessing it, and using the trained model for predictions.

    Args:
        batch_size (int): Number of rows to read from the database at a time.

    Returns:
        dict: A dictionary containing the total number of rows processed
              and a list of triggered alerts.
    """
    print(f"Starting real-time threat detection from database '{DATABASE_PATH}' with batch size {batch_size}...")
    alerts = []
    conn = None

    # Check if preprocessor and model are loaded
    if preprocessor is None or model is None or X is None:
         raise HTTPException(status_code=500, detail="Model or preprocessor not loaded. Application is not ready.")


    feature_column_names = X.columns.tolist()

    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        # Get the total number of rows in the flows table
        cursor.execute("SELECT COUNT(*) FROM flows")
        total_rows = cursor.fetchone()[0]
        print(f"Total rows in the database: {total_rows}")

        offset = 0
        rows_processed = 0

        while offset < total_rows:
            # Select a batch of data, excluding 'id' and 'timestamp'
            select_sql = f"SELECT {', '.join(feature_column_names)} FROM flows LIMIT ? OFFSET ?"
            cursor.execute(select_sql, (batch_size, offset))
            batch_data = cursor.fetchall()

            if not batch_data:
                break # No more data to read

            # Convert batch data to DataFrame for preprocessing
            # Let pandas infer dtypes initially, then cast later if needed for consistency
            batch_df = pd.DataFrame(batch_data, columns=feature_column_names, dtype=False)

            # Convert dtypes to match the original DataFrame's feature columns
            if 'unsw_df' in globals(): # Use unsw_df for dtype reference
                 original_dtypes = unsw_df[feature_column_names].dtypes
                 for col in feature_column_names:
                     if col in original_dtypes:
                          try:
                               if pd.api.types.is_numeric_dtype(original_dtypes[col]):
                                    batch_df.loc[:, col] = pd.to_numeric(batch_df.loc[:, col], errors='coerce')
                                    batch_df.loc[:, col] = batch_df.loc[:, col].fillna(0)
                               else:
                                    batch_df.loc[:, col] = batch_df.loc[:, col].astype(original_dtypes[col])
                          except Exception as e:
                               print(f"Warning: Could not cast column '{col}' to original dtype from DB: {e}")
                     else:
                          print(f"Warning: Column '{col}' not found in original unsw_df dtypes for DB casting.")
            else:
                 print("Warning: Original unsw_df not found for dtype casting of DB data.")


            # Use the process_and_predict function
            # The process_and_predict function expects the original index for reporting.
            # We can add a temporary 'original_index' column to the batch_df
            # corresponding to the row's offset in the database read.
            batch_df['original_index'] = range(offset, offset + len(batch_df))


            # Use the configured THREAT_THRESHOLD
            batch_alerts = process_and_predict(batch_df, threshold=THREAT_THRESHOLD)

            # The process_and_predict function returns alerts with 'original_index'.
            # For the /simulate-live/ endpoint, we want to report 'db_row_index'.
            for alert in batch_alerts:
                if 'original_index' in alert:
                    alert['db_row_index'] = alert.pop('original_index') # Rename key
                alerts.append(alert) # Append the modified alert


            # Update rows processed count
            rows_processed += len(batch_df)

            # Move to the next batch
            offset += batch_size

            # Simulate time delay between batches (optional, for realistic simulation)
            # time.sleep(0.01)


    except sqlite3.Error as se:
        print(f"Database error: {se}")
        raise HTTPException(status_code=500, detail=f"Database error: {se}")
    except Exception as e:
        print(f"An unexpected error occurred during simulation: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")

    print("\nReal-time simulation from database finished.")
    print(f"Total alerts triggered: {len(alerts)}")

    # Return the results in the defined JSON structure
    return {"total_rows_processed": rows_processed, "alerts": alerts}

# --- Voice Input Handling (Placeholder) ---
# This is a placeholder. Actual implementation would involve STT and parsing.
# You would need to install libraries like `SpeechRecognition` or `openai-whisper`.
# A POST endpoint would receive audio data, process it to text, parse the text
# into the required feature format, and then call process_and_predict.

@app.post("/predict-voice/")
async def predict_voice(audio_file: UploadFile = File(...)):
    """
    Receives audio file, converts to text (placeholder),
    parses text (placeholder), processes, and returns alerts.
    """
    print(f"Received audio file: {audio_file.filename}, Content Type: {audio_file.content_type}")

    # Placeholder: Save audio file temporarily
    # audio_content = await audio_file.read()
    # with open(audio_file.filename, "wb") as f:
    #     f.write(audio_content)
    # print(f"Saved temporary audio file: {audio_file.filename}")

    # Placeholder for Speech-to-Text conversion
    transcribed_text = "This is a placeholder for transcribed text." # Replace with actual STT logic
    print(f"Simulated transcribed text: {transcribed_text}")

    # Placeholder for Text Parsing into features
    # This would involve using NLP or rule-based parsing on `transcribed_text`
    # to extract values for network flow features (dur, proto, service, etc.)
    # and create a pandas DataFrame row.
    # For now, we'll simulate creating a dummy DataFrame row.
    if 'X' not in globals():
         raise HTTPException(status_code=500, detail="Original feature column names (X) not found. Cannot process voice input (parsing placeholder).")

    feature_columns = X.columns.tolist()
    # Create a dummy DataFrame row with default values for all features
    dummy_data = {col: 0 for col in feature_columns} # Use 0 or 'Unknown' based on dtype
    # A more sophisticated parser would populate this dictionary from `transcribed_text`

    # Attempt to use original dtypes for dummy data creation if unsw_df is available
    if 'unsw_df' in globals():
         original_dtypes = unsw_df[feature_columns].dtypes
         dummy_data = {}
         for col in feature_columns:
              if col in original_dtypes:
                   if pd.api.types.is_numeric_dtype(original_dtypes[col]):
                        dummy_data[col] = 0.0 # Use float default for numerical
                   elif pd.api.types.is_object_dtype(original_dtypes[col]) or pd.api.types.is_categorical_dtype(original_dtypes[col]):
                        dummy_data[col] = 'Unknown' # Use string default for categorical/object
                   else:
                        dummy_data[col] = None # Default to None for others
              else:
                   dummy_data[col] = None # Default to None if dtype not found


    input_df = pd.DataFrame([dummy_data])
    print(f"Simulated parsed input DataFrame. Shape: {input_df.shape}")


    # Process the DataFrame and get predictions (using the simulated input_df)
    alerts = process_and_predict(input_df, threshold=THREAT_THRESHOLD)


    # Placeholder: Clean up temporary audio file
    # if os.path.exists(audio_file.filename):
    #     os.remove(audio_file.filename)


    # Return the results
    return {"filename": audio_file.filename, "transcribed_text": transcribed_text, "alerts": alerts}

# --- Main execution block for local testing ---
# To run this application locally using uvicorn directly:
# if __name__ == "__main__":
#     # Ensure database is set up on startup
#     setup_database()
#     uvicorn.run(app, host="0.0.0.0", port=8000)

# For Colab, you might use ngrok and threading as shown in testing cells.
# Ensure setup_database() is called before starting the server if running in Colab.
setup_database() # Ensure database is set up when this cell is run in Colab
