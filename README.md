## ChainLearn Nexus — Federated Healthcare Prediction Platform

A privacy-preserving healthcare FL platform combining a Streamlit client app with MFA-secured authentication and a Flask + Flower server for federated learning across three disease models: Diabetes, Heart Disease, and Parkinson's.

### Features
- **Federated Learning (Flower)**: Separate gRPC servers per model (Diabetes, Heart, Parkinson's) with SVM pipelines and SMOTE balancing.
- **API Server (Flask)**: Endpoints to fetch global models and upload client records, plus health/status endpoints.
- **Client App (Streamlit)**: Role-based app (Doctor, Receptionist, Patient) for predictions, report generation (PDF), data syncing to FL server, and viewing SQL records.
- **Security**:
  - Passwords: SHA-256 + bcrypt layered hashing with per-user salt.
  - **MFA (TOTP)**: QR-code onboarding; codes validated on login via `pyotp`.
  - Record integrity: SHA-256 per-field hashing and combined bcrypt integrity hash for stored test results.
  - **Zero-Day Threat (ZDT)** protections on the server: basic rate-limiting, temporary IP blocking, and attack logs.

## Project Structure
```
.
├── client5_mfa.py                      # Streamlit client app (MFA, predictions, PDF, data sync)
├── server5_mfa.py                      # Flask API + Flower federated servers (3 models)
├── diabetes.csv                        # Sample/reference dataset
├── heart.csv                           # Sample/reference dataset
├── parkinsons.csv                      # Sample/reference dataset
├── global_diabetes_svm_model.pkl       # Initial/global model (optional seed)
├── global_heart_svm_model.pkl          # Initial/global model (optional seed)
└── global_parkinsons_svm_model.pkl     # Initial/global model (optional seed)
```

## Requirements
- Python 3.9+ (3.10 recommended)
- Pip and virtual environment tools

Install dependencies:
```bash
pip install streamlit streamlit-option-menu flask flask-cors flwr scikit-learn imbalanced-learn pandas numpy bcrypt pyotp qrcode[pil] fpdf requests
```

## Configuration
- Server environment variables (optional, with defaults):
  - `SERVER_HOST` (default `0.0.0.0`)
  - `FLASK_PORT` (default `5000`)
  - `FLOWER_DIABETES_GRPC_PORT` (default `8080`)
  - `FLOWER_HEART_GRPC_PORT` (default `8081`)
  - `FLOWER_PARKINSONS_GRPC_PORT` (default `8082`)
  - `DIABETES_GLOBAL_MODEL_FILE` (default `global_diabetes_svm_model.pkl`)
  - `HEART_GLOBAL_MODEL_FILE` (default `global_heart_svm_model.pkl`)
  - `PARKINSONS_GLOBAL_MODEL_FILE` (default `global_parkinsons_svm_model.pkl`)
  - `CLIENT_*_DATA_DIR` and `MASTER_*_AGGREGATED_DATA_FILE` paths for each model

- Client configuration:
  - Update `FL_SERVER_URL` in `client.py` to point to your server, e.g. `http://<SERVER_IP>:5000`.
  - Ensure your firewall allows inbound connections on `FLASK_PORT` and Flower gRPC ports (8080/8081/8082, or your custom values).

## Running the Server
1. (Optional) Set environment variables.
2. Start the server:
```bash
python server.py
```
- This launches:
  - Flask API on `http://SERVER_HOST:FLASK_PORT`
  - Flower gRPC servers on `FLOWER_DIABETES_GRPC_PORT`, `FLOWER_HEART_GRPC_PORT`, `FLOWER_PARKINSONS_GRPC_PORT`
- Server creates/uses per-client CSVs in `client_diabetes_data/`, `client_heart_data/`, `client_parkinsons_data/` and master aggregate CSVs.

## Running the Client App
1. Set `FL_SERVER_URL` inside `client.py` to match your server.
2. Launch the Streamlit app:
```bash
streamlit run client.py
```
3. Use the sidebar navigation to access:
   - Diabetes / Heart Disease / Parkinson's predictions
   - SQL Records
   - About / Developed By
   - Federated Learning Sync actions to send your latest local data to the FL server

### Accounts and MFA
- Register a new user in the app (tabs: Register/Login).
- After successful registration, the app shows a QR code for MFA. Scan it with an authenticator app (e.g., Google Authenticator, Authy).
- Login requires the TOTP code. Ensure the device time is synced.

### Predictions and Reports
- Provide input features in the disease-specific page and generate predictions with the latest FL global model (fetched from the server).
- The app can generate PDF reports via `FPDF` and stores hashed records in an SQLite database `healthcare.db`.

## API Reference (Flask)
Base URL: `http://<SERVER_HOST>:<FLASK_PORT>`

- Models (download latest global model pickle):
  - `GET /get_diabetes_model`
  - `GET /get_heart_model`
  - `GET /get_parkinsons_model`

- Model info/metrics:
  - `GET /get_diabetes_model_info`
  - `GET /get_heart_model_info`
  - `GET /get_parkinsons_model_info`

- Upload client data for federated learning:
  - `POST /upload_diabetes_client_data`
  - `POST /upload_heart_client_data`
  - `POST /upload_parkinsons_client_data`

- Status endpoints:
  - `GET /zdt_status` — zero-day threat counters, blocked IPs, and recent logs
  - `GET /client_status` — liveness and data sizes for all connected clients
  - `GET /fl_status` — federated training status per model

### Upload Payloads
- Common envelope:
```json
{
  "client_id": "<string>",
  "records": [ { /* feature dict per record, see below */ } ]
}
```
- Diabetes features (`Outcome` is 0/1):
```json
{
  "Pregnancies": 6,
  "Glucose": 148,
  "BloodPressure": 72,
  "SkinThickness": 35,
  "Insulin": 0,
  "BMI": 33.6,
  "DiabetesPedigreeFunction": 0.627,
  "Age": 50,
  "Outcome": 1
}
```
- Heart Disease features (`Outcome` is 0/1; `target` will be mapped if provided):
```json
{
  "age": 63,
  "sex": 1,
  "cp": 3,
  "trestbps": 145,
  "chol": 233,
  "fbs": 1,
  "restecg": 0,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 0,
  "ca": 0,
  "thal": 1,
  "Outcome": 1
}
```
- Parkinson's features (`Outcome` is 0/1):
```json
{
  "fo": 119.992,
  "fhi": 157.302,
  "flo": 74.997,
  "Jitter_percent": 0.00784,
  "Jitter_Abs": 0.00007,
  "RAP": 0.00370,
  "PPQ": 0.00554,
  "DDP": 0.01110,
  "Shimmer": 0.04374,
  "Shimmer_dB": 0.426,
  "APQ3": 0.02182,
  "APQ5": 0.03130,
  "APQ": 0.02971,
  "DDA": 0.06545,
  "NHR": 0.02211,
  "HNR": 21.033,
  "RPDE": 0.414783,
  "DFA": 0.815285,
  "spread1": -4.81303,
  "spread2": 0.266482,
  "D2": 2.301442,
  "PPE": 0.284654,
  "Outcome": 1
}
```

### Notes on Server Validation and Protection
- Requests must be JSON; malformed or non-numeric values are rejected.
- Basic rate limiting per client IP; excessive requests cause temporary IP blocking.
- The server deduplicates records per client and master files using row-content hashing.

## Federated Learning Details
- Each model uses an `ImbPipeline` with `StandardScaler`, `SMOTE`, and `SVC(kernel='linear', probability=True)`.
- FL strategy is a customized FedAvg per model.
- Global models are persisted to `global_*_svm_model.pkl` files.

## Data and Storage
- Client CSVs are maintained per `client_id` under:
  - `client_diabetes_data/`, `client_heart_data/`, `client_parkinsons_data/`
- Aggregated master CSVs: `server_*_master.csv` files per model.
- Application database: `healthcare.db` (SQLite) with tables for users, test results, doctors, and appointments.

## Troubleshooting
- **Client cannot load model**: Ensure the server is running, ports are open, and the global model file exists. Wait for at least one FL round or seed the `global_*` model files.
- **MFA codes fail**: Check time synchronization on the authenticator device and server machine.
- **429 or blocked**: You may have hit the rate limit; wait for the block to expire or reduce request rate.
- **CORS issues**: The server allows `http://127.0.0.1:5001` by default. Adjust in `server5_mfa.py` if your client origin differs.

## License
Add your license of choice (e.g., MIT) here.

## Acknowledgements
- Flower (Federated Learning): `https://flower.dev`
- Streamlit: `https://streamlit.io`
- scikit-learn and imbalanced-learn 
