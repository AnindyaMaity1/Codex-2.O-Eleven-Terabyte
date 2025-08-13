# server4.py
from flask import Flask, request, jsonify, send_file
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline # Ensure ImbPipeline is imported
from imblearn.over_sampling import SMOTE
import pickle
import os
import logging
import threading
import datetime
import flwr as fl
from collections import defaultdict
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Union
from flask_cors import CORS # Import CORS

# --- Configuration Constants for Diabetes Model ---
SERVER_HOST = os.getenv('SERVER_HOST', '0.0.0.0')
FLASK_PORT = int(os.getenv('FLASK_PORT', 5000))

FLOWER_DIABETES_GRPC_PORT = int(os.getenv('FLOWER_DIABETES_GRPC_PORT', 8080))
DIABETES_GLOBAL_MODEL_FILE = os.getenv('DIABETES_GLOBAL_MODEL_FILE', 'global_diabetes_svm_model.pkl')
CLIENT_DIABETES_DATA_DIR = os.getenv('CLIENT_DIABETES_DATA_DIR', 'client_diabetes_data')
MASTER_DIABETES_AGGREGATED_DATA_FILE = os.getenv('MASTER_DIABETES_AGGREGATED_DATA_FILE', 'server_diabetes_master.csv')

# --- Configuration Constants for Heart Disease Model ---
FLOWER_HEART_GRPC_PORT = int(os.getenv('FLOWER_HEART_GRPC_PORT', 8081)) # Different port for heart FL server
HEART_GLOBAL_MODEL_FILE = os.getenv('HEART_GLOBAL_MODEL_FILE', 'global_heart_svm_model.pkl')
CLIENT_HEART_DATA_DIR = os.getenv('CLIENT_HEART_DATA_DIR', 'client_heart_data')
MASTER_HEART_AGGREGATED_DATA_FILE = os.getenv('MASTER_HEART_AGGREGATED_DATA_FILE', 'server_heart_master.csv')

# --- Configuration Constants for Parkinson's Disease Model ---
FLOWER_PARKINSONS_GRPC_PORT = int(os.getenv('FLOWER_PARKINSONS_GRPC_PORT', 8082)) # Different port for Parkinson's FL server
PARKINSONS_GLOBAL_MODEL_FILE = os.getenv('PARKINSONS_GLOBAL_MODEL_FILE', 'global_parkinsons_svm_model.pkl')
CLIENT_PARKINSONS_DATA_DIR = os.getenv('CLIENT_PARKINSONS_DATA_DIR', 'client_parkinsons_data')
MASTER_PARKINSONS_AGGREGATED_DATA_FILE = os.getenv('MASTER_PARKINSONS_AGGREGATED_DATA_FILE', 'server_parkinsons_master.csv')


# --- Common FL Configuration ---
MAX_FLOWER_ROUNDS = int(os.getenv('MAX_FLOWER_ROUNDS', 5)) # Example: Run 5 FL rounds
MIN_CLIENTS_PER_ROUND = int(os.getenv('MIN_CLIENTS_PER_ROUND', 1))
SMOTE_RANDOM_STATE = int(os.getenv('SMOTE_RANDOM_STATE', 42))
SVM_RANDOM_STATE = int(os.getenv('SVM_RANDOM_STATE', 42))

# --- Flask App Initialization ---
app = Flask(__name__)
# Configure CORS to allow requests from the ZDT Admin Dashboard, which runs on port 5001
# This is necessary because the ZDT Admin Dashboard (client) will be making requests to this Flask server.
CORS(app, resources={r"/*": {"origins": "http://127.0.0.1:5001"}})

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# Ensure client data directories exist
os.makedirs(CLIENT_DIABETES_DATA_DIR, exist_ok=True)
os.makedirs(CLIENT_HEART_DATA_DIR, exist_ok=True)
os.makedirs(CLIENT_PARKINSONS_DATA_DIR, exist_ok=True)

# --- Global State for Diabetes FL ---
connected_diabetes_clients_info = defaultdict(lambda: {'last_seen': None, 'ip': None, 'data_size': 0})
flower_diabetes_client_instances = {} # Stores SklearnFlowerClient objects for diabetes
flower_diabetes_client_lock = threading.Lock()
diabetes_fl_training_status = {'is_running': False, 'current_round': 0, 'total_rounds': MAX_FLOWER_ROUNDS, 'last_aggregation_time': None, 'global_model_accuracy': None}
global_diabetes_model = None
global_diabetes_model_version = 0
global_diabetes_model_lock = threading.Lock() # Protects global_diabetes_model and global_diabetes_model_version

# --- Global State for Heart Disease FL ---
connected_heart_clients_info = defaultdict(lambda: {'last_seen': None, 'ip': None, 'data_size': 0})
flower_heart_client_instances = {} # Stores SklearnHeartFlowerClient objects for heart
flower_heart_client_lock = threading.Lock()
heart_fl_training_status = {'is_running': False, 'current_round': 0, 'total_rounds': MAX_FLOWER_ROUNDS, 'last_aggregation_time': None, 'global_model_accuracy': None}
global_heart_model = None
global_heart_model_version = 0
global_heart_model_lock = threading.Lock() # Protects global_heart_model and global_heart_model_version

# --- Global State for Parkinson's Disease FL ---
connected_parkinsons_clients_info = defaultdict(lambda: {'last_seen': None, 'ip': None, 'data_size': 0})
flower_parkinsons_client_instances = {} # Stores SklearnParkinsonsFlowerClient objects
flower_parkinsons_client_lock = threading.Lock()
parkinsons_fl_training_status = {'is_running': False, 'current_round': 0, 'total_rounds': MAX_FLOWER_ROUNDS, 'last_aggregation_time': None, 'global_model_accuracy': None}
global_parkinsons_model = None
global_parkinsons_model_version = 0
global_parkinsons_model_lock = threading.Lock() # Protects global_parkinsons_model and global_parkinsons_model_version


# --- Zero-Day Threat (ZDT) Configuration ---
ATTACK_THRESHOLD_PER_MINUTE = 100 # Max requests per minute from one IP/client
BLOCK_DURATION_SECONDS = 300 # Block for 5 minutes

# Global state for ZDT
request_counts = defaultdict(lambda: {'count': 0, 'timestamp': time.time()})
blocked_ips = {} # IP -> unblock_time (timestamp when block expires)
attack_log = [] # Stores details of detected attacks (timestamp, type, ip, details)
attack_counts = defaultdict(int) # Type of attack -> count

zdt_lock = threading.Lock() # Protects ZDT global state

def log_attack(attack_type, ip, details=""):
    """Logs a detected attack and increments its count."""
    with zdt_lock:
        attack_counts[attack_type] += 1
        timestamp = datetime.datetime.now().isoformat()
        attack_log.append({"timestamp": timestamp, "type": attack_type, "ip": ip, "details": details})
        # Keep log size manageable (e.g., last 1000 entries)
        if len(attack_log) > 1000:
            attack_log.pop(0) # Remove oldest entry


# --- Helper to create initial empty master CSVs if they don't exist ---
def initialize_master_aggregated_data(file_path, columns):
    """Initializes an empty master CSV file with specified columns if it doesn't exist."""
    if not os.path.exists(file_path):
        log.info(f"Creating initial empty {file_path} for central aggregation.")
        pd.DataFrame(columns=columns).to_csv(file_path, index=False)
    else:
        log.info(f"{file_path} already exists.")

# --- Scikit-learn Pipeline Factory for Diabetes ---
def create_diabetes_sklearn_pipeline():
    """Creates a new Scikit-learn pipeline for the Diabetes SVM model."""
    return ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=SMOTE_RANDOM_STATE)),
        ('svm', SVC(kernel='linear', random_state=SVM_RANDOM_STATE, probability=True))
    ])

# --- Scikit-learn Pipeline Factory for Heart Disease ---
def create_heart_sklearn_pipeline():
    """Creates a new Scikit-learn pipeline for the Heart Disease SVM model."""
    return ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=SMOTE_RANDOM_STATE)),
        ('svm', SVC(kernel='linear', random_state=SVM_RANDOM_STATE, probability=True))
    ])

# --- Scikit-learn Pipeline Factory for Parkinson's Disease ---
def create_parkinsons_sklearn_pipeline():
    """Creates a new Scikit-learn pipeline for the Parkinson's SVM model."""
    return ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=SMOTE_RANDOM_STATE)),
        ('svm', SVC(kernel='linear', random_state=SVM_RANDOM_STATE, probability=True))
    ])


# --- Flower Client Definitions (Server-Side Simulation of Clients) ---

class SklearnBaseFlowerClient(fl.client.NumPyClient):
    """Base class for Flower clients to reduce code duplication."""
    def __init__(self, client_id, data_path, expected_features, create_pipeline_fn):
        self.client_id = client_id
        self.data_path = data_path
        self.expected_features = expected_features
        self.model = create_pipeline_fn()
        self.X, self.y = self._load_data() # Initial load
        log.info(f"Client {self.client_id} initialized with {len(self.X)} samples for model type: {type(self).__name__}.")

    def _load_data(self):
        """Loads data from the client's CSV file and handles edge cases."""
        try:
            df = pd.read_csv(self.data_path)
            
            # Ensure all expected features are present, fill with 0 if not
            for feature in self.expected_features:
                if feature not in df.columns:
                    df[feature] = 0
            
            # Select only the expected features in the correct order
            X = df[self.expected_features].values
            
            # Robustly handle 'Outcome' column
            if 'Outcome' in df.columns:
                y = df['Outcome'].values
            else:
                log.warning(f"Client {self.client_id} data missing 'Outcome' column. Initializing y with zeros.")
                y = np.zeros(len(df))

            # Handle insufficient unique classes or empty data for SMOTE/training
            if len(np.unique(y)) < 2 and len(y) > 0:
                log.warning(f"Client {self.client_id} data has only one class ({np.unique(y)[0]}) in 'Outcome'. Appending dummy data for training to enable SMOTE.")
                dummy_X_row = np.zeros(X.shape[1])
                dummy_y_row = 1 if np.unique(y)[0] == 0 else 0
                X = np.vstack([X, dummy_X_row])
                y = np.append(y, dummy_y_row)
            elif len(y) == 0:
                log.warning(f"Client {self.client_id} data is empty. Initializing with dummy data for training.")
                X = np.zeros((2, len(self.expected_features))) # At least 2 samples for SMOTE to work
                y = np.array([0, 1]) # Ensure both classes are present

            return X.astype(np.float32), y.astype(np.int64) # Ensure X and y are NumPy arrays with correct dtypes
        except FileNotFoundError:
            log.warning(f"Client data file not found for {self.client_id} at {self.data_path}. Initializing with dummy data.")
            return np.zeros((2, len(self.expected_features))).astype(np.float32), np.array([0, 1]).astype(np.int64)
        except Exception as e:
            log.error(f"Error loading data for client {self.client_id}: {e}")
            return np.zeros((2, len(self.expected_features))).astype(np.float32), np.array([0, 1]).astype(np.int64)

    def get_parameters(self, config):
        """Returns the current model parameters (coef_ and intercept_ for SVM)."""
        if hasattr(self.model.named_steps['svm'], 'coef_') and hasattr(self.model.named_steps['svm'], 'intercept_'):
            return [self.model.named_steps['svm'].coef_, self.model.named_steps['svm'].intercept_]
        log.warning(f"Client {self.client_id}: SVM model does not have 'coef_' or 'intercept_'. Returning empty parameters.")
        return []

    def set_parameters(self, parameters):
        """Sets the model parameters from the aggregated global model."""
        if len(parameters) == 2:
            self.model.named_steps['svm'].coef_ = parameters[0]
            self.model.named_steps['svm'].intercept_ = parameters[1]
            log.debug(f"Client {self.client_id}: Parameters set successfully.")
        else:
            log.warning(f"Client {self.client_id}: Expected 2 parameters (coef_, intercept_), got {len(parameters)}. Not updating model.")

    def fit(self, parameters, config):
        """Performs local training (fit) on the client's data."""
        log.info(f"Client {self.client_id}: Starting fit for round {config.get('server_round')}")
        self.set_parameters(parameters) # Update local model with global parameters
        
        # Reload data before fitting to ensure it's up-to-date with any new uploads
        self.X, self.y = self._load_data()

        if len(self.X) < 2 or len(np.unique(self.y)) < 2:
            log.warning(f"Client {self.client_id}: Insufficient data ({len(self.X)} samples, {len(np.unique(self.y))} classes) for training. Skipping fit.")
            # Return current parameters, 0 examples, and empty metrics to signify no training
            return self.get_parameters(config={}), 0, {}
        
        try:
            # Check if SMOTE is in the pipeline and if it can be applied
            if 'smote' in self.model.named_steps and len(np.unique(self.y)) < 2:
                log.warning(f"Client {self.client_id}: SMOTE cannot be applied with only one class. Training without SMOTE.")
                # Temporarily create a pipeline without SMOTE
                temp_pipeline = ImbPipeline([('scaler', self.model.named_steps['scaler']), ('svm', self.model.named_steps['svm'])])
                # Ensure the SVM's attributes are copied to the temporary pipeline's SVM
                if hasattr(self.model.named_steps['svm'], 'coef_'):
                    temp_pipeline.named_steps['svm'].coef_ = self.model.named_steps['svm'].coef_
                if hasattr(self.model.named_steps['svm'], 'intercept_'):
                    temp_pipeline.named_steps['svm'].intercept_ = self.model.named_steps['svm'].intercept_
                if hasattr(self.model.named_steps['svm'], 'classes_'):
                    temp_pipeline.named_steps['svm'].classes_ = self.model.named_steps['svm'].classes_
                temp_pipeline.fit(self.X, self.y)
                self.model = temp_pipeline # Update the client's model
            else:
                self.model.fit(self.X, self.y)
            num_examples = len(self.X)
            log.info(f"Client {self.client_id}: Finished fit. Trained on {num_examples} samples.")
            return self.get_parameters(config={}), num_examples, {}
        except Exception as e:
            log.error(f"Client {self.client_id}: Error during fit: {e}. Returning no parameters.")
            return self.get_parameters(config={}), 0, {"error": str(e)}

    def evaluate(self, parameters, config):
        """Performs local evaluation on the client's data."""
        log.info(f"Client {self.client_id}: Starting evaluation for round {config.get('server_round')}")
        self.set_parameters(parameters) # Update local model with global parameters for evaluation
        
        # Reload data before evaluating
        X_eval, y_eval = self._load_data()
        
        if len(X_eval) < 1 or len(np.unique(y_eval)) < 2:
            log.warning(f"Client {self.client_id}: Insufficient data ({len(X_eval)} samples, {len(np.unique(y_eval))} classes) for evaluation. Returning default metrics.")
            return 1.0, 0, {"accuracy": 0.0, "loss": 1.0} # Return a default loss to avoid issues

        try:
            loss = 1.0 - self.model.score(X_eval, y_eval) # Simple loss approximation
            accuracy = self.model.score(X_eval, y_eval)
            num_examples = len(X_eval)
            log.info(f"Client {self.client_id}: Finished evaluation. Accuracy: {accuracy:.4f}")
            return loss, num_examples, {"accuracy": accuracy, "loss": loss}
        except Exception as e:
            log.error(f"Client {self.client_id}: Error during evaluate: {e}. Returning default metrics.")
            return 1.0, 0, {"accuracy": 0.0, "error": str(e)}


# Diabetes Specific Client
DIABETES_FEATURES = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]
class SklearnDiabetesFlowerClient(SklearnBaseFlowerClient):
    def __init__(self, client_id, data_path):
        super().__init__(client_id, data_path, DIABETES_FEATURES, create_diabetes_sklearn_pipeline)

# Heart Disease Specific Client
HEART_FEATURES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]
class SklearnHeartFlowerClient(SklearnBaseFlowerClient):
    def __init__(self, client_id, data_path):
        super().__init__(client_id, data_path, HEART_FEATURES, create_heart_sklearn_pipeline)

# Parkinson's Disease Specific Client
PARKINSONS_FEATURES = [
    'fo', 'fhi', 'flo', 'Jitter_percent', 'Jitter_Abs', 'RAP', 'PPQ', 'DDP',
    'Shimmer', 'Shimmer_dB', 'APQ3', 'APQ5', 'APQ', 'DDA', 'NHR', 'HNR',
    'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
]
class SklearnParkinsonsFlowerClient(SklearnBaseFlowerClient):
    def __init__(self, client_id, data_path):
        super().__init__(client_id, data_path, PARKINSONS_FEATURES, create_parkinsons_sklearn_pipeline)


# --- Flower Server Strategies ---

class BaseModelFedAvg(fl.server.strategy.FedAvg):
    """Base class for FedAvg strategy to reduce code duplication."""
    def __init__(self, model_file, global_model_ref, global_model_version_ref, global_model_lock_ref, training_status_ref, create_pipeline_fn, dummy_feature_count, **kwargs):
        super().__init__(**kwargs)
        self.model_file = model_file
        self.global_model_ref = global_model_ref
        self.global_model_version_ref = global_model_version_ref
        self.global_model_lock_ref = global_model_lock_ref
        self.training_status_ref = training_status_ref
        self.create_pipeline_fn = create_pipeline_fn
        self.dummy_feature_count = dummy_feature_count
        self.pipeline = None
        self.latest_global_accuracy = 0.0

    def initialize_parameters(self, client_manager: fl.server.client_manager.ClientManager):
        """Initializes global model parameters. Tries to load from file, else creates new."""
        with self.global_model_lock_ref:
            if os.path.exists(self.model_file):
                try:
                    with open(self.model_file, 'rb') as f:
                        self.pipeline = pickle.load(f)
                    log.info(f"Loaded existing global model from {self.model_file}.")
                    self.global_model_version_ref[0] += 1 # Increment version if loaded
                except Exception as e:
                    log.warning(f"Could not load existing global model from {self.model_file}: {e}. Creating a new one.")
                    self.pipeline = self.create_pipeline_fn()
                    # Fit with dummy data to initialize coef_ and intercept_
                    dummy_X = pd.DataFrame(np.zeros((2, self.dummy_feature_count)))
                    dummy_y = np.array([0, 1])
                    self.pipeline.fit(dummy_X, dummy_y)
            else:
                log.info("No existing global model found. Initializing a new one.")
                self.pipeline = self.create_pipeline_fn()
                # Fit with dummy data to initialize coef_ and intercept_
                dummy_X = pd.DataFrame(np.zeros((2, self.dummy_feature_count)))
                dummy_y = np.array([0, 1])
                self.pipeline.fit(dummy_X, dummy_y)
            
            self.global_model_ref[0] = self.pipeline # Update global reference (using mutable list for ref)

            if hasattr(self.pipeline.named_steps['svm'], 'coef_') and hasattr(self.pipeline.named_steps['svm'], 'intercept_'):
                model_weights = [self.pipeline.named_steps['svm'].coef_, self.pipeline.named_steps['svm'].intercept_]
                ndarrays = [np.array(w, dtype=np.float32) for w in model_weights]
                log.info(f"Initialized SVM with coef_ shape: {ndarrays[0].shape}, intercept_ shape: {ndarrays[1].shape}")
                return fl.common.ndarrays_to_parameters(ndarrays)
            else:
                log.error("SVM model in pipeline does not have 'coef_' or 'intercept_'. This is unexpected for a linear SVM. Returning empty parameters.")
                return fl.common.ndarrays_to_parameters([])

    def aggregate_fit(self,
                      rnd: int,
                      results: list[tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
                      failures: list[Union[tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]],
    ) -> tuple[fl.common.Parameters | None, dict[str, fl.common.Scalar]]:
        """Aggregates client updates and saves the global model."""
        log.info(f"Aggregating fit results for round {rnd} from {len(results)} clients for {self.model_file}.")
        aggregated_parameters, metrics = super().aggregate_fit(rnd, results, failures)
        
        with self.global_model_lock_ref:
            if aggregated_parameters is not None:
                log.info(f"Aggregation complete for round {rnd}. Saving global model to {self.model_file}.")
                ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)
                
                if self.pipeline and len(ndarrays) == 2:
                    self.pipeline.named_steps['svm'].coef_ = ndarrays[0]
                    self.pipeline.named_steps['svm'].intercept_ = ndarrays[1]
                    
                    with open(self.model_file, 'wb') as f:
                        pickle.dump(self.pipeline, f)
                    self.global_model_ref[0] = self.pipeline # Update global reference
                    self.global_model_version_ref[0] += 1
                    log.info(f"Global model saved to {self.model_file}. New version: {self.global_model_version_ref[0]}")
                else:
                    log.warning(f"Failed to update or save global model for round {rnd} ({self.model_file}). Pipeline not initialized or unexpected ndarrays: {len(ndarrays)}.")
            else:
                log.warning(f"No aggregated parameters for round {rnd} for {self.model_file}. Global model not updated.")
            
        # Update FL status
        self.training_status_ref['current_round'] = rnd
        self.training_status_ref['last_aggregation_time'] = datetime.datetime.now().isoformat()
            
        return aggregated_parameters, metrics

    def aggregate_evaluate(self,
                            rnd: int,
                            results: list[tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
                            failures: list[Union[tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes], BaseException]],
    ) -> tuple[fl.common.Scalar | None, dict[str, fl.common.Scalar]]:
        """Aggregates evaluation results and updates global accuracy."""
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(rnd, results, failures)
        
        if aggregated_metrics and 'accuracy' in aggregated_metrics:
            self.latest_global_accuracy = aggregated_metrics['accuracy']
            self.training_status_ref['global_model_accuracy'] = aggregated_metrics['accuracy']
            log.info(f"Round {rnd} Global Evaluation for {self.model_file}: Accuracy = {self.latest_global_accuracy:.4f}")
        else:
            log.warning(f"Round {rnd} Global Evaluation for {self.model_file}: No accuracy metric found in aggregated results.")
            self.training_status_ref['global_model_accuracy'] = None

        return aggregated_loss, aggregated_metrics

# Diabetes Specific Strategy
class DiabetesFedAvg(BaseModelFedAvg):
    def __init__(self, **kwargs):
        # Using a list to pass mutable integer references for global_diabetes_model_version
        super().__init__(
            model_file=DIABETES_GLOBAL_MODEL_FILE,
            global_model_ref=[global_diabetes_model],
            global_model_version_ref=[global_diabetes_model_version],
            global_model_lock_ref=global_diabetes_model_lock,
            training_status_ref=diabetes_fl_training_status,
            create_pipeline_fn=create_diabetes_sklearn_pipeline,
            dummy_feature_count=len(DIABETES_FEATURES),
            **kwargs
        )

# Heart Disease Specific Strategy
class HeartFedAvg(BaseModelFedAvg):
    def __init__(self, **kwargs):
        # Using a list to pass mutable integer references for global_heart_model_version
        super().__init__(
            model_file=HEART_GLOBAL_MODEL_FILE,
            global_model_ref=[global_heart_model],
            global_model_version_ref=[global_heart_model_version],
            global_model_lock_ref=global_heart_model_lock,
            training_status_ref=heart_fl_training_status,
            create_pipeline_fn=create_heart_sklearn_pipeline,
            dummy_feature_count=len(HEART_FEATURES),
            **kwargs
        )

# Parkinson's Disease Specific Strategy
class ParkinsonsFedAvg(BaseModelFedAvg):
    def __init__(self, **kwargs):
        super().__init__(
            model_file=PARKINSONS_GLOBAL_MODEL_FILE,
            global_model_ref=[global_parkinsons_model],
            global_model_version_ref=[global_parkinsons_model_version],
            global_model_lock_ref=global_parkinsons_model_lock,
            training_status_ref=parkinsons_fl_training_status,
            create_pipeline_fn=create_parkinsons_sklearn_pipeline,
            dummy_feature_count=len(PARKINSONS_FEATURES),
            **kwargs
        )


# --- Flower Server and Client Simulation Logic ---

def run_flower_server_in_thread(grpc_port, strategy_instance, model_name, client_fn_callback):
    """Runs a Flower server in a separate thread."""
    log.info(f"Starting Flower server for {model_name} on port {grpc_port}...")
    # Update the corresponding training status, assuming status object is passed or accessed by reference
    if model_name == "Diabetes":
        diabetes_fl_training_status['is_running'] = True
    elif model_name == "Heart":
        heart_fl_training_status['is_running'] = True
    elif model_name == "Parkinsons":
        parkinsons_fl_training_status['is_running'] = True

    try:
        fl.server.start_server(
            server_address=f"0.0.0.0:{grpc_port}",
            config=fl.server.ServerConfig(num_rounds=MAX_FLOWER_ROUNDS),
            strategy=strategy_instance,
            client_manager=fl.server.SimpleClientManager(), # Using default manager for simplicity
        )
    except Exception as e:
        log.error(f"Error starting Flower server for {model_name}: {e}")
    finally:
        if model_name == "Diabetes":
            diabetes_fl_training_status['is_running'] = False
        elif model_name == "Heart":
            heart_fl_training_status['is_running'] = False
        elif model_name == "Parkinsons":
            parkinsons_fl_training_status['is_running'] = False
    log.info(f"Flower server for {model_name} stopped.")


# Use global ThreadPoolExecutors for Flower client simulation
diabetes_flower_client_executor = ThreadPoolExecutor(max_workers=10) # Adjust max_workers as needed
heart_flower_client_executor = ThreadPoolExecutor(max_workers=10) # Adjust max_workers as needed
parkinsons_flower_client_executor = ThreadPoolExecutor(max_workers=10)

def connect_flower_client(client_id, grpc_port, client_fn_callback, model_name):
    """Connects a simulated Flower client to the server with retries."""
    log.info(f"Attempting to connect Flower client {client_id} for {model_name}...")
    max_retries = 5
    retry_delay = 2 # seconds
    for attempt in range(max_retries):
        try:
            # Add a small delay before attempting to connect, increasing with each attempt
            time.sleep(retry_delay * attempt) 
            fl.client.start_client(
                server_address=f"127.0.0.1:{grpc_port}", # Clients connect to local Flower gRPC port
                client=client_fn_callback(client_id),
            )
            log.info(f"Flower client {client_id} for {model_name} connected successfully (simulated) on attempt {attempt + 1}.")
            return # Success, exit loop
        except Exception as e:
            log.warning(f"Error connecting Flower client {client_id} for {model_name} on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt == max_retries - 1:
                log.error(f"Failed to connect Flower client {client_id} for {model_name} after {max_retries} attempts.")
                return # All retries failed


# Client function for Diabetes model
def diabetes_client_fn(cid: str) -> fl.client.NumPyClient:
    """Returns a Flower client instance for a given client ID for Diabetes model."""
    with flower_diabetes_client_lock:
        if cid not in flower_diabetes_client_instances:
            client_data_file = os.path.join(CLIENT_DIABETES_DATA_DIR, f"client_diabetes_data_{cid}.csv")
            flower_diabetes_client_instances[cid] = SklearnDiabetesFlowerClient(cid, client_data_file)
            log.info(f"Dynamically created Diabetes client_fn for {cid}.")
        return flower_diabetes_client_instances[cid]

# Client function for Heart model
def heart_client_fn(cid: str) -> fl.client.NumPyClient:
    """Returns a Flower client instance for a given client ID for Heart model."""
    with flower_heart_client_lock:
        if cid not in flower_heart_client_instances:
            client_data_file = os.path.join(CLIENT_HEART_DATA_DIR, f"client_heart_data_{cid}.csv")
            flower_heart_client_instances[cid] = SklearnHeartFlowerClient(cid, client_data_file)
            log.info(f"Dynamically created Heart client_fn for {cid}.")
        return flower_heart_client_instances[cid]

# Client function for Parkinson's model
def parkinsons_client_fn(cid: str) -> fl.client.NumPyClient:
    """Returns a Flower client instance for a given client ID for Parkinson's model."""
    with flower_parkinsons_client_lock:
        if cid not in flower_parkinsons_client_instances:
            client_data_file = os.path.join(CLIENT_PARKINSONS_DATA_DIR, f"client_parkinsons_data_{cid}.csv")
            flower_parkinsons_client_instances[cid] = SklearnParkinsonsFlowerClient(cid, client_data_file)
            log.info(f"Dynamically created Parkinson's client_fn for {cid}.")
        return flower_parkinsons_client_instances[cid]


# --- Flask App Routes for Diabetes Model ---

@app.route('/get_diabetes_model', methods=['GET'])
def get_diabetes_model():
    """Provides the latest global federated model for download for Diabetes."""
    client_ip = request.remote_addr # Get client IP for ZDT
    with zdt_lock:
        if client_ip in blocked_ips and time.time() < blocked_ips[client_ip]:
            log.warning(f"Blocked request from {client_ip} (Diabetes model download).")
            return jsonify({"status": "error", "message": "Your IP has been temporarily blocked due to suspicious activity."}), 403

    with global_diabetes_model_lock:
        if os.path.exists(DIABETES_GLOBAL_MODEL_FILE):
            try:
                return send_file(DIABETES_GLOBAL_MODEL_FILE, 
                                 as_attachment=True, 
                                 download_name=f'global_diabetes_model_v{global_diabetes_model_version}.pkl',
                                 mimetype='application/octet-stream')
            except Exception as e:
                log.error(f"Error sending Diabetes model file: {e}")
                return jsonify({"status": "error", "message": f"Server error sending Diabetes model: {str(e)}"}), 500
        else:
            log.warning("Global Diabetes model not found. This might happen initially before the first Flower round or if Flower server hasn't saved it yet.")
            return jsonify({"status": "error", "message": "Global Diabetes model not available yet. Please try again later."}), 404

@app.route('/get_diabetes_model_info', methods=['GET'])
def get_diabetes_model_info():
    """Provides metadata about the current global Diabetes model."""
    client_ip = request.remote_addr # Get client IP for ZDT
    with zdt_lock:
        if client_ip in blocked_ips and time.time() < blocked_ips[client_ip]:
            log.warning(f"Blocked request from {client_ip} (Diabetes model info).")
            return jsonify({"status": "error", "message": "Your IP has been temporarily blocked due to suspicious activity."}), 403

    with global_diabetes_model_lock:
        return jsonify({
            "status": "success",
            "model_type": "Diabetes",
            "model_available": os.path.exists(DIABETES_GLOBAL_MODEL_FILE), # Corrected: Use os.path.exists
            "version": global_diabetes_model_version,
            "fl_status": diabetes_fl_training_status
        })

@app.route('/upload_diabetes_client_data', methods=['POST'])
def upload_diabetes_client_data():
    """Receives and processes new data from clients for Diabetes model."""
    client_ip = request.remote_addr
    
    # ZDT: Check if IP is blocked
    with zdt_lock:
        if client_ip in blocked_ips and time.time() < blocked_ips[client_ip]:
            log.warning(f"Blocked request from {client_ip} (Diabetes upload).")
            return jsonify({"status": "error", "message": "Your IP has been temporarily blocked due to suspicious activity."}), 403
        
        # ZDT: Rate limiting
        current_time = time.time()
        if current_time - request_counts[client_ip]['timestamp'] > 60: # Reset count after 1 minute
            request_counts[client_ip]['count'] = 1
            request_counts[client_ip]['timestamp'] = current_time
        else:
            request_counts[client_ip]['count'] += 1
            if request_counts[client_ip]['count'] > ATTACK_THRESHOLD_PER_MINUTE:
                blocked_ips[client_ip] = current_time + BLOCK_DURATION_SECONDS
                log_attack("RateLimitExceeded", client_ip, f"Too many requests ({request_counts[client_ip]['count']}) in 1 minute.")
                log.error(f"Blocking IP {client_ip} for {BLOCK_DURATION_SECONDS} seconds due to rate limit.")
                return jsonify({"status": "error", "message": "Too many requests. Your IP has been temporarily blocked."}), 429 # Too Many Requests

    # ZDT: Check if request is JSON
    if not request.is_json:
        log_attack("InvalidPayloadFormat", client_ip, "Request not JSON.")
        return jsonify({"status": "error", "message": "Request must be JSON"}), 400

    data = request.get_json()
    client_id = data.get('client_id')
    new_records = data.get('records')

    # ZDT: Check for missing essential data
    if not client_id or not new_records:
        log_attack("MissingData", client_ip, "Missing client_id or records in payload.")
        return jsonify({"status": "error", "message": "Missing client_id or records"}), 400
    
    # ZDT: Basic data validation for Diabetes features and Outcome
    DIABETES_FEATURES_AND_OUTCOME = DIABETES_FEATURES + ['Outcome']
    for record in new_records:
        # Check for expected keys
        if not all(k in record for k in DIABETES_FEATURES_AND_OUTCOME):
            log_attack("MalformedRecord", client_ip, f"Missing expected features in Diabetes record: {record}")
            return jsonify({"status": "error", "message": "Malformed data record detected (missing features)."}), 400
        
        try:
            # Outcome should be 0 or 1
            if 'Outcome' in record and record['Outcome'] not in [0, 1]:
                log_attack("InvalidOutcome", client_ip, f"Invalid 'Outcome' value in Diabetes record: {record}")
                return jsonify({"status": "error", "message": "Invalid outcome value detected."}), 400
            
        except (ValueError, TypeError):
            log_attack("InvalidDataType", client_ip, f"Non-numeric data in Diabetes record: {record}")
            return jsonify({"status": "error", "message": "Data contains non-numeric values."}), 400

    log.info(f"Received {len(new_records)} new records from client {client_id} for Diabetes.")

    # Update client liveness information
    connected_diabetes_clients_info[client_id]['last_seen'] = datetime.datetime.now().isoformat()
    connected_diabetes_clients_info[client_id]['ip'] = request.remote_addr
    
    client_data_file = os.path.join(CLIENT_DIABETES_DATA_DIR, f"client_diabetes_data_{client_id}.csv")

    processed_records = []
    for record in new_records:
        # Basic validation and standardization of 'Outcome'
        if 'Outcome' not in record:
            outcome_val = record.get('result', '').lower()
            if outcome_val == 'yes':
                record['Outcome'] = 1
            elif outcome_val == 'no':
                record['Outcome'] = 0
            else:
                log.warning(f"Client {client_id} (Diabetes): Invalid or missing 'Outcome'/'result' for a record. Defaulting to 0. Record: {record}")
                record['Outcome'] = 0 
            if 'result' in record:
                del record['result'] # Remove 'result' if 'Outcome' is derived
        processed_records.append(record)

    new_df = pd.DataFrame(processed_records)
    
    try:
        # --- Update client-specific CSV ---
        hash_cols = DIABETES_FEATURES + ['Outcome']
        
        if os.path.exists(client_data_file):
            existing_client_df = pd.read_csv(client_data_file)
            
            for col in hash_cols:
                if col not in new_df.columns:
                    new_df[col] = 0
            new_df_hashes = new_df[hash_cols].astype(str).agg(''.join, axis=1).apply(hash)
            
            # Ensure existing_client_df also has all hash_cols for consistent hashing
            for col in hash_cols:
                if col not in existing_client_df.columns:
                    existing_client_df[col] = 0
            existing_client_hashes = existing_client_df[hash_cols].astype(str).agg(''.join, axis=1).apply(hash)

            unique_new_records_df = new_df[~new_df_hashes.isin(existing_client_hashes)]

            if not unique_new_records_df.empty:
                # Ensure combined columns are consistent
                combined_columns = list(set(existing_client_df.columns).union(unique_new_records_df.columns))
                existing_client_df = existing_client_df.reindex(columns=combined_columns, fill_value=0)
                unique_new_records_df = unique_new_records_df.reindex(columns=combined_columns, fill_value=0)
                
                updated_client_df = pd.concat([existing_client_df, unique_new_records_df], ignore_index=True)
                updated_client_df.to_csv(client_data_file, index=False)
                log.info(f"{len(unique_new_records_df)} unique new records added to {client_data_file} for client {client_id} (Diabetes).")
                connected_diabetes_clients_info[client_id]['data_size'] = len(updated_client_df) # Update data size
            else:
                log.info(f"No new unique records to add for client {client_id} (Diabetes). Data size: {len(existing_client_df)}")
                connected_diabetes_clients_info[client_id]['data_size'] = len(existing_client_df)
                # No return here, continue to master file logic even if client's local file didn't change from this upload

        else:
            new_df.to_csv(client_data_file, index=False)
            log.info(f"{len(new_records)} new records created for {client_data_file} for client {client_id} (Diabetes).")
            connected_diabetes_clients_info[client_id]['data_size'] = len(new_df) # Set initial data size

        # --- Update central MASTER_DIABETES_AGGREGATED_DATA_FILE ---
        log.info(f"Attempting to update master aggregated data for Diabetes. Records from client {client_id} to consider: {len(new_df)}")
        if os.path.exists(MASTER_DIABETES_AGGREGATED_DATA_FILE):
            master_df = pd.read_csv(MASTER_DIABETES_AGGREGATED_DATA_FILE)
            
            for col in hash_cols:
                if col not in master_df.columns:
                    master_df[col] = 0
            master_df_hashes = master_df[hash_cols].astype(str).agg(''.join, axis=1).apply(hash)
            
            for col in hash_cols:
                if col not in new_df.columns:
                    new_df[col] = 0 
            new_df_hashes_for_master = new_df[hash_cols].astype(str).agg(''.join, axis=1).apply(hash)

            unique_new_records_for_master = new_df[~new_df_hashes_for_master.isin(master_df_hashes)]
            
            log.info(f"Number of unique new records for MASTER_DIABETES_AGGREGATED_DATA_FILE from client {client_id}: {len(unique_new_records_for_master)}")
            if not unique_new_records_for_master.empty:
                all_master_columns = list(set(master_df.columns).union(unique_new_records_for_master.columns))
                master_df = master_df.reindex(columns=all_master_columns, fill_value=0)
                unique_new_records_for_master = unique_new_records_for_master.reindex(columns=all_master_columns, fill_value=0)
                
                combined_master_df = pd.concat([master_df, unique_new_records_for_master], ignore_index=True)
                combined_master_df.to_csv(MASTER_DIABETES_AGGREGATED_DATA_FILE, index=False)
                log.info(f"{len(unique_new_records_for_master)} unique new records also added to {MASTER_DIABETES_AGGREGATED_DATA_FILE}.")
            else:
                log.info(f"No new unique records from client {client_id} to add to master aggregated Diabetes data.")
        else:
            new_df.to_csv(MASTER_DIABETES_AGGREGATED_DATA_FILE, index=False)
            log.warning(f"{MASTER_DIABETES_AGGREGATED_DATA_FILE} was not found, created it with new data from client {client_id}.")

        # --- Flower Client Management ---
        with flower_diabetes_client_lock:
            if client_id not in flower_diabetes_client_instances:
                flower_diabetes_client_instances[client_id] = SklearnDiabetesFlowerClient(client_id, client_data_file)
                log.info(f"New Flower client instance created for Diabetes ID: {client_id}.")
                diabetes_flower_client_executor.submit(connect_flower_client, client_id, FLOWER_DIABETES_GRPC_PORT, diabetes_client_fn, "Diabetes")
            else:
                # Explicitly reload data for existing client instance to ensure it has the latest data
                flower_diabetes_client_instances[client_id].X, flower_diabetes_client_instances[client_id].y = flower_diabetes_client_instances[client_id]._load_data()
                log.info(f"Refreshed data for existing Flower Diabetes client instance {client_id}.")

    except Exception as e:
        log.error(f"Error handling data upload for Diabetes client {client_id}: {e}", exc_info=True)
        return jsonify({"status": "error", "message": f"Server internal error: {str(e)}"}), 500

    return jsonify({"status": "success", "message": "Diabetes data received and saved, client registered/updated for FL."})


# --- Flask App Routes for Heart Disease Model ---

@app.route('/get_heart_model', methods=['GET'])
def get_heart_model():
    """Provides the latest global federated model for download for Heart Disease."""
    client_ip = request.remote_addr # Get client IP for ZDT
    with zdt_lock:
        if client_ip in blocked_ips and time.time() < blocked_ips[client_ip]:
            log.warning(f"Blocked request from {client_ip} (Heart model download).")
            return jsonify({"status": "error", "message": "Your IP has been temporarily blocked due to suspicious activity."}), 403

    with global_heart_model_lock:
        if os.path.exists(HEART_GLOBAL_MODEL_FILE):
            try:
                return send_file(HEART_GLOBAL_MODEL_FILE, 
                                 as_attachment=True, 
                                 download_name=f'global_heart_model_v{global_heart_model_version}.pkl',
                                 mimetype='application/octet-stream')
            except Exception as e:
                log.error(f"Error sending Heart model file: {e}")
                return jsonify({"status": "error", "message": f"Server error sending Heart model: {str(e)}"}), 500
        else:
            log.warning("Global Heart model not found. This might happen initially before the first Flower round or if Flower server hasn't saved it yet.")
            return jsonify({"status": "error", "message": "Global Heart model not available yet. Please try again later."}), 404

@app.route('/get_heart_model_info', methods=['GET'])
def get_heart_model_info():
    """Provides metadata about the current global Heart Disease model."""
    client_ip = request.remote_addr # Get client IP for ZDT
    with zdt_lock:
        if client_ip in blocked_ips and time.time() < blocked_ips[client_ip]:
            log.warning(f"Blocked request from {client_ip} (Heart model info).")
            return jsonify({"status": "error", "message": "Your IP has been temporarily blocked due to suspicious activity."}), 403

    with global_heart_model_lock:
        return jsonify({
            "status": "success",
            "model_type": "Heart Disease",
            "model_available": os.path.exists(HEART_GLOBAL_MODEL_FILE), # Corrected: Use os.path.exists
            "version": global_heart_model_version,
            "fl_status": heart_fl_training_status
        })

@app.route('/upload_heart_client_data', methods=['POST'])
def upload_heart_client_data():
    """Receives and processes new data from clients for Heart Disease model."""
    client_ip = request.remote_addr
    
    # ZDT: Check if IP is blocked
    with zdt_lock:
        if client_ip in blocked_ips and time.time() < blocked_ips[client_ip]:
            log.warning(f"Blocked request from {client_ip} (Heart upload).")
            return jsonify({"status": "error", "message": "Your IP has been temporarily blocked due to suspicious activity."}), 403
        
        # ZDT: Rate limiting
        current_time = time.time()
        if current_time - request_counts[client_ip]['timestamp'] > 60: # Reset count after 1 minute
            request_counts[client_ip]['count'] = 1
            request_counts[client_ip]['timestamp'] = current_time
        else:
            request_counts[client_ip]['count'] += 1
            if request_counts[client_ip]['count'] > ATTACK_THRESHOLD_PER_MINUTE:
                blocked_ips[client_ip] = current_time + BLOCK_DURATION_SECONDS
                log_attack("RateLimitExceeded", client_ip, f"Too many requests ({request_counts[client_ip]['count']}) in 1 minute.")
                log.error(f"Blocking IP {client_ip} for {BLOCK_DURATION_SECONDS} seconds due to rate limit.")
                return jsonify({"status": "error", "message": "Too many requests. Your IP has been temporarily blocked."}), 429 # Too Many Requests

    # ZDT: Check if request is JSON
    if not request.is_json:
        log_attack("InvalidPayloadFormat", client_ip, "Request not JSON.")
        return jsonify({"status": "error", "message": "Request must be JSON"}), 400

    data = request.get_json()
    client_id = data.get('client_id')
    new_records = data.get('records')

    # ZDT: Check for missing essential data
    if not client_id or not new_records:
        log_attack("MissingData", client_ip, "Missing client_id or records in payload.")
        return jsonify({"status": "error", "message": "Missing client_id or records"}), 400

    # ZDT: Basic data validation for Heart Disease features and Outcome
    HEART_FEATURES_AND_OUTCOME = HEART_FEATURES + ['Outcome']
    for record in new_records:
        # Check for expected keys
        if not all(k in record for k in HEART_FEATURES_AND_OUTCOME):
            log_attack("MalformedRecord", client_ip, f"Missing expected features in Heart record: {record}")
            return jsonify({"status": "error", "message": "Malformed data record detected (missing features)."}), 400
        
        try:
            # Outcome should be 0 or 1
            if 'Outcome' in record and record['Outcome'] not in [0, 1]:
                log_attack("InvalidOutcome", client_ip, f"Invalid 'Outcome' value in Heart record: {record}")
                return jsonify({"status": "error", "message": "Invalid outcome value detected."}), 400

        except (ValueError, TypeError):
            log_attack("InvalidDataType", client_ip, f"Non-numeric data in Heart record: {record}")
            return jsonify({"status": "error", "message": "Data contains non-numeric values."}), 400

    log.info(f"Received {len(new_records)} new records from client {client_id} for Heart Disease.")

    # Update client liveness information
    connected_heart_clients_info[client_id]['last_seen'] = datetime.datetime.now().isoformat()
    connected_heart_clients_info[client_id]['ip'] = request.remote_addr
    
    client_data_file = os.path.join(CLIENT_HEART_DATA_DIR, f"client_heart_data_{client_id}.csv")

    processed_records = []
    for record in new_records:
        # Basic validation and standardization of 'Outcome'
        if 'Outcome' not in record:
            outcome_val = record.get('target', '').lower() # Assuming 'target' for heart disease outcome
            if outcome_val in ['1', 1, True, 'yes']: # Robustly handle target=1
                record['Outcome'] = 1
            elif outcome_val in ['0', 0, False, 'no']: # Robustly handle target=0
                record['Outcome'] = 0
            else:
                log.warning(f"Client {client_id} (Heart): Invalid or missing 'Outcome'/'target' for a record. Defaulting to 0. Record: {record}")
                record['Outcome'] = 0 
            if 'target' in record: # Remove 'target' if 'Outcome' is derived
                del record['target']
        processed_records.append(record)

    new_df = pd.DataFrame(processed_records)
    
    try:
        # --- Update client-specific CSV ---
        hash_cols = HEART_FEATURES + ['Outcome']
        
        if os.path.exists(client_data_file):
            existing_client_df = pd.read_csv(client_data_file)
            
            for col in hash_cols:
                if col not in new_df.columns:
                    new_df[col] = 0
            new_df_hashes = new_df[hash_cols].astype(str).agg(''.join, axis=1).apply(hash)

            # Ensure existing_client_df also has all hash_cols for consistent hashing
            for col in hash_cols:
                if col not in existing_client_df.columns:
                    existing_client_df[col] = 0
            existing_client_hashes = existing_client_df[hash_cols].astype(str).agg(''.join, axis=1).apply(hash)

            unique_new_records_df = new_df[~new_df_hashes.isin(existing_client_hashes)]

            if not unique_new_records_df.empty:
                # Ensure combined columns are consistent
                combined_columns = list(set(existing_client_df.columns).union(unique_new_records_df.columns))
                existing_client_df = existing_client_df.reindex(columns=combined_columns, fill_value=0)
                unique_new_records_df = unique_new_records_df.reindex(columns=combined_columns, fill_value=0)
                
                updated_client_df = pd.concat([existing_client_df, unique_new_records_df], ignore_index=True)
                updated_client_df.to_csv(client_data_file, index=False)
                log.info(f"{len(unique_new_records_df)} unique new records added to {client_data_file} for client {client_id} (Heart).")
                connected_heart_clients_info[client_id]['data_size'] = len(updated_client_df) # Update data size
            else:
                log.info(f"No new unique records to add for client {client_id} (Heart). Data size: {len(existing_client_df)}")
                connected_heart_clients_info[client_id]['data_size'] = len(existing_client_df)
                # No return here, continue to master file logic even if client's local file didn't change from this upload

        else:
            new_df.to_csv(client_data_file, index=False)
            log.info(f"{len(new_records)} new records created for {client_data_file} for client {client_id} (Heart).")
            connected_heart_clients_info[client_id]['data_size'] = len(new_df) # Set initial data size

        # --- Update central MASTER_HEART_AGGREGATED_DATA_FILE ---
        log.info(f"Attempting to update master aggregated data for Heart. Records from client {client_id} to consider: {len(new_df)}")
        if os.path.exists(MASTER_HEART_AGGREGATED_DATA_FILE):
            master_df = pd.read_csv(MASTER_HEART_AGGREGATED_DATA_FILE)
            
            for col in hash_cols:
                if col not in master_df.columns:
                    master_df[col] = 0
            master_df_hashes = master_df[hash_cols].astype(str).agg(''.join, axis=1).apply(hash)

            for col in hash_cols:
                if col not in new_df.columns:
                    new_df[col] = 0 
            new_df_hashes_for_master = new_df[hash_cols].astype(str).agg(''.join, axis=1).apply(hash)

            unique_new_records_for_master = new_df[~new_df_hashes_for_master.isin(master_df_hashes)]
            
            log.info(f"Number of unique new records for MASTER_HEART_AGGREGATED_DATA_FILE from client {client_id}: {len(unique_new_records_for_master)}")
            if not unique_new_records_for_master.empty:
                all_master_columns = list(set(master_df.columns).union(unique_new_records_for_master.columns))
                master_df = master_df.reindex(columns=all_master_columns, fill_value=0)
                unique_new_records_for_master = unique_new_records_for_master.reindex(columns=all_master_columns, fill_value=0)

                combined_master_df = pd.concat([master_df, unique_new_records_for_master], ignore_index=True)
                combined_master_df.to_csv(MASTER_HEART_AGGREGATED_DATA_FILE, index=False)
                log.info(f"{len(unique_new_records_for_master)} unique new records also added to {MASTER_HEART_AGGREGATED_DATA_FILE}.")
            else:
                log.info(f"No new unique records from client {client_id} to add to master aggregated Heart data.")
        else:
            new_df.to_csv(MASTER_HEART_AGGREGATED_DATA_FILE, index=False)
            log.warning(f"{MASTER_HEART_AGGREGATED_DATA_FILE} was not found, created it with new data from client {client_id}.")

        # --- Flower Client Management ---
        with flower_heart_client_lock:
            if client_id not in flower_heart_client_instances:
                flower_heart_client_instances[client_id] = SklearnHeartFlowerClient(client_id, client_data_file)
                log.info(f"New Flower client instance created for Heart ID: {client_id}.")
                heart_flower_client_executor.submit(connect_flower_client, client_id, FLOWER_HEART_GRPC_PORT, heart_client_fn, "Heart")
            else:
                # Explicitly reload data for existing client instance to ensure it has the latest data
                flower_heart_client_instances[client_id].X, flower_heart_client_instances[client_id].y = flower_heart_client_instances[client_id]._load_data()
                log.info(f"Refreshed data for existing Flower Heart client instance {client_id}.")

    except Exception as e:
        log.error(f"Error handling data upload for Heart client {client_id}: {e}", exc_info=True)
        return jsonify({"status": "error", "message": f"Server internal error: {str(e)}"}), 500

    return jsonify({"status": "success", "message": "Heart data received and saved, client registered/updated for FL."})


# Flask App Routes for Parkinson's Disease Model
@app.route('/get_parkinsons_model', methods=['GET'])
def get_parkinsons_model():
    """Provides the latest global federated model for download for Parkinson's Disease."""
    client_ip = request.remote_addr # Get client IP for ZDT
    with zdt_lock:
        if client_ip in blocked_ips and time.time() < blocked_ips[client_ip]:
            log.warning(f"Blocked request from {client_ip} (Parkinson's model download).")
            return jsonify({"status": "error", "message": "Your IP has been temporarily blocked due to suspicious activity."}), 403

    with global_parkinsons_model_lock:
        if os.path.exists(PARKINSONS_GLOBAL_MODEL_FILE):
            try:
                return send_file(PARKINSONS_GLOBAL_MODEL_FILE, 
                                 as_attachment=True, 
                                 download_name=f'global_parkinsons_model_v{global_parkinsons_model_version}.pkl',
                                 mimetype='application/octet-stream')
            except Exception as e:
                log.error(f"Error sending Parkinson's model file: {e}")
                return jsonify({"status": "error", "message": f"Server error sending Parkinson's model: {str(e)}"}), 500
        else:
            log.warning("Global Parkinson's model not found. This might happen initially before the first Flower round or if Flower server hasn't saved it yet.")
            return jsonify({"status": "error", "message": "Global Parkinson's model not available yet. Please try again later."}), 404

@app.route('/get_parkinsons_model_info', methods=['GET'])
def get_parkinsons_model_info():
    """Provides metadata about the current global Parkinson's Disease model."""
    client_ip = request.remote_addr # Get client IP for ZDT
    with zdt_lock:
        if client_ip in blocked_ips and time.time() < blocked_ips[client_ip]:
            log.warning(f"Blocked request from {client_ip} (Parkinson's model info).")
            return jsonify({"status": "error", "message": "Your IP has been temporarily blocked due to suspicious activity."}), 403

    with global_parkinsons_model_lock:
        return jsonify({
            "status": "success",
            "model_type": "Parkinsons Disease",
            "model_available": os.path.exists(PARKINSONS_GLOBAL_MODEL_FILE), # Corrected: Use os.path.exists
            "version": global_parkinsons_model_version,
            "fl_status": parkinsons_fl_training_status
        })

@app.route('/upload_parkinsons_client_data', methods=['POST'])
def upload_parkinsons_client_data():
    """Receives and processes new data from clients for Parkinson's Disease model."""
    client_ip = request.remote_addr
    
    # ZDT: Check if IP is blocked
    with zdt_lock:
        if client_ip in blocked_ips and time.time() < blocked_ips[client_ip]:
            log.warning(f"Blocked request from {client_ip} (Parkinson's upload).")
            return jsonify({"status": "error", "message": "Your IP has been temporarily blocked due to suspicious activity."}), 403
        
        # ZDT: Rate limiting
        current_time = time.time()
        if current_time - request_counts[client_ip]['timestamp'] > 60: # Reset count after 1 minute
            request_counts[client_ip]['count'] = 1
            request_counts[client_ip]['timestamp'] = current_time
        else:
            request_counts[client_ip]['count'] += 1
            if request_counts[client_ip]['count'] > ATTACK_THRESHOLD_PER_MINUTE:
                blocked_ips[client_ip] = current_time + BLOCK_DURATION_SECONDS
                log_attack("RateLimitExceeded", client_ip, f"Too many requests ({request_counts[client_ip]['count']}) in 1 minute.")
                log.error(f"Blocking IP {client_ip} for {BLOCK_DURATION_SECONDS} seconds due to rate limit.")
                return jsonify({"status": "error", "message": "Too many requests. Your IP has been temporarily blocked."}), 429 # Too Many Requests

    # ZDT: Check if request is JSON
    if not request.is_json:
        log_attack("InvalidPayloadFormat", client_ip, "Request not JSON.")
        return jsonify({"status": "error", "message": "Request must be JSON"}), 400

    data = request.get_json()
    client_id = data.get('client_id')
    new_records = data.get('records')

    # ZDT: Check for missing essential data
    if not client_id or not new_records:
        log_attack("MissingData", client_ip, "Missing client_id or records in payload.")
        return jsonify({"status": "error", "message": "Missing client_id or records"}), 400

    # ZDT: Basic data validation for Parkinson's features and Outcome
    PARKINSONS_FEATURES_AND_OUTCOME = PARKINSONS_FEATURES + ['Outcome']
    for record in new_records:
        # Check for expected keys
        if not all(k in record for k in PARKINSONS_FEATURES_AND_OUTCOME):
            log_attack("MalformedRecord", client_ip, f"Missing expected features in Parkinson's record: {record}")
            return jsonify({"status": "error", "message": "Malformed data record detected (missing features)."}), 400
        
        try:
            # Outcome should be 0 or 1
            if 'Outcome' in record and record['Outcome'] not in [0, 1]:
                log_attack("InvalidOutcome", client_ip, f"Invalid 'Outcome' value in Parkinson's record: {record}")
                return jsonify({"status": "error", "message": "Invalid outcome value detected."}), 400

        except (ValueError, TypeError):
            log_attack("InvalidDataType", client_ip, f"Non-numeric data in Parkinson's record: {record}")
            return jsonify({"status": "error", "message": "Data contains non-numeric values."}), 400

    log.info(f"Received {len(new_records)} new records from client {client_id} for Parkinson's Disease.")

    # Update client liveness information
    connected_parkinsons_clients_info[client_id]['last_seen'] = datetime.datetime.now().isoformat()
    connected_parkinsons_clients_info[client_id]['ip'] = request.remote_addr
    
    client_data_file = os.path.join(CLIENT_PARKINSONS_DATA_DIR, f"client_parkinsons_data_{client_id}.csv")

    processed_records = []
    for record in new_records:
        # Basic validation and standardization of 'Outcome'
        if 'Outcome' not in record:
            outcome_val = record.get('result', '').lower()
            if outcome_val == 'yes':
                record['Outcome'] = 1
            elif outcome_val == 'no':
                record['Outcome'] = 0
            else:
                log.warning(f"Client {client_id} (Parkinson's): Invalid or missing 'Outcome'/'result' for a record. Defaulting to 0. Record: {record}")
                record['Outcome'] = 0 
            if 'result' in record:
                del record['result'] # Remove 'result' if 'Outcome' is derived
        processed_records.append(record)

    new_df = pd.DataFrame(processed_records)
    
    try:
        # --- Update client-specific CSV ---
        hash_cols = PARKINSONS_FEATURES + ['Outcome']
        
        if os.path.exists(client_data_file):
            existing_client_df = pd.read_csv(client_data_file)
            
            for col in hash_cols:
                if col not in new_df.columns:
                    new_df[col] = 0
            new_df_hashes = new_df[hash_cols].astype(str).agg(''.join, axis=1).apply(hash)
            
            # Ensure existing_client_df also has all hash_cols for consistent hashing
            for col in hash_cols:
                if col not in existing_client_df.columns:
                    existing_client_df[col] = 0
            existing_client_hashes = existing_client_df[hash_cols].astype(str).agg(''.join, axis=1).apply(hash)

            unique_new_records_df = new_df[~new_df_hashes.isin(existing_client_hashes)]

            if not unique_new_records_df.empty:
                # Ensure combined columns are consistent
                combined_columns = list(set(existing_client_df.columns).union(unique_new_records_df.columns))
                existing_client_df = existing_client_df.reindex(columns=combined_columns, fill_value=0)
                unique_new_records_df = unique_new_records_df.reindex(columns=combined_columns, fill_value=0)
                
                updated_client_df = pd.concat([existing_client_df, unique_new_records_df], ignore_index=True)
                updated_client_df.to_csv(client_data_file, index=False)
                log.info(f"{len(unique_new_records_df)} unique new records added to {client_data_file} for client {client_id} (Parkinson's).")
                connected_parkinsons_clients_info[client_id]['data_size'] = len(updated_client_df) # Update data size
            else:
                log.info(f"No new unique records to add for client {client_id} (Parkinson's). Data size: {len(existing_client_df)}")
                connected_parkinsons_clients_info[client_id]['data_size'] = len(existing_client_df)
                # No return here, continue to master file logic even if client's local file didn't change from this upload

        else:
            new_df.to_csv(client_data_file, index=False)
            log.info(f"{len(new_records)} new records created for {client_data_file} for client {client_id} (Parkinson's).")
            connected_parkinsons_clients_info[client_id]['data_size'] = len(new_df) # Set initial data size

        # --- Update central MASTER_PARKINSONS_AGGREGATED_DATA_FILE ---
        log.info(f"Attempting to update master aggregated data for Parkinson's. Records from client {client_id} to consider: {len(new_df)}")
        if os.path.exists(MASTER_PARKINSONS_AGGREGATED_DATA_FILE):
            master_df = pd.read_csv(MASTER_PARKINSONS_AGGREGATED_DATA_FILE)
            
            for col in hash_cols:
                if col not in master_df.columns:
                    master_df[col] = 0
            master_df_hashes = master_df[hash_cols].astype(str).agg(''.join, axis=1).apply(hash)
            
            for col in hash_cols:
                if col not in new_df.columns:
                    new_df[col] = 0 
            new_df_hashes_for_master = new_df[hash_cols].astype(str).agg(''.join, axis=1).apply(hash)

            unique_new_records_for_master = new_df[~new_df_hashes_for_master.isin(master_df_hashes)]
            
            log.info(f"Number of unique new records for MASTER_PARKINSONS_AGGREGATED_DATA_FILE from client {client_id}: {len(unique_new_records_for_master)}")
            if not unique_new_records_for_master.empty:
                all_master_columns = list(set(master_df.columns).union(unique_new_records_for_master.columns))
                master_df = master_df.reindex(columns=all_master_columns, fill_value=0)
                unique_new_records_for_master = unique_new_records_for_master.reindex(columns=all_master_columns, fill_value=0)

                combined_master_df = pd.concat([master_df, unique_new_records_for_master], ignore_index=True)
                combined_master_df.to_csv(MASTER_PARKINSONS_AGGREGATED_DATA_FILE, index=False)
                log.info(f"{len(unique_new_records_for_master)} unique new records also added to {MASTER_PARKINSONS_AGGREGATED_DATA_FILE}.")
            else:
                log.info(f"No new unique records from client {client_id} to add to master aggregated Parkinson's data.")
        else:
            new_df.to_csv(MASTER_PARKINSONS_AGGREGATED_DATA_FILE, index=False)
            log.warning(f"{MASTER_PARKINSONS_AGGREGATED_DATA_FILE} was not found, created it with new data from client {client_id}.")

        # --- Flower Client Management ---
        with flower_parkinsons_client_lock:
            if client_id not in flower_parkinsons_client_instances:
                flower_parkinsons_client_instances[client_id] = SklearnParkinsonsFlowerClient(client_id, client_data_file)
                log.info(f"New Flower client instance created for Parkinson's ID: {client_id}.")
                parkinsons_flower_client_executor.submit(connect_flower_client, client_id, FLOWER_PARKINSONS_GRPC_PORT, parkinsons_client_fn, "Parkinsons")
            else:
                # Explicitly reload data for existing client instance to ensure it has the latest data
                flower_parkinsons_client_instances[client_id].X, flower_parkinsons_client_instances[client_id].y = flower_parkinsons_client_instances[client_id]._load_data()
                log.info(f"Refreshed data for existing Flower Parkinson's client instance {client_id}.")

    except Exception as e:
        log.error(f"Error handling data upload for Parkinson's client {client_id}: {e}", exc_info=True)
        return jsonify({"status": "error", "message": f"Server internal error: {str(e)}"}), 500

    return jsonify({"status": "success", "message": "Parkinson's data received and saved, client registered/updated for FL."})


@app.route('/zdt_status', methods=['GET'])
def zdt_status():
    """Provides real-time zero-day threat status to the admin UI."""
    with zdt_lock:
        # Clean up expired blocked IPs
        current_time = time.time()
        expired_ips = [ip for ip, unblock_time in blocked_ips.items() if current_time >= unblock_time]
        for ip in expired_ips:
            del blocked_ips[ip]

        return jsonify({
            "status": "success",
            "attack_counts": dict(attack_counts),
            "blocked_ips": {ip: datetime.datetime.fromtimestamp(unblock_time).isoformat() for ip, unblock_time in blocked_ips.items()},
            "attack_log": attack_log # Consider sending a limited number of recent logs for performance
        })


@app.route('/client_status', methods=['GET'])
def client_status():
    """Returns the status of all connected clients for both models."""
    current_time = datetime.datetime.now()
    all_clients = {}

    # Diabetes Clients
    for client_id, info in connected_diabetes_clients_info.items():
        if info['last_seen']:
            last_seen_dt = datetime.datetime.fromisoformat(info['last_seen'])
            is_active = (current_time - last_seen_dt).total_seconds() < 300 # 5 minutes
            all_clients[f"diabetes_{client_id}"] = {
                "model_type": "Diabetes",
                "last_seen": info['last_seen'],
                "ip": info['ip'],
                "data_size": info['data_size'],
                "is_active": is_active
            }
        else:
            all_clients[f"diabetes_{client_id}"] = {
                "model_type": "Diabetes",
                "last_seen": "Never",
                "ip": info['ip'],
                "data_size": info['data_size'],
                "is_active": False
            }
    
    # Heart Clients
    for client_id, info in connected_heart_clients_info.items():
        if info['last_seen']:
            last_seen_dt = datetime.datetime.fromisoformat(info['last_seen'])
            is_active = (current_time - last_seen_dt).total_seconds() < 300 # 5 minutes
            all_clients[f"heart_{client_id}"] = {
                "model_type": "Heart",
                "last_seen": info['last_seen'],
                "ip": info['ip'],
                "data_size": info['data_size'],
                "is_active": is_active
            }
        else:
            all_clients[f"heart_{client_id}"] = {
                "model_type": "Heart",
                "last_seen": "Never",
                "ip": info['ip'],
                "data_size": info['data_size'],
                "is_active": False
            }

    # Parkinson's Clients
    for client_id, info in connected_parkinsons_clients_info.items():
        if info['last_seen']:
            last_seen_dt = datetime.datetime.fromisoformat(info['last_seen'])
            is_active = (current_time - last_seen_dt).total_seconds() < 300 # 5 minutes
            all_clients[f"parkinsons_{client_id}"] = {
                "model_type": "Parkinsons",
                "last_seen": info['last_seen'],
                "ip": info['ip'],
                "data_size": info['data_size'],
                "is_active": is_active
            }
        else:
            all_clients[f"parkinsons_{client_id}"] = {
                "model_type": "Parkinsons",
                "last_seen": "Never",
                "ip": info['ip'],
                "data_size": info['data_size'],
                "is_active": False
            }

    return jsonify({"status": "success", "clients": all_clients})


@app.route('/fl_status', methods=['GET'])
def fl_status():
    """Returns the current status of Federated Learning training for both models."""
    return jsonify({
        "status": "success", 
        "diabetes_fl_training": diabetes_fl_training_status,
        "heart_fl_training": heart_fl_training_status,
        "parkinsons_fl_training": parkinsons_fl_training_status
    })

# --- Main Server Execution Block ---
if __name__ == '__main__':
    log.info("Starting Federated Learning Server (Flask + Flower for Diabetes, Heart Disease, and Parkinson's Disease).")
    
    # Initialize master CSVs for all models
    initialize_master_aggregated_data(MASTER_DIABETES_AGGREGATED_DATA_FILE, DIABETES_FEATURES + ['Outcome'])
    initialize_master_aggregated_data(MASTER_HEART_AGGREGATED_DATA_FILE, HEART_FEATURES + ['Outcome'])
    initialize_master_aggregated_data(MASTER_PARKINSONS_AGGREGATED_DATA_FILE, PARKINSONS_FEATURES + ['Outcome'])

    # Start the Flower server for Diabetes in a separate thread
    diabetes_flower_strategy = DiabetesFedAvg(
        fraction_fit=1.0, 
        fraction_evaluate=1.0, 
        min_fit_clients=MIN_CLIENTS_PER_ROUND,
        min_evaluate_clients=MIN_CLIENTS_PER_ROUND,
        min_available_clients=MIN_CLIENTS_PER_ROUND,
    )
    diabetes_flower_server_thread = threading.Thread(target=run_flower_server_in_thread, 
                                                     args=(FLOWER_DIABETES_GRPC_PORT, diabetes_flower_strategy, "Diabetes", diabetes_client_fn))
    diabetes_flower_server_thread.daemon = True
    diabetes_flower_server_thread.start()
    log.info(f"Flower server for Diabetes thread starting on port {FLOWER_DIABETES_GRPC_PORT}...")

    heart_flower_strategy = HeartFedAvg(
        fraction_fit=1.0, 
        fraction_evaluate=1.0, 
        min_fit_clients=MIN_CLIENTS_PER_ROUND,
        min_evaluate_clients=MIN_CLIENTS_PER_ROUND,
        min_available_clients=MIN_CLIENTS_PER_ROUND,
    )
    heart_flower_server_thread = threading.Thread(target=run_flower_server_in_thread, 
                                                  args=(FLOWER_HEART_GRPC_PORT, heart_flower_strategy, "Heart", heart_client_fn))
    heart_flower_server_thread.daemon = True
    heart_flower_server_thread.start()
    log.info(f"Flower server for Heart Disease thread starting on port {FLOWER_HEART_GRPC_PORT}...")

    # Start the Flower server for Parkinson's Disease in a separate thread
    parkinsons_flower_strategy = ParkinsonsFedAvg(
        fraction_fit=1.0, 
        fraction_evaluate=1.0, 
        min_fit_clients=MIN_CLIENTS_PER_ROUND,
        min_evaluate_clients=MIN_CLIENTS_PER_ROUND,
        min_available_clients=MIN_CLIENTS_PER_ROUND,
    )
    parkinsons_flower_server_thread = threading.Thread(target=run_flower_server_in_thread, 
                                                        args=(FLOWER_PARKINSONS_GRPC_PORT, parkinsons_flower_strategy, "Parkinsons", parkinsons_client_fn))
    parkinsons_flower_server_thread.daemon = True
    parkinsons_flower_server_thread.start()
    log.info(f"Flower server for Parkinson's Disease thread starting on port {FLOWER_PARKINSONS_GRPC_PORT}...")


    # Give Flower servers a moment to start up (initial sleep, retries handle later connections)
    time.sleep(2) 
    log.info(f"Flower gRPC servers expected to be running on ports {FLOWER_DIABETES_GRPC_PORT} (Diabetes), {FLOWER_HEART_GRPC_PORT} (Heart), and {FLOWER_PARKINSONS_GRPC_PORT} (Parkinson's).")

    # Start the Flask server in the main thread (blocking call)
    log.info(f"Flask server running on http://{SERVER_HOST}:{FLASK_PORT}")
    try:
        app.run(host=SERVER_HOST, port=FLASK_PORT, debug=False, use_reloader=False)
    except (KeyboardInterrupt, SystemExit):
        log.info("Flask server shut down.")
    finally:
        # Ensure thread pools are shut down gracefully
        diabetes_flower_client_executor.shutdown(wait=True)
        heart_flower_client_executor.shutdown(wait=True)
        parkinsons_flower_client_executor.shutdown(wait=True)
        log.info("Main application shutting down.")
