"""
Mental State Classifier for Real-time EEG Analysis
Implements advanced classification algorithms for detecting multiple mental states
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any
import logging
from collections import deque
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os


class MentalStateClassifier:
    """
    Advanced mental state classifier using multiple machine learning approaches
    """
    
    def __init__(self, model_type: str = "ensemble"):
        """
        Initialize mental state classifier
        
        Args:
            model_type: Type of classifier ("ensemble", "random_forest", "svm", "logistic")
        """
        self.model_type = model_type
        self.logger = logging.getLogger(__name__)
        
        # Define mental states
        self.mental_states = {
            0: "neutral",
            1: "focused", 
            2: "relaxed",
            3: "meditative",
            4: "alert",
            5: "stressed"
        }
        
        # Model configurations
        self.models = {}
        self.feature_scalers = {}
        self.is_trained = False
        
        # Training data storage
        self.training_features = []
        self.training_labels = []
        
        # Real-time classification parameters
        self.classification_window = 2.0  # seconds
        self.confidence_threshold = 0.6
        self.temporal_smoothing = True
        self.smoothing_window = 5  # number of predictions to average
        
        # History for temporal analysis
        self.prediction_history = deque(maxlen=50)
        self.confidence_history = deque(maxlen=50)
        self.feature_history = deque(maxlen=100)
        
        # Adaptive learning parameters
        self.adaptive_learning = True
        self.adaptation_rate = 0.1
        self.min_confidence_for_adaptation = 0.8
        
        # Initialize models
        self._initialize_models()
        
        # State transition matrix for temporal consistency
        self.transition_matrix = self._initialize_transition_matrix()
        
    def _initialize_models(self):
        """Initialize machine learning models"""
        try:
            # Random Forest - good for feature importance and robustness
            self.models['random_forest'] = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'
            )
            
            # Support Vector Machine - good for high-dimensional data
            self.models['svm'] = SVC(
                kernel='rbf',
                gamma='scale',
                C=1.0,
                probability=True,  # Enable probability predictions
                random_state=42,
                class_weight='balanced'
            )
            
            # Logistic Regression - fast and interpretable
            self.models['logistic'] = LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced',
                multi_class='ovr'
            )
            
            # Initialize feature scalers for each model
            for model_name in self.models.keys():
                self.feature_scalers[model_name] = StandardScaler()
            
            self.logger.info(f"Initialized {len(self.models)} classification models")
            
        except Exception as e:
            self.logger.error(f"Error initializing models: {e}")
    
    def _initialize_transition_matrix(self) -> np.ndarray:
        """Initialize state transition matrix for temporal consistency"""
        n_states = len(self.mental_states)
        
        # Create transition matrix with higher probability for staying in same state
        transition_matrix = np.full((n_states, n_states), 0.05)  # Low probability for all transitions
        
        # Higher probability for staying in the same state
        np.fill_diagonal(transition_matrix, 0.7)
        
        # Normalize rows to sum to 1
        transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)
        
        return transition_matrix
    
    def extract_classification_features(self, spectral_features: Dict[str, Any],
                                     temporal_features: Dict[str, Any],
                                     cross_features: Dict[str, Any]) -> np.ndarray:
        """
        Extract and combine features for classification
        
        Args:
            spectral_features: Spectral domain features from signal processor
            temporal_features: Temporal domain features from signal processor
            cross_features: Cross-channel features from signal processor
            
        Returns:
            Feature vector for classification
        """
        try:
            feature_vector = []
            
            # Extract key spectral features across channels
            for channel_name in ['TP9', 'AF7', 'AF8', 'TP10']:
                if channel_name in spectral_features:
                    channel_data = spectral_features[channel_name]
                    
                    # Power in each frequency band
                    for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
                        feature_vector.append(channel_data.get(band, 0.0))
                    
                    # Relative power
                    for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
                        feature_vector.append(channel_data.get(f"{band}_rel", 0.0))
                    
                    # Peak frequencies
                    for band in ['alpha', 'beta', 'theta']:
                        feature_vector.append(channel_data.get(f"{band}_peak", 0.0))
                    
                    # Spectral characteristics
                    feature_vector.append(channel_data.get('spectral_edge', 0.0))
                    feature_vector.append(channel_data.get('spectral_entropy', 0.0))
                    feature_vector.append(channel_data.get('total_power', 0.0))
            
            # Extract key temporal features across channels
            for channel_name in ['TP9', 'AF7', 'AF8', 'TP10']:
                if channel_name in temporal_features:
                    channel_data = temporal_features[channel_name]
                    
                    # Statistical moments
                    feature_vector.append(channel_data.get('std', 0.0))
                    feature_vector.append(channel_data.get('skewness', 0.0))
                    feature_vector.append(channel_data.get('kurtosis', 0.0))
                    
                    # Signal complexity
                    feature_vector.append(channel_data.get('hjorth_activity', 0.0))
                    feature_vector.append(channel_data.get('hjorth_mobility', 0.0))
                    feature_vector.append(channel_data.get('hjorth_complexity', 0.0))
                    
                    # Amplitude features
                    feature_vector.append(channel_data.get('rms', 0.0))
                    feature_vector.append(channel_data.get('zero_crossings', 0.0))
            
            # Cross-channel features
            feature_vector.append(cross_features.get('frontal_asymmetry', 0.0))
            feature_vector.append(cross_features.get('temporal_asymmetry', 0.0))
            
            # Correlation features
            for key, value in cross_features.items():
                if 'corr' in key:
                    feature_vector.append(value)
            
            # Convert to numpy array and handle any NaN values
            feature_array = np.array(feature_vector, dtype=np.float32)
            feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=0.0, neginf=0.0)
            
            return feature_array
            
        except Exception as e:
            self.logger.error(f"Error extracting classification features: {e}")
            return np.array([])
    
    def add_training_sample(self, spectral_features: Dict[str, Any],
                          temporal_features: Dict[str, Any],
                          cross_features: Dict[str, Any],
                          mental_state: str):
        """
        Add a labeled training sample
        
        Args:
            spectral_features: Spectral features
            temporal_features: Temporal features
            cross_features: Cross-channel features
            mental_state: True mental state label
        """
        try:
            # Extract feature vector
            features = self.extract_classification_features(
                spectral_features, temporal_features, cross_features
            )
            
            if len(features) > 0:
                # Convert state name to label
                state_label = self._state_name_to_label(mental_state)
                
                if state_label is not None:
                    self.training_features.append(features)
                    self.training_labels.append(state_label)
                    
                    self.logger.debug(f"Added training sample: {mental_state} (label: {state_label})")
                else:
                    self.logger.warning(f"Unknown mental state: {mental_state}")
            
        except Exception as e:
            self.logger.error(f"Error adding training sample: {e}")
    
    def _state_name_to_label(self, state_name: str) -> Optional[int]:
        """Convert state name to numerical label"""
        state_mapping = {v: k for k, v in self.mental_states.items()}
        return state_mapping.get(state_name.lower())
    
    def _label_to_state_name(self, label: int) -> str:
        """Convert numerical label to state name"""
        return self.mental_states.get(label, "unknown")
    
    def train_models(self) -> Dict[str, float]:
        """
        Train all classification models
        
        Returns:
            Dictionary with training scores for each model
        """
        if len(self.training_features) < 10:
            self.logger.warning("Insufficient training data. Need at least 10 samples.")
            return {}
        
        try:
            # Convert training data to arrays
            X = np.array(self.training_features)
            y = np.array(self.training_labels)
            
            training_scores = {}
            
            for model_name, model in self.models.items():
                self.logger.info(f"Training {model_name} model...")
                
                # Scale features
                X_scaled = self.feature_scalers[model_name].fit_transform(X)
                
                # Train model
                model.fit(X_scaled, y)
                
                # Evaluate with cross-validation
                cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
                training_scores[model_name] = np.mean(cv_scores)
                
                self.logger.info(f"{model_name} CV accuracy: {training_scores[model_name]:.3f} ± {np.std(cv_scores):.3f}")
            
            self.is_trained = True
            self.logger.info("Model training completed")
            
            return training_scores
            
        except Exception as e:
            self.logger.error(f"Error training models: {e}")
            return {}
    
    def predict_mental_state(self, spectral_features: Dict[str, Any],
                           temporal_features: Dict[str, Any],
                           cross_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict mental state from features
        
        Args:
            spectral_features: Spectral features
            temporal_features: Temporal features  
            cross_features: Cross-channel features
            
        Returns:
            Dictionary with prediction results
        """
        if not self.is_trained:
            return self._rule_based_classification(spectral_features, temporal_features, cross_features)
        
        try:
            # Extract features
            features = self.extract_classification_features(
                spectral_features, temporal_features, cross_features
            )
            
            if len(features) == 0:
                return {'state': 'unknown', 'confidence': 0.0}
            
            # Store features for history
            self.feature_history.append(features)
            
            # Predictions from all models
            model_predictions = {}
            model_confidences = {}
            
            for model_name, model in self.models.items():
                try:
                    # Scale features
                    features_scaled = self.feature_scalers[model_name].transform(features.reshape(1, -1))
                    
                    # Predict
                    prediction = model.predict(features_scaled)[0]
                    
                    # Get prediction probabilities
                    if hasattr(model, 'predict_proba'):
                        probabilities = model.predict_proba(features_scaled)[0]
                        confidence = np.max(probabilities)
                        state_probabilities = {
                            self._label_to_state_name(i): prob 
                            for i, prob in enumerate(probabilities)
                        }
                    else:
                        confidence = 0.5  # Default confidence
                        state_probabilities = {}
                    
                    model_predictions[model_name] = {
                        'state': self._label_to_state_name(prediction),
                        'confidence': confidence,
                        'probabilities': state_probabilities
                    }
                    model_confidences[model_name] = confidence
                    
                except Exception as e:
                    self.logger.error(f"Error in {model_name} prediction: {e}")
                    continue
            
            # Ensemble prediction
            final_prediction = self._ensemble_prediction(model_predictions)
            
            # Apply temporal smoothing if enabled
            if self.temporal_smoothing:
                final_prediction = self._apply_temporal_smoothing(final_prediction)
            
            # Store prediction history
            self.prediction_history.append(final_prediction)
            self.confidence_history.append(final_prediction.get('confidence', 0.0))
            
            # Add temporal consistency score
            consistency_score = self._calculate_prediction_consistency()
            final_prediction['temporal_consistency'] = consistency_score
            
            # Add model-specific predictions for debugging
            final_prediction['model_predictions'] = model_predictions
            
            return final_prediction
            
        except Exception as e:
            self.logger.error(f"Error in mental state prediction: {e}")
            return {'state': 'unknown', 'confidence': 0.0}
    
    def _ensemble_prediction(self, model_predictions: Dict[str, Dict]) -> Dict[str, Any]:
        """Combine predictions from multiple models"""
        if not model_predictions:
            return {'state': 'unknown', 'confidence': 0.0}
        
        # Weighted voting based on model confidence
        state_votes = {}
        total_weight = 0.0
        
        for model_name, prediction in model_predictions.items():
            state = prediction['state']
            confidence = prediction['confidence']
            
            # Weight by confidence
            weight = confidence
            total_weight += weight
            
            if state in state_votes:
                state_votes[state] += weight
            else:
                state_votes[state] = weight
        
        # Normalize votes
        if total_weight > 0:
            state_votes = {state: vote/total_weight for state, vote in state_votes.items()}
        
        # Select state with highest vote
        if state_votes:
            final_state = max(state_votes, key=state_votes.get)
            final_confidence = state_votes[final_state]
        else:
            final_state = 'unknown'
            final_confidence = 0.0
        
        # Aggregate probabilities
        all_probabilities = {}
        for prediction in model_predictions.values():
            for state, prob in prediction.get('probabilities', {}).items():
                if state in all_probabilities:
                    all_probabilities[state].append(prob)
                else:
                    all_probabilities[state] = [prob]
        
        # Average probabilities across models
        averaged_probabilities = {
            state: np.mean(probs) for state, probs in all_probabilities.items()
        }
        
        return {
            'state': final_state,
            'confidence': final_confidence,
            'state_votes': state_votes,
            'probabilities': averaged_probabilities,
            'timestamp': time.time()
        }
    
    def _apply_temporal_smoothing(self, current_prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Apply temporal smoothing to reduce prediction jitter"""
        if len(self.prediction_history) < 2:
            return current_prediction
        
        # Get recent predictions
        recent_predictions = list(self.prediction_history)[-self.smoothing_window:]
        
        # Count state occurrences
        state_counts = {}
        confidence_sum = 0.0
        
        for pred in recent_predictions:
            state = pred.get('state', 'unknown')
            confidence = pred.get('confidence', 0.0)
            
            if state in state_counts:
                state_counts[state] += 1
            else:
                state_counts[state] = 1
                
            confidence_sum += confidence
        
        # Add current prediction
        current_state = current_prediction.get('state', 'unknown')
        current_confidence = current_prediction.get('confidence', 0.0)
        
        if current_state in state_counts:
            state_counts[current_state] += 1
        else:
            state_counts[current_state] = 1
        confidence_sum += current_confidence
        
        # Select most frequent state
        smoothed_state = max(state_counts, key=state_counts.get)
        smoothed_confidence = confidence_sum / (len(recent_predictions) + 1)
        
        # Keep other information from current prediction
        smoothed_prediction = current_prediction.copy()
        smoothed_prediction['state'] = smoothed_state
        smoothed_prediction['confidence'] = smoothed_confidence
        smoothed_prediction['smoothed'] = True
        
        return smoothed_prediction
    
    def _calculate_prediction_consistency(self) -> float:
        """Calculate consistency of recent predictions"""
        if len(self.prediction_history) < 2:
            return 0.0
        
        recent_states = [pred.get('state', 'unknown') for pred in list(self.prediction_history)[-10:]]
        
        if not recent_states:
            return 0.0
        
        # Calculate consistency as percentage of most common state
        most_common_state = max(set(recent_states), key=recent_states.count)
        consistency = recent_states.count(most_common_state) / len(recent_states)
        
        return consistency
    
    def _rule_based_classification(self, spectral_features: Dict[str, Any],
                                 temporal_features: Dict[str, Any],
                                 cross_features: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback rule-based classification when no trained model is available"""
        try:
            # Calculate key ratios from spectral features
            total_alpha = 0.0
            total_beta = 0.0
            total_theta = 0.0
            total_gamma = 0.0
            
            channel_count = 0
            
            for channel_name in ['TP9', 'AF7', 'AF8', 'TP10']:
                if channel_name in spectral_features:
                    channel_data = spectral_features[channel_name]
                    total_alpha += channel_data.get('alpha', 0.0)
                    total_beta += channel_data.get('beta', 0.0)
                    total_theta += channel_data.get('theta', 0.0)
                    total_gamma += channel_data.get('gamma', 0.0)
                    channel_count += 1
            
            if channel_count == 0:
                return {'state': 'unknown', 'confidence': 0.0}
            
            # Average across channels
            avg_alpha = total_alpha / channel_count
            avg_beta = total_beta / channel_count
            avg_theta = total_theta / channel_count
            avg_gamma = total_gamma / channel_count
            
            # Calculate ratios
            alpha_beta_ratio = avg_alpha / (avg_beta + 1e-8)
            theta_alpha_ratio = avg_theta / (avg_alpha + 1e-8)
            gamma_beta_ratio = avg_gamma / (avg_beta + 1e-8)
            
            # Rule-based classification
            if alpha_beta_ratio > 1.5:
                state = 'relaxed'
                confidence = min(alpha_beta_ratio / 3.0, 1.0)
            elif alpha_beta_ratio < 0.5:
                state = 'focused'
                confidence = min((1.0 / alpha_beta_ratio) / 3.0, 1.0)
            elif theta_alpha_ratio > 1.0:
                state = 'meditative'
                confidence = min(theta_alpha_ratio / 2.0, 1.0)
            elif gamma_beta_ratio > 0.5:
                state = 'alert'
                confidence = min(gamma_beta_ratio / 1.0, 1.0)
            else:
                state = 'neutral'
                confidence = 0.5
            
            return {
                'state': state,
                'confidence': confidence,
                'rule_based': True,
                'ratios': {
                    'alpha_beta': alpha_beta_ratio,
                    'theta_alpha': theta_alpha_ratio,
                    'gamma_beta': gamma_beta_ratio
                },
                'timestamp': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Error in rule-based classification: {e}")
            return {'state': 'unknown', 'confidence': 0.0}
    
    def get_feature_importance(self, model_name: str = 'random_forest') -> Dict[str, float]:
        """Get feature importance from trained model"""
        if not self.is_trained or model_name not in self.models:
            return {}
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            # Create feature names (simplified)
            n_features = len(model.feature_importances_)
            feature_names = [f'feature_{i}' for i in range(n_features)]
            
            importance_dict = dict(zip(feature_names, model.feature_importances_))
            
            # Sort by importance
            sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            
            return dict(sorted_importance)
        
        return {}
    
    def save_model(self, file_path: str):
        """Save trained models to file"""
        if not self.is_trained:
            self.logger.warning("No trained models to save")
            return
        
        try:
            model_data = {
                'models': self.models,
                'feature_scalers': self.feature_scalers,
                'mental_states': self.mental_states,
                'is_trained': self.is_trained,
                'training_features': self.training_features,
                'training_labels': self.training_labels
            }
            
            with open(file_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            self.logger.info(f"Models saved to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving models: {e}")
    
    def load_model(self, file_path: str) -> bool:
        """Load trained models from file"""
        if not os.path.exists(file_path):
            self.logger.warning(f"Model file not found: {file_path}")
            return False
        
        try:
            with open(file_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.models = model_data['models']
            self.feature_scalers = model_data['feature_scalers']
            self.mental_states = model_data['mental_states']
            self.is_trained = model_data['is_trained']
            self.training_features = model_data.get('training_features', [])
            self.training_labels = model_data.get('training_labels', [])
            
            self.logger.info(f"Models loaded from {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            return False
    
    def get_classification_report(self) -> str:
        """Get detailed classification report"""
        if not self.is_trained or len(self.training_features) == 0:
            return "No training data available for report"
        
        try:
            X = np.array(self.training_features)
            y = np.array(self.training_labels)
            
            report = "=== Mental State Classification Report ===\n\n"
            
            for model_name, model in self.models.items():
                # Scale features
                X_scaled = self.feature_scalers[model_name].transform(X)
                
                # Predict
                y_pred = model.predict(X_scaled)
                
                # Get target names
                target_names = [self._label_to_state_name(i) for i in range(len(self.mental_states))]
                
                report += f"\n{model_name.upper()} MODEL:\n"
                report += classification_report(y, y_pred, target_names=target_names)
                report += "\n" + "="*50 + "\n"
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating classification report: {e}")
            return f"Error generating report: {e}"


def main():
    """
    Test function for mental state classifier
    """
    print("=== Mental State Classifier Test ===")
    
    # Create classifier
    classifier = MentalStateClassifier()
    
    # Generate some test data
    print("Generating test training data...")
    
    # Simulate different mental states with characteristic features
    test_states = ['relaxed', 'focused', 'meditative', 'alert', 'neutral']
    
    for state in test_states:
        for _ in range(10):  # 10 samples per state
            # Generate mock features
            spectral_features = {
                'TP9': {'alpha': np.random.rand(), 'beta': np.random.rand(), 'theta': np.random.rand()},
                'AF7': {'alpha': np.random.rand(), 'beta': np.random.rand(), 'theta': np.random.rand()},
                'AF8': {'alpha': np.random.rand(), 'beta': np.random.rand(), 'theta': np.random.rand()},
                'TP10': {'alpha': np.random.rand(), 'beta': np.random.rand(), 'theta': np.random.rand()}
            }
            
            temporal_features = {
                'TP9': {'std': np.random.rand(), 'skewness': np.random.rand()},
                'AF7': {'std': np.random.rand(), 'skewness': np.random.rand()},
                'AF8': {'std': np.random.rand(), 'skewness': np.random.rand()},
                'TP10': {'std': np.random.rand(), 'skewness': np.random.rand()}
            }
            
            cross_features = {'frontal_asymmetry': np.random.rand(), 'temporal_asymmetry': np.random.rand()}
            
            classifier.add_training_sample(spectral_features, temporal_features, cross_features, state)
    
    print(f"Added {len(classifier.training_features)} training samples")
    
    # Train models
    print("\nTraining models...")
    training_scores = classifier.train_models()
    
    for model_name, score in training_scores.items():
        print(f"{model_name}: {score:.3f}")
    
    # Test prediction
    print("\nTesting prediction...")
    test_spectral = {
        'TP9': {'alpha': 0.8, 'beta': 0.3, 'theta': 0.4},
        'AF7': {'alpha': 0.7, 'beta': 0.4, 'theta': 0.3},
        'AF8': {'alpha': 0.9, 'beta': 0.2, 'theta': 0.5},
        'TP10': {'alpha': 0.6, 'beta': 0.5, 'theta': 0.4}
    }
    
    test_temporal = {
        'TP9': {'std': 0.5, 'skewness': 0.1},
        'AF7': {'std': 0.4, 'skewness': 0.2},
        'AF8': {'std': 0.6, 'skewness': 0.0},
        'TP10': {'std': 0.3, 'skewness': 0.3}
    }
    
    test_cross = {'frontal_asymmetry': 0.1, 'temporal_asymmetry': -0.2}
    
    prediction = classifier.predict_mental_state(test_spectral, test_temporal, test_cross)
    
    print(f"Predicted state: {prediction.get('state', 'unknown')}")
    print(f"Confidence: {prediction.get('confidence', 0.0):.3f}")
    
    if 'probabilities' in prediction:
        print("State probabilities:")
        for state, prob in prediction['probabilities'].items():
            print(f"  {state}: {prob:.3f}")
    
    # Get feature importance
    print("\nFeature importance (top 10):")
    importance = classifier.get_feature_importance()
    for i, (feature, score) in enumerate(list(importance.items())[:10]):
        print(f"  {feature}: {score:.3f}")
    
    print("\nMental state classifier test completed!")


if __name__ == "__main__":
    main()
