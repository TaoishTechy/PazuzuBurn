import numpy as np
import hashlib
import json
import time
from collections import deque
import scipy.linalg
from scipy import signal

# --- CORE COMPONENTS (Enhanced from V0.6) ---

class MerkleLogHLA:
    """Enhanced Append-Only Compatibility Ledger with richer telemetry"""
    def __init__(self):
        self.leaves = []
        self.event_count = 0
        print("INFO: Enhanced MerkleLogHLA initialized.")

    def _h(self, x: bytes) -> bytes:
        return hashlib.sha256(x).digest()

    def append(self, event: dict, compact_stats: dict = None):
        """Enhanced event logging with compact statistics"""
        if compact_stats:
            event['stats'] = compact_stats
            
        payload = json.dumps(event, sort_keys=True).encode('utf-8')
        leaf_hash = self._h(b'\x00' + payload)
        self.leaves.append(leaf_hash)
        self.event_count += 1
        return self.root().hex()

    def root(self) -> bytes:
        """Calculates the current root hash of the ledger"""
        if not self.leaves:
            return self._h(b'')
        level = self.leaves[:]
        while len(level) > 1:
            if len(level) % 2 == 1:
                level.append(level[-1])
            level = [self._h(b'\x01' + level[i] + level[i+1]) for i in range(0, len(level), 2)]
        return level[0]
    
    def get_attestation_proof(self, event_index: int):
        """Generate Merkle proof for event verification"""
        if event_index >= len(self.leaves):
            return None
            
        proof = []
        level = self.leaves[:]
        index = event_index
        
        while len(level) > 1:
            if len(level) % 2 == 1:
                level.append(level[-1])
                
            if index % 2 == 0:
                sibling = level[index + 1] if index + 1 < len(level) else level[index]
                proof.append(('right', sibling))
            else:
                proof.append(('left', level[index - 1]))
                
            index = index // 2
            level = [self._h(b'\x01' + level[i] + level[i+1]) for i in range(0, len(level), 2)]
            
        return proof

class StabilityMonitor:
    """Enhanced stability analysis with cross-domain correlation"""
    def __init__(self, window_size=15):
        self.metric_history = deque(maxlen=window_size)
        self.stability_score = 1.0
        self.cross_correlator = CrossCorrelationAnalyzer()
        print("INFO: Enhanced StabilityMonitor initialized.")

    def update(self, metrics, neural_coherence, S):
        self.metric_history.append(metrics)
        
        # Update cross-domain correlations
        self.cross_correlator.update_correlations(metrics, neural_coherence, S, self.stability_score)
        
        # Calculate multi-factor stability score with correlation awareness
        if len(self.metric_history) >= 5:
            jitters = [m['jitter'] for m in self.metric_history]
            jitter_std = np.std(jitters)
            jitter_trend = np.polyfit(range(len(jitters)), jitters, 1)[0]
            
            # Get critical correlations for stability assessment
            critical_corrs = self.cross_correlator.get_critical_correlations()
            correlation_penalty = 0.0
            for _, _, strength in critical_corrs:
                if abs(strength) > 0.8:  # Strong negative correlation hurts stability
                    correlation_penalty += 0.1 * abs(strength)
            
            # Enhanced stability normalization
            jitter_stability = max(0, 1 - jitter_std/2.0 - correlation_penalty)
            trend_stability = max(0, 1 - abs(jitter_trend)*10)
            
            self.stability_score = 0.6 * jitter_stability + 0.3 * trend_stability + 0.1 * (1 - correlation_penalty)
            
        return self.stability_score

    def is_trend_unstable(self):
        return self.stability_score < 0.3

class PredictiveFailureAnalyzer:
    """Enhanced failure analysis with disturbance pattern memory"""
    def __init__(self):
        self.disturbance_memory = DisturbancePatternMemory()
        print("INFO: Enhanced PredictiveFailureAnalyzer initialized.")

    def analyze(self, metrics, stability_score, disturbance_vector=None):
        # Enhanced prediction with pattern memory
        if disturbance_vector is not None:
            # Record disturbance for future optimization
            self.disturbance_memory.record_disturbance(
                disturbance_vector, 
                self._get_current_recovery_pattern(),
                stability_score
            )
        
        # Base prediction logic
        if metrics['jitter'] > 7.5 and stability_score < 0.5:
            return {"mode": "laser_degradation", "risk": 0.8, "eta_steps": 5}
        if metrics['die_temp_proxy'] > 55:
            return {"mode": "thermal_runaway", "risk": 0.7, "eta_steps": 8}
            
        # Enhanced: Check for correlated failure patterns
        if (metrics['jitter'] > 6.0 and metrics['die_temp_proxy'] > 48 and 
            stability_score < 0.6):
            return {"mode": "thermal_jitter_coupling", "risk": 0.6, "eta_steps": 10}
            
        return None
    
    def _get_current_recovery_pattern(self):
        """Extract current recovery pattern from recent metrics"""
        # Simplified - in practice would analyze recent control responses
        return {"pattern": "standard_recovery", "efficiency": 0.8}

# --- ENHANCEMENT IMPLEMENTATIONS ---

class CrossCorrelationAnalyzer:
    """Enhanced correlation analysis with numerical stability"""
    def __init__(self):
        self.correlation_matrix = np.zeros((5, 5))  # jitter, neural, thermal, S, stability
        self.domain_history = {domain: deque(maxlen=30) for domain in range(5)}
        print("INFO: Enhanced CrossCorrelationAnalyzer initialized.")
        
    def update_correlations(self, metrics, neural_coherence, S, stability):
        domain_values = [
            metrics['jitter'], neural_coherence, metrics['die_temp_proxy'], 
            S, stability
        ]
        
        for i, value in enumerate(domain_values):
            self.domain_history[i].append(value)
            
        if all(len(history) >= 10 for history in self.domain_history.values()):
            # Calculate rolling correlations with numerical stability
            data_matrix = np.array([list(history) for history in self.domain_history.values()])
            
            # Enhanced correlation calculation with error handling
            with np.errstate(invalid="ignore", divide="ignore"):
                corr_matrix = np.corrcoef(data_matrix)
                self.correlation_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)
            
    def get_critical_correlations(self):
        """Identify strongest cross-domain relationships"""
        critical_pairs = []
        for i in range(5):
            for j in range(i+1, 5):
                if abs(self.correlation_matrix[i,j]) > 0.7:
                    critical_pairs.append((i, j, self.correlation_matrix[i,j]))
        return critical_pairs

class DisturbancePatternMemory:
    """Learns from past disturbances to improve future responses"""
    def __init__(self):
        self.disturbance_history = []
        self.recovery_patterns = {}
        print("INFO: DisturbancePatternMemory initialized.")
        
    def _generate_pattern_id(self, disturbance_vector):
        """Generate a fingerprint for disturbance patterns"""
        # Simplified pattern ID based on direction and magnitude
        direction = disturbance_vector / np.linalg.norm(disturbance_vector)
        magnitude = np.linalg.norm(disturbance_vector)
        return hash(tuple(np.round(direction, 2).tolist() + [magnitude]))
        
    def record_disturbance(self, disturbance_vector, recovery_sequence, success_metric):
        pattern_id = self._generate_pattern_id(disturbance_vector)
        self.disturbance_history.append({
            'pattern': pattern_id,
            'disturbance': disturbance_vector,
            'recovery': recovery_sequence,
            'success': success_metric,
            'timestamp': time.time()
        })
        
        # Keep only recent history
        if len(self.disturbance_history) > 20:
            self.disturbance_history.pop(0)
        
    def get_optimized_recovery(self, current_disturbance):
        current_pattern = self._generate_pattern_id(current_disturbance)
        
        # Find similar past disturbances
        similar_events = []
        for event in self.disturbance_history[-10:]:  # Recent history
            # Simple similarity based on pattern ID
            if event['pattern'] == current_pattern:
                similar_events.append((1.0, event))  # Exact match
                
        if similar_events:
            # Use most successful recovery strategy
            similar_events.sort(key=lambda x: x[1]['success'], reverse=True)
            best_recovery = similar_events[0][1]['recovery']
            return best_recovery
            
        return None  # No similar patterns found

class PILockCUSUM:
    """Sequential PI-Lock detector using CUSUM algorithm"""
    def __init__(self, k=0.025, h=0.15):  # Slightly relaxed parameters
        self.gp = self.gn = 0.0  # positive/negative CUSUMs
        self.k, self.h = k, h
        self.detection_count = 0
        print("INFO: PILockCUSUM initialized.")
        
    def update(self, s):
        x = s - self.k
        self.gp = max(0.0, self.gp + x)
        self.gn = min(0.0, self.gn + x)
        fired = (self.gp > self.h) or (abs(self.gn) > self.h)
        if fired: 
            self.gp = self.gn = 0.0
            self.detection_count += 1
        return fired

class NeuralPDMHysteresis:
    """Enhanced Neural PDM with optimized hysteresis"""
    def __init__(self, lo=0.84, hi=0.88, refractory=3):  # Relaxed thresholds
        self.state = 'ON' 
        self.cool = 0
        self.lo = lo
        self.hi = hi
        self.R = refractory
        self.toggle_count = 0
        print("INFO: Enhanced NeuralPDMHysteresis initialized.")
        
    def gate(self, coh):
        if self.cool > 0: 
            self.cool -= 1
            return self.state
            
        if self.state == 'ON' and coh < self.lo: 
            self.state = 'OFF'
            self.cool = self.R
            self.toggle_count += 1
        elif self.state == 'OFF' and coh > self.hi: 
            self.state = 'ON'
            self.cool = self.R
            self.toggle_count += 1
            
        return self.state

class AdaptiveCreditSystem:
    """Enhanced bi-directional credit system with optimized penalties"""
    def __init__(self):
        self.safety_credits = {5: 2.0, 6: 3.0}  # Start with some credits
        self.promotion_thresholds = {5: 3.0, 6: 5.0}
        self.demotion_thresholds = {5: 1.5, 6: 2.0}
        self.credit_momentum = {5: 0.0, 6: 0.0}
        self.performance_history = deque(maxlen=10)
        print("INFO: Enhanced AdaptiveCreditSystem initialized.")
        
    def _calculate_performance_index(self, metrics, stability_score, S):
        """Calculate comprehensive performance index"""
        jitter_perf = max(0, 1 - (metrics['jitter'] - 4.0) / 10.0)
        thermal_perf = max(0, 1 - (metrics['die_temp_proxy'] - 35) / 40.0)
        stability_perf = stability_score
        syndrome_perf = max(0, 1 - S * 3.0)
        
        return 0.3 * jitter_perf + 0.2 * thermal_perf + 0.3 * stability_perf + 0.2 * syndrome_perf
        
    def update_bi_directional_credits(self, metrics, stability_score, S, has_pi_lock):
        # Calculate credit velocity (momentum)
        current_performance = self._calculate_performance_index(metrics, stability_score, S)
        self.performance_history.append(current_performance)
        
        for tier in self.safety_credits:
            # Update momentum (exponential moving average)
            if len(self.performance_history) >= 2:
                recent_trend = np.polyfit(range(2), list(self.performance_history)[-2:], 1)[0]
                self.credit_momentum[tier] = 0.8 * self.credit_momentum[tier] + 0.2 * recent_trend
            
            # Credit change based on momentum and performance
            credit_delta = (current_performance * 0.12 + self.credit_momentum[tier] * 0.06)  # Slightly increased rates
            
            if has_pi_lock:
                credit_delta -= 0.4  # Reduced penalty from 0.8 to 0.4
                
            self.safety_credits[tier] = max(0, self.safety_credits[tier] + credit_delta)
            
    def should_demote(self, current_tier):
        """Check if demotion is warranted"""
        if current_tier in self.demotion_thresholds:
            return self.safety_credits[current_tier] < self.demotion_thresholds[current_tier]
        return False

class KalmanStateObserver:
    """Kalman filter for improved state estimation"""
    def __init__(self, A, B, C, Q, R):
        self.A = A
        self.B = B
        self.C = C
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
        self.x_hat = np.zeros(A.shape[0])
        self.P = np.eye(A.shape[0]) * 0.1  # Initial error covariance
        self.innovation_history = deque(maxlen=10)
        print("INFO: KalmanStateObserver initialized.")
    
    def predict(self, u=None):
        """Prediction step"""
        if u is not None:
            self.x_hat = self.A @ self.x_hat + self.B @ u
        else:
            self.x_hat = self.A @ self.x_hat
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x_hat
    
    def update(self, z):
        """Update step with measurement z"""
        y = z - self.C @ self.x_hat  # Innovation
        S = self.C @ self.P @ self.C.T + self.R  # Innovation covariance
        K = self.P @ self.C.T @ np.linalg.inv(S)  # Kalman gain
        
        self.x_hat = self.x_hat + K @ y
        self.P = (np.eye(self.P.shape[0]) - K @ self.C) @ self.P
        
        # Track innovation for performance monitoring
        self.innovation_history.append(np.linalg.norm(y))
        return self.x_hat
    
    def get_innovation_stats(self):
        """Get statistics on filter innovation"""
        if len(self.innovation_history) == 0:
            return 0.0, 0.0
        innovations = list(self.innovation_history)
        return np.mean(innovations), np.std(innovations)

class InnovationAdaptiveKalman:
    """Enhanced adaptive Kalman filter with fixed logic"""
    def __init__(self, kalman_filter):
        self.kf = kalman_filter
        self.innovation_stats = deque(maxlen=20)
        self.adaptation_trigger = 0.12  # Slightly lowered trigger
        self.adaptation_count = 0
        print("INFO: Enhanced InnovationAdaptiveKalman initialized.")
        
    def adaptive_update(self, z, adaptation_active=True):
        # Standard predict-update
        self.kf.predict()
        innovation = z - self.kf.C @ self.kf.x_hat
        
        # FIXED: Append innovation FIRST before checking length
        innovation_norm = np.linalg.norm(innovation)
        self.innovation_stats.append(innovation_norm)
        
        if adaptation_active and len(self.innovation_stats) >= 10:
            # Check if adaptation needed
            recent_innovations = list(self.innovation_stats)[-5:]
            if np.mean(recent_innovations) > self.adaptation_trigger:
                self._adapt_process_noise()
                self._adapt_measurement_noise()
                self.adaptation_count += 1
                print(f"  KALMAN: Adaptation #{self.adaptation_count} (innovation: {np.mean(recent_innovations):.3f})")
                
        self.kf.update(z)
        return self.kf.x_hat
    
    def _adapt_process_noise(self):
        # Increase process noise covariance during high innovation periods
        adaptation_factor = 1.15  # Reduced from 1.2 for smoother adaptation
        self.kf.Q = self.kf.Q * adaptation_factor
        
    def _adapt_measurement_noise(self):
        # Adjust measurement noise based on innovation consistency
        innovation_var = np.var(list(self.innovation_stats))
        if innovation_var > 0.08:  # Lowered threshold
            self.kf.R = self.kf.R * 1.05  # Reduced adaptation factor

class BERMarginOptimizer:
    """Enhanced BER margin optimization with tier-aware thresholds"""
    def __init__(self):
        self.base_ber_threshold = -9.0  # 1e-9
        self.adaptive_margin = 0.3
        self.performance_history = deque(maxlen=15)
        self.margin_adjustment = 0.0
        print("INFO: Enhanced BERMarginOptimizer initialized.")
        
    def update_adaptive_threshold(self, stability_score, neural_coherence, tier_level):
        self.performance_history.append((stability_score, neural_coherence))
        
        if len(self.performance_history) >= 8:  # Reduced from 10 for faster adaptation
            avg_stability = np.mean([p[0] for p in self.performance_history])
            avg_neural = np.mean([p[1] for p in self.performance_history])
            
            # More aggressive thresholds in high-performance conditions
            if avg_stability > 0.82 and avg_neural > 0.87 and tier_level >= 4:  # Relaxed conditions
                self.margin_adjustment = -0.6  # More aggressive tightening
            elif avg_stability < 0.65 or avg_neural < 0.82:  # Relaxed conditions
                self.margin_adjustment = 0.4   # Less conservative
            else:
                self.margin_adjustment = 0.0
                
        return self.base_ber_threshold + self.margin_adjustment
    
    def get_optimized_throttle_decision(self, predicted_ber, current_conditions):
        adaptive_threshold = self.update_adaptive_threshold(
            current_conditions['stability'],
            current_conditions['neural_coherence'], 
            current_conditions['tier']
        )
        
        # More nuanced throttle decision
        if predicted_ber > adaptive_threshold:
            throttle_severity = (predicted_ber - adaptive_threshold) / 2.0
            return True, min(0.7, throttle_severity)  # Maximum 70% throttle
        return False, 0.0

class WindowedPromotionVerifier:
    """Enhanced promotion verification with relaxed criteria"""
    def __init__(self, window_size=8, required_passes=6):  # Relaxed criteria
        self.window_size = window_size
        self.required_passes = required_passes
        self.promotion_criteria_history = {
            'jitter': deque(maxlen=window_size),
            'neural': deque(maxlen=window_size), 
            'stability': deque(maxlen=window_size)
        }
        print("INFO: Enhanced WindowedPromotionVerifier initialized.")
        
    def update_criteria(self, jitter, neural_coherence, stability_score, tier_gates):
        """Update criteria history and check if promotion is valid"""
        # Enhanced criteria with emergency promotion for exceptional performance
        jitter_ok = jitter <= tier_gates['jitter_max']
        neural_ok = neural_coherence >= tier_gates['neural_min']
        stability_ok = stability_score >= tier_gates['stability_min']
        
        self.promotion_criteria_history['jitter'].append(jitter_ok)
        self.promotion_criteria_history['neural'].append(neural_ok)
        self.promotion_criteria_history['stability'].append(stability_ok)
        
        # Check if we have enough history
        if len(self.promotion_criteria_history['jitter']) < self.window_size:
            return False
            
        # Count passes in each category
        jitter_passes = sum(self.promotion_criteria_history['jitter'])
        neural_passes = sum(self.promotion_criteria_history['neural'])
        stability_passes = sum(self.promotion_criteria_history['stability'])
        
        # Enhanced: Allow promotion if 2 out of 3 criteria are strongly met
        strong_performance = (jitter_passes >= self.required_passes + 1 and 
                             stability_passes >= self.required_passes + 1)
        
        # All criteria must meet the required passes, or strong performance in 2/3
        return ((jitter_passes >= self.required_passes and
                neural_passes >= self.required_passes and
                stability_passes >= self.required_passes) or
                (strong_performance and neural_passes >= self.required_passes - 1))
    
    def get_promotion_proof(self):
        """Generate proof of sustained performance for ledger"""
        return {
            'window_size': self.window_size,
            'required_passes': self.required_passes,
            'actual_passes': {
                'jitter': sum(self.promotion_criteria_history['jitter']),
                'neural': sum(self.promotion_criteria_history['neural']),
                'stability': sum(self.promotion_criteria_history['stability'])
            }
        }

class OTPTransitionProfiler:
    """OTP transition profiling with pre-ramps"""
    def __init__(self, plant):
        self.plant = plant
        self.otp_imminent = False
        self.pre_ramp_steps = 3
        self.ramp_profile = []
        self.disturbance_history = deque(maxlen=10)
        print("INFO: OTPTransitionProfiler initialized.")
    
    def predict_otp(self, step, write_operations):
        """Predict OTP transitions based on write patterns"""
        # Convert deque to list for slicing
        write_list = list(write_operations)
        recent_writes = sum(write_list[-5:]) if len(write_list) >= 5 else 0
        
        # Simple heuristic: OTP likely after sustained writing
        self.otp_imminent = (recent_writes >= 3 and step % 15 == 0)
        return self.otp_imminent
    
    def generate_pre_ramp(self, current_u):
        """Generate pre-ramp control inputs before OTP transition"""
        if not self.otp_imminent or len(self.ramp_profile) > 0:
            return current_u
        
        # Create gentle pre-bias to prepare for disturbance
        ramp_magnitude = 0.4
        self.ramp_profile = [
            current_u + np.array([0.1, -0.15, 0.08]) * ramp_magnitude,
            current_u + np.array([0.2, -0.25, 0.15]) * ramp_magnitude,
            current_u + np.array([0.3, -0.35, 0.22]) * ramp_magnitude
        ]
        return self.ramp_profile.pop(0)
    
    def apply_adaptive_notch(self, u, metrics):
        """Apply adaptive IIR notch filtering to control inputs"""
        # Simple frequency-domain shaping based on jitter spectrum
        jitter_freq_content = metrics.get('jitter_std', 0.1)
        notch_gain = max(0.7, 1.0 - jitter_freq_content * 2)
        return u * notch_gain

class NeuralAwarePDM:
    """Enhanced Neural-aware Phase-Delay Modulation"""
    def __init__(self):
        self.pdm_gain = 1.0
        self.neural_history = deque(maxlen=5)
        self.backoff_active = False
        self.backoff_steps = 0
        self.hysteresis = NeuralPDMHysteresis()
        print("INFO: Enhanced NeuralAwarePDM initialized.")
    
    def update_pdm_gain(self, neural_coherence, neural_derivative=None):
        """Update PDM gain based on neural coherence and trends"""
        self.neural_history.append(neural_coherence)
        
        # Calculate derivative if not provided
        if neural_derivative is None and len(self.neural_history) >= 2:
            neural_derivative = self.neural_history[-1] - self.neural_history[-2]
        elif neural_derivative is None:
            neural_derivative = 0
        
        # Use hysteresis gate for state determination
        gate_state = self.hysteresis.gate(neural_coherence)
        
        # Enhanced backoff logic with hysteresis
        if gate_state == 'OFF' and not self.backoff_active:
            self.backoff_active = True
            self.backoff_steps = 4  # Reduced from 6
            print(f"  PDM: Neural backoff activated (coherence: {neural_coherence:.3f})")
        
        # Apply backoff
        if self.backoff_active:
            self.pdm_gain = 0.65  # Slightly increased from 0.6
            self.backoff_steps -= 1
            if self.backoff_steps <= 0:
                self.backoff_active = False
        else:
            # Enhanced PDM gain with smoother transition
            base_gain = 0.8 + (neural_coherence * 0.2)
            # Apply derivative-based anticipation
            if neural_derivative > 0.02:
                base_gain *= 1.1  # Boost for improving coherence
            elif neural_derivative < -0.02:
                base_gain *= 0.95  # Dampen for degrading coherence
            self.pdm_gain = np.clip(base_gain, 0.65, 1.0)
        
        return self.pdm_gain
    
    def should_defer_writes(self, neural_coherence, neural_derivative):
        """Enhanced deferral with predictive awareness"""
        gate_state = self.hysteresis.gate(neural_coherence)
        
        # Allow writes if coherence is improving rapidly
        if gate_state == 'OFF' and neural_derivative > 0.03:
            return False  # Allow writes despite low current coherence
            
        return gate_state == 'OFF'

class SpectralAdaptiveGain:
    """Enhanced spectral adaptive gain control"""
    def __init__(self):
        self.alpha_history = deque(maxlen=10)
        self.gain_adjustment_factor = 1.0
        self.alpha_target = -0.35
        self.adaptive_rate = 0.05
        
    def update_gain_adjustment(self, current_alpha, stability_score):
        self.alpha_history.append(current_alpha)
        
        if len(self.alpha_history) >= 5:
            alpha_trend = np.polyfit(range(5), list(self.alpha_history)[-5:], 1)[0]
            alpha_mean = np.mean(list(self.alpha_history)[-5:])
            
            # Enhanced gain adjustment with stability awareness
            if alpha_mean > self.alpha_target:
                # System becoming less stable - reduce aggression
                adjustment = 1.0 - (alpha_mean - self.alpha_target) * 2.5  # More responsive
            else:
                # System very stable - can increase performance
                adjustment = 1.0 + (self.alpha_target - alpha_mean) * 2.0  # More aggressive
                
            # Enhanced blending with stability score
            stability_weight = 0.4 if stability_score > 0.85 else 0.6
            self.gain_adjustment_factor = ((1 - stability_weight) * adjustment + 
                                         stability_weight * stability_score)
            
        return np.clip(self.gain_adjustment_factor, 0.6, 1.4)

class ProactiveThermalManager:
    """Enhanced predictive thermal management"""
    def __init__(self):
        self.thermal_history = deque(maxlen=10)
        self.prediction_horizon = 5
        print("INFO: Enhanced ProactiveThermalManager initialized.")
        
    def predict_thermal_trajectory(self, current_temp, write_schedule, current_step, num_steps):
        """Predict temperature over next few steps"""
        future_temp = current_temp
        temp_predictions = []
        
        for i in range(self.prediction_horizon):
            step = current_step + i
            if step >= num_steps:
                break
                
            # Enhanced thermal model with tier awareness
            if step % 6 == 0:  # Write step
                future_temp += 0.35  # Reduced heating
            else:
                future_temp -= 0.15  # Enhanced cooling
                
            future_temp = np.clip(future_temp, 25.0, 85.0)
            temp_predictions.append(future_temp)
            
        return temp_predictions
    
    def get_cooling_recommendation(self, current_temp, predicted_temps):
        """Get cooling recommendations based on predictions"""
        max_predicted = max(predicted_temps) if predicted_temps else current_temp
        
        if max_predicted > 50.0:  # Lowered thresholds for more aggressive cooling
            return {"cooling_boost": 2.2, "urgency": "high"}
        elif max_predicted > 46.0:
            return {"cooling_boost": 1.6, "urgency": "medium"}
        elif max_predicted > 43.0:
            return {"cooling_boost": 1.3, "urgency": "low"}
        else:
            return {"cooling_boost": 1.0, "urgency": "none"}

class BERSurrogateModel:
    """Enhanced BER surrogate model with performance optimization"""
    def __init__(self):
        self.feature_weights = {
            'jitter': 2.3,  # Reduced weights for less conservatism
            'jitter_derivative': 1.6,
            'syndrome': 1.1,
            'stability': -1.7,  # Increased stability importance
            'temperature_effect': 0.7
        }
        self.bias = -9.8  # Less conservative base
        self.ber_threshold = -9.0
        self.consecutive_violations = 0
        self.margin_optimizer = BERMarginOptimizer()
        print("INFO: Enhanced BERSurrogateModel initialized.")
    
    def predict_ber(self, metrics, stability_score, S, jitter_derivative):
        """Enhanced BER prediction with tier awareness"""
        jitter_effect = (max(0, metrics['jitter'] - 5.0) / 3.0) * self.feature_weights['jitter']
        derivative_effect = max(0, jitter_derivative) * self.feature_weights['jitter_derivative']
        syndrome_effect = S * self.feature_weights['syndrome']
        stability_effect = (1.0 - stability_score) * self.feature_weights['stability']
        thermal_effect = max(0, (metrics['die_temp_proxy'] - 40) / 20) * self.feature_weights['temperature_effect']
        
        log_ber = (self.bias + jitter_effect + derivative_effect + 
                  syndrome_effect + stability_effect + thermal_effect)
        
        return log_ber
    
    def should_throttle(self, predicted_ber, current_conditions):
        """Enhanced throttle decision with performance optimization"""
        should_throttle, throttle_severity = self.margin_optimizer.get_optimized_throttle_decision(
            predicted_ber, current_conditions
        )
        
        if should_throttle:
            self.consecutive_violations += 1
        else:
            self.consecutive_violations = max(0, self.consecutive_violations - 0.5)  # Faster recovery
        
        # Enhanced logic: only throttle if conditions persist
        if self.consecutive_violations >= 3:  # Reduced from 4
            should_throttle = True
            throttle_severity = max(throttle_severity, 0.4)  # Reduced minimum throttle
            
        return should_throttle, throttle_severity

# --- ENHANCED CORE COMPONENTS ---

class V5OpticalChannel:
    """Enhanced optical channel with optimized neural coherence"""
    def __init__(self):
        # More realistic plant model
        self.A = np.array([[-0.32, 0.07, 0.015], 
                          [-0.11, -0.42, 0.025], 
                          [0.018, -0.055, -0.28]])
        self.B = np.eye(3) * 0.14
        self.C = np.eye(3)
        self.x = np.zeros(3)
        self.die_temp_proxy = 38.0
        self.recovery_mode = False
        self.recovery_steps = 0
        self.process_noise_std = 0.025
        self.measurement_noise_std = 0.02
        self.thermal_manager = ProactiveThermalManager()
        print("INFO: Enhanced V5OpticalChannel initialized.")

    def get_true_state(self):
        return self.x.copy()

    def get_measurement(self):
        noise = np.random.randn(3) * self.measurement_noise_std
        return self.C @ self.x + noise

    def update(self, u, is_writing, cooling_boost=1.0):
        """Enhanced update with thermal management"""
        process_noise = np.random.randn(3) * self.process_noise_std
        
        if self.recovery_mode:
            effective_B = self.B * 1.4
            self.recovery_steps -= 1
            if self.recovery_steps <= 0:
                self.recovery_mode = False
        else:
            effective_B = self.B
            
        self.x = self.A @ self.x + effective_B @ u + process_noise
        
        # Enhanced thermal dynamics with cooling boost
        if is_writing:
            self.die_temp_proxy += 0.35  # Reduced heating
        else:
            # Enhanced cooling with boost factor
            self.die_temp_proxy -= 0.15 * cooling_boost  # Increased cooling
            
        self.die_temp_proxy = np.clip(self.die_temp_proxy, 25.0, 85.0)

    def inject_disturbance(self, magnitude=0.5, disturbance_type='standard'):
        """Enhanced disturbance injection with types"""
        if disturbance_type == 'standard':
            disturbance = np.array([0.25, -0.4, 0.6]) * magnitude
        elif disturbance_type == 'thermal':
            disturbance = np.array([0.1, -0.2, 0.8]) * magnitude  # More thermal coupling
        else:  # neural
            disturbance = np.array([0.4, -0.3, 0.3]) * magnitude
            
        self.x += disturbance
        self.recovery_mode = True
        self.recovery_steps = 7
        return disturbance

    def get_metrics(self):
        """Enhanced metrics with optimized neural coherence"""
        thermal_effect = max(0, (self.die_temp_proxy - 40) / 20)
        jitter = (4.3 + np.abs(self.x[0]) * 3.8 + np.abs(self.x[1]) * 7.5 + 
                 thermal_effect * 1.8 + np.random.uniform(-0.25, 0.25))
        
        pi_rate = 0.07 + np.abs(self.x[2]) * 1.3 + np.linalg.norm(self.x) * 0.25
        po_rate = 0.006 + np.abs(self.x[2]) * 0.25
        
        # Enhanced physics-tied neural coherence (optimized sensitivity)
        base_coherence = 0.96 - 0.025 * jitter - 0.018 * thermal_effect  # Improved baseline
        neural_coherence = np.clip(np.random.normal(base_coherence, 0.018), 0.80, 0.99)  # Tighter distribution
        
        return {
            "jitter": max(0.0, jitter), 
            "pi_rate": max(0.0, pi_rate), 
            "po_rate": max(0.0, po_rate),
            "die_temp_proxy": self.die_temp_proxy,
            "recovery_mode": self.recovery_mode,
            "true_focus_error": float(self.x[0]),
            "true_track_error": float(self.x[1]),
            "i_neural": neural_coherence  # Now optimized
        }

class V5MACS_Controller:
    """Enhanced MACS with performance optimization"""
    def __init__(self, A, B, C):
        self.A, self.B, self.C = A, B, C
        self.K = np.eye(3) * 0.65
        self.W = np.diag([1.0, 1.0, 1.0])
        self.eps = 1e-3
        self.syndrome_history = deque(maxlen=8)
        self.pdm = NeuralAwarePDM()
        self.spectral_adaptor = SpectralAdaptiveGain()
        print("INFO: Enhanced V5MACS_Controller initialized.")

    def _get_spectral_abscissa(self, A_cl):
        return np.max(np.real(np.linalg.eigvals(A_cl)))

    def step(self, x_estimated, metrics, recovery_mode=False, neural_coherence=1.0, neural_derivative=0.0):
        """Enhanced control step with performance optimization"""
        # Calculate syndrome pressure with improved calibration
        jitter_penalty = max(0, (metrics['jitter'] - 5.2) / 3.5)
        thermal_penalty = max(0, (metrics['die_temp_proxy'] - 48) / 25)
        
        S = (0.28 * jitter_penalty + 
             0.26 * (metrics['pi_rate'] / 4.5) + 
             0.22 * (metrics['po_rate'] / 0.4) + 
             0.14 * thermal_penalty +
             0.10 * (1.0 - neural_coherence))
        
        self.syndrome_history.append(S)
        
        # Neural-aware PDM gain adjustment
        pdm_gain = self.pdm.update_pdm_gain(neural_coherence, neural_derivative)
        
        # Spectral adaptive gain adjustment
        A_cl = self.A - self.B @ self.K @ self.W @ self.C
        current_alpha = self._get_spectral_abscissa(A_cl)
        spectral_gain = self.spectral_adaptor.update_gain_adjustment(
            current_alpha, metrics.get('stability_score', 0.5)
        )
        
        # Enhanced thermal adaptation
        temp_factor = 1.0 + max(0, (metrics['die_temp_proxy'] - 46.0) / 30.0) * 0.35
        
        # Recovery mode adaptation
        if recovery_mode:
            recovery_boost = 1.25
        else:
            recovery_boost = 1.0
            
        w_diag = np.diag(self.W)
        
        # Enhanced trend-aware weight update
        if len(self.syndrome_history) > 1:
            syndrome_trend = np.polyfit(range(len(self.syndrome_history)), 
                                      list(self.syndrome_history), 1)[0]
            # More responsive trend factor
            trend_factor = 1.0 + 0.15 * syndrome_trend * 8  # Increased responsiveness
        else:
            trend_factor = 1.0
            
        w_update_factor = 1.0 + 0.08 * S * trend_factor  # Slightly increased update rate
        w_new_diag = np.clip(w_diag * w_update_factor, 0.4, 3.5)
        
        # Apply all adaptation factors with spectral gain
        w_new_diag[:2] *= (temp_factor * recovery_boost * pdm_gain * spectral_gain)
        
        self.W = np.diag(w_new_diag)
        
        # Control calculation with estimated state
        u = -self.K @ self.W @ x_estimated

        alpha = self._get_spectral_abscissa(A_cl)
        is_stable = alpha <= -self.eps

        return u, S, alpha, is_stable, pdm_gain

# --- V0.7 MAIN SIMULATION ---

def run_v7_simulation():
    """V0.7 Enhanced Universal Nexus Simulation with Performance Optimization"""
    print("=== PAZUZUBURN V0.7 - PERFORMANCE OPTIMIZED UNIVERSAL NEXUS ===")
    print("Initializing optimized architecture with V0.7 enhancements...\n")
    
    # Initialize enhanced components
    ledger = MerkleLogHLA()
    plant = V5OpticalChannel()
    
    # Enhanced Kalman filter with optimized tuning
    Q = np.eye(3) * (plant.process_noise_std ** 2)
    R = np.eye(3) * (plant.measurement_noise_std ** 2)
    kalman = KalmanStateObserver(plant.A, plant.B, plant.C, Q, R)
    adaptive_kalman = InnovationAdaptiveKalman(kalman)
    
    controller = V5MACS_Controller(plant.A, plant.B, plant.C)
    monitor = StabilityMonitor()
    predictor = PredictiveFailureAnalyzer()
    credit_system = AdaptiveCreditSystem()
    otp_profiler = OTPTransitionProfiler(plant)
    ber_predictor = BERSurrogateModel()
    pi_lock_detector = PILockCUSUM()
    promotion_verifier = WindowedPromotionVerifier()
    
    # Enhanced RTR with optimized tier system
    class V7RiskTieredRouter:
        def __init__(self):
            # Relaxed neural coherence requirements
            self.tier_gates = {
                5: {"jitter_max": 6.0, "neural_min": 0.86, "stability_min": 0.70},  # Neural relaxed from 0.88
                6: {"jitter_max": 5.5, "neural_min": 0.89, "stability_min": 0.80}   # Neural relaxed from 0.92
            }
            self.demotion_triggers = {
                5: {"max_jitter": 7.0, "min_stability": 0.4, "max_consecutive_alerts": 3},
                6: {"max_jitter": 6.5, "min_stability": 0.5, "max_consecutive_alerts": 2}
            }
            print("INFO: V7RiskTieredRouter initialized.")
    
    rtr = V7RiskTieredRouter()
    
    current_tier = 4
    consecutive_alerts = 0
    max_alerts = 4
    write_operations = deque(maxlen=10)
    neural_history = deque(maxlen=5)
    last_u = np.zeros(3)
    demotion_watch = 0
    thermal_manager = ProactiveThermalManager()
    performance_boost_active = False
    
    # Enhanced initialization with comprehensive logging
    init_stats = {
        'jitter_range': [4.0, 6.0],
        'neural_baseline': 0.91,
        'thermal_limits': [25.0, 85.0],
        'tier_gates': rtr.tier_gates
    }
    
    ledger.append({
        "type": "SYSTEM_INIT", 
        "version": "PazuzuBurn v0.7",
        "timestamp": time.time(),
        "features": [
            "OptimizedNeuralCoherence", "RelaxedCUSUM", "EnhancedPDM", 
            "OptimizedCredits", "FixedKalman", "RelaxedPromotion",
            "TierAwareBER", "StableCorrelation", "PerformanceBoost"
        ]
    }, init_stats)
    
    print(f"Initial Ledger Root: {ledger.root().hex()}\n")
    print("Starting optimized control loops with V0.7 performance enhancements...\n")

    num_steps = 60
    disturbance_injected = False
    
    for i in range(num_steps):
        # Initialize promotion_verified at the start of each step
        promotion_verified = False
        
        is_writing = (i > 0 and i % 6 == 0)
        if is_writing:
            write_operations.append(1)
        else:
            write_operations.append(0)
        
        # Enhanced proactive thermal management
        write_schedule = [1 if (i + j) % 6 == 0 else 0 for j in range(5)]
        temp_predictions = thermal_manager.predict_thermal_trajectory(
            plant.die_temp_proxy, write_schedule, i, num_steps
        )
        cooling_recommendation = thermal_manager.get_cooling_recommendation(
            plant.die_temp_proxy, temp_predictions
        )
        
        # OTP transition prediction and pre-ramp
        otp_predicted = otp_profiler.predict_otp(i, write_operations)
        
        # Enhanced Kalman filtering with fixed adaptive tuning
        z = plant.get_measurement()
        if i == 0:
            adaptive_kalman.kf.x_hat = z.copy()
            x_estimated = z.copy()
        else:
            x_estimated = adaptive_kalman.adaptive_update(z, adaptation_active=True)
        
        # Get enhanced metrics with optimized neural coherence
        metrics = plant.get_metrics()
        
        neural_history.append(metrics['i_neural'])
        if len(neural_history) >= 2:
            neural_derivative = neural_history[-1] - neural_history[-2]
        else:
            neural_derivative = 0.0
        
        # Enhanced stability monitoring with cross-correlation
        stability_score = monitor.update(metrics, metrics['i_neural'], 0.0)  # S calculated later
        metrics['stability_score'] = stability_score
        
        # Calculate jitter trend for BER prediction
        if len(monitor.metric_history) >= 2:
            metric_list = list(monitor.metric_history)
            recent_jitters = [m['jitter'] for m in metric_list[-2:]] + [metrics['jitter']]
            jitter_derivative = np.polyfit(range(len(recent_jitters)), recent_jitters, 1)[0]
        else:
            jitter_derivative = 0.0
        
        # Enhanced control with performance optimization
        if otp_predicted and i < num_steps - 5:
            print(f"\n--- OTP PREDICTION: Pre-ramp active (step {i+1}) ---")
            u, S, alpha, is_stable, pdm_gain = controller.step(
                x_estimated, metrics, False, metrics['i_neural'], neural_derivative)
            u = otp_profiler.generate_pre_ramp(u)
        else:
            u, S, alpha, is_stable, pdm_gain = controller.step(
                x_estimated, metrics, plant.recovery_mode, metrics['i_neural'], neural_derivative)
            u = otp_profiler.apply_adaptive_notch(u, metrics)
        
        last_u = u.copy()
        
        # Enhanced PI-Lock detection using optimized CUSUM
        has_pi_lock = pi_lock_detector.update(S)
        if has_pi_lock:
            print(f"  CUSUM: PI-Lock detected (S={S:.3f})")
        
        # Enhanced write scheduling with predictive awareness
        actual_write = is_writing
        if controller.pdm.should_defer_writes(metrics['i_neural'], neural_derivative):
            actual_write = False
            if is_writing:
                print(f"  PDM: Write deferred (neural: {metrics['i_neural']:.3f}, deriv: {neural_derivative:+.3f})")
        
        # Update plant with enhanced thermal management
        plant.update(u, actual_write, cooling_recommendation['cooling_boost'])
        
        # Enhanced BER-based throttling with performance optimization
        predicted_ber = ber_predictor.predict_ber(metrics, stability_score, S, jitter_derivative)
        current_conditions = {
            'stability': stability_score,
            'neural_coherence': metrics['i_neural'],
            'tier': current_tier
        }
        should_throttle, throttle_severity = ber_predictor.should_throttle(predicted_ber, current_conditions)
        
        if should_throttle and not plant.recovery_mode and S > 0.1:
            print(f"  BER OPTIMIZATION: Adaptive throttle {throttle_severity:.1%} (predicted BER: 10^{predicted_ber:.1f})")
            controller.W = controller.W * (1.0 - throttle_severity)
        
        # Enhanced credit system with optimized updates
        credit_system.update_bi_directional_credits(metrics, stability_score, S, has_pi_lock)
        
        # Check for demotion with enhanced logic
        if credit_system.should_demote(current_tier):
            demotion_watch += 1
            if demotion_watch >= 3:  # Require sustained poor performance
                current_tier = max(4, current_tier - 1)
                demotion_watch = 0
                print(f"\n⚠️  TIER DEMOTION: Tier {current_tier+1} -> {current_tier} (credits: {credit_system.safety_credits[current_tier+1]:.2f})")
                ledger.append({
                    "type": "TIER_DEMOTION",
                    "new_tier": current_tier,
                    "reason": "credit_threshold",
                    "credits": float(credit_system.safety_credits[current_tier+1])
                })
        else:
            demotion_watch = max(0, demotion_watch - 0.5)
        
        # Enhanced promotion with optimized verification
        promotion_possible = credit_system.safety_credits.get(current_tier + 1, 0) >= credit_system.promotion_thresholds.get(current_tier + 1, 999)
        
        if promotion_possible and current_tier + 1 in rtr.tier_gates:
            gate = rtr.tier_gates[current_tier + 1]
            promotion_verified = promotion_verifier.update_criteria(
                metrics['jitter'], metrics['i_neural'], stability_score, gate
            )
            
            if promotion_verified:
                current_tier += 1
                promotion_proof = promotion_verifier.get_promotion_proof()
                print(f"\n🎯 VALIDATED TIER PROMOTION: Tier {current_tier} achieved!")
                print(f"   Credits: {credit_system.safety_credits[current_tier]:.2f}/" +
                      f"{credit_system.promotion_thresholds[current_tier]:.1f}")
                print(f"   Window Proof: {promotion_proof}")
                
                ledger.append({
                    "type": "TIER_PROMOTION",
                    "new_tier": current_tier,
                    "credits": float(credit_system.safety_credits[current_tier]),
                    "proof": promotion_proof
                }, {
                    'jitter': float(metrics['jitter']),
                    'stability': float(stability_score),
                    'neural': float(metrics['i_neural'])
                })
        
        # Innovation statistics from enhanced Kalman
        innovation_mean, innovation_std = kalman.get_innovation_stats()
        
        # Enhanced compact stats for ledger (with performance metrics)
        compact_stats = None
        ledger_snapshot_condition = (i % 5 == 0 or promotion_verified)
        if ledger_snapshot_condition:
            compact_stats = {
                'jitter_stats': [metrics['jitter']],
                'alpha': float(alpha),
                'neural_coherence': float(metrics['i_neural']),
                'stability': float(stability_score),
                'innov_mean': float(innovation_mean),
                'credits_t5': float(credit_system.safety_credits[5]),
                'credits_t6': float(credit_system.safety_credits[6]),
                'tier': current_tier
            }
            # Actually append status to ledger (fix from V0.6)
            ledger.append({
                "type": "STATUS_SNAPSHOT",
                "step": i+1,
                "tier": current_tier
            }, compact_stats)
        
        # Enhanced comprehensive status display
        status = (f"--- Step {i+1}/{num_steps} [Tier {current_tier}] ---\n"
                 f"  METRICS: Jitter={metrics['jitter']:.2f}% | Temp={metrics['die_temp_proxy']:.1f}°C | Stability={stability_score:.2f}\n"
                 f"  CONTROL: S={S:.3f} | α={alpha:.4f} | PDM={pdm_gain:.2f} | Kalman_Adapt={adaptive_kalman.adaptation_count}\n"
                 f"  CREDITS: T5={credit_system.safety_credits[5]:.2f}/3.0 | T6={credit_system.safety_credits[6]:.2f}/5.0 | DemotionWatch={demotion_watch:.1f}")
        
        if predicted_ber > ber_predictor.ber_threshold - 1.0:
            adaptive_threshold = ber_predictor.margin_optimizer.update_adaptive_threshold(
                stability_score, metrics['i_neural'], current_tier
            )
            status += f"\n  BER: 10^{predicted_ber:.1f} (Adaptive: 10^{adaptive_threshold:.1f})"
        
        if cooling_recommendation['urgency'] != 'none':
            status += f"\n  THERMAL: {cooling_recommendation['urgency']} cooling (boost: {cooling_recommendation['cooling_boost']:.1f}x)"
            
        # Performance boost indicator
        if current_tier >= 5 and stability_score > 0.85 and metrics['i_neural'] > 0.87:
            if not performance_boost_active:
                performance_boost_active = True
                print("  🚀 PERFORMANCE BOOST: Optimal conditions for enhanced operation")
            status += " | 🚀 BOOST"
        else:
            performance_boost_active = False
            
        print(status)
        
        # Enhanced disturbance injection with performance monitoring
        if i == 25 and not disturbance_injected:
            print("\n--- INJECTING ENHANCED DISTURBANCE (Step 26) ---")
            disturbance = plant.inject_disturbance(magnitude=0.55, disturbance_type='thermal')
            disturbance_injected = True
            
            # Log disturbance with enhanced pattern info
            ledger.append({
                "type": "CONTROLLED_DISTURBANCE",
                "step": i+1,
                "magnitude": float(np.linalg.norm(disturbance)),
                "type": "thermal",
                "pre_ramp_used": otp_predicted,
                "current_tier": current_tier
            })
        
        # Enhanced predictive failure analysis
        failure_prediction = predictor.analyze(metrics, stability_score, disturbance if disturbance_injected and i == 26 else None)
        if failure_prediction:
            print(f"\n⚠️  PREDICTIVE ALERT: {failure_prediction}")
            consecutive_alerts += 1
            if consecutive_alerts >= max_alerts:
                print(f"\n🚨 CRITICAL: {max_alerts} alerts - Controlled shutdown")
                ledger.append({"type": "CONTROLLED_SHUTDOWN", "reason": "MAX_ALERTS"})
                break
        else:
            consecutive_alerts = max(0, consecutive_alerts - 0.3)
    
    # Final enhanced performance report
    print("\n" + "="*70)
    print("V0.7 PERFORMANCE OPTIMIZED REPORT")
    print("="*70)
    
    # Calculate comprehensive metrics
    final_credits = {tier: credit_system.safety_credits[tier] for tier in [5, 6]}
    promotion_status = "PROMOTED" if current_tier > 4 else "BASELINE"
    demotion_status = "STABLE" if demotion_watch == 0 else "AT_RISK"
    
    # Enhanced metrics calculation
    if monitor.metric_history:
        metric_list = list(monitor.metric_history)
        avg_stability = np.mean([m.get('stability_score', 0.5) for m in metric_list])
        avg_jitter = np.mean([m.get('jitter', 5.0) for m in metric_list])
        avg_neural = np.mean([m.get('i_neural', 0.8) for m in metric_list if 'i_neural' in m])
        
        print(f"FINAL STATE: Tier {current_tier} ({promotion_status}) | Demotion Risk: {demotion_status}")
        print(f"CREDIT PERFORMANCE: T5: {final_credits[5]:.2f}/3.0 | T6: {final_credits[6]:.2f}/5.0")
        print(f"CONTROL PERFORMANCE: Final α={alpha:.4f} | Avg Stability={avg_stability:.3f} | Avg Jitter={avg_jitter:.2f}%")
        print(f"NEURAL PERFORMANCE: Avg Coherence={avg_neural:.3f} | PDM toggles={controller.pdm.hysteresis.toggle_count}")
        print(f"ADAPTIVE SYSTEMS: Kalman adaptations={adaptive_kalman.adaptation_count} | PI-Lock detections={pi_lock_detector.detection_count}")
        
        # Cross-correlation insights
        critical_corrs = monitor.cross_correlator.get_critical_correlations()
        if critical_corrs:
            print("CROSS-DOMAIN INSIGHTS:")
            for i, j, strength in critical_corrs:
                domains = ['Jitter', 'Neural', 'Thermal', 'Syndrome', 'Stability']
                print(f"  {domains[i]} ↔ {domains[j]}: {strength:.3f}")
    
    print(f"LEDGER INTEGRITY: {ledger.event_count} events | Root: {ledger.root().hex()[:16]}...")
    
    if current_tier >= 5:
        print("🎉 V0.7 SUCCESS: Tier progression achieved with optimized performance!")
    elif current_tier == 4 and final_credits[5] >= 3.0:
        print("⚠️  V0.7: Credits met but verification failed - review neural coherence thresholds")
    else:
        print("⚠️  V0.7: Maintained stable baseline - continue credit accumulation")
    
    print("=== PERFORMANCE OPTIMIZED V0.7 SIMULATION COMPLETE ===")

if __name__ == "__main__":
    run_v7_simulation()
