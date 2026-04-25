"""API defense mechanisms for model stealing detection."""

import hashlib
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class RateLimiter:
    """
    Rate limiting defense mechanism.
    
    This class implements rate limiting to prevent excessive API calls
    that may indicate model stealing attempts.
    """
    
    def __init__(
        self,
        max_requests_per_minute: int = 60,
        max_requests_per_hour: int = 1000,
        max_requests_per_day: int = 10000
    ):
        """
        Initialize the rate limiter.
        
        Args:
            max_requests_per_minute: Maximum requests per minute
            max_requests_per_hour: Maximum requests per hour
            max_requests_per_day: Maximum requests per day
        """
        self.max_requests_per_minute = max_requests_per_minute
        self.max_requests_per_hour = max_requests_per_hour
        self.max_requests_per_day = max_requests_per_day
        
        # Track requests per user
        self.user_requests = {}
        
    def is_allowed(self, user_id: str) -> Tuple[bool, str]:
        """
        Check if a request is allowed for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Tuple of (is_allowed, reason)
        """
        current_time = time.time()
        
        # Initialize user tracking if not exists
        if user_id not in self.user_requests:
            self.user_requests[user_id] = {
                "minute": [],
                "hour": [],
                "day": []
            }
        
        user_data = self.user_requests[user_id]
        
        # Clean old requests
        self._clean_old_requests(user_data, current_time)
        
        # Check rate limits
        if len(user_data["minute"]) >= self.max_requests_per_minute:
            return False, "Rate limit exceeded: too many requests per minute"
        
        if len(user_data["hour"]) >= self.max_requests_per_hour:
            return False, "Rate limit exceeded: too many requests per hour"
        
        if len(user_data["day"]) >= self.max_requests_per_day:
            return False, "Rate limit exceeded: too many requests per day"
        
        # Add current request
        user_data["minute"].append(current_time)
        user_data["hour"].append(current_time)
        user_data["day"].append(current_time)
        
        return True, "Request allowed"
    
    def _clean_old_requests(self, user_data: Dict, current_time: float) -> None:
        """Clean old requests from tracking."""
        # Clean minute-level requests (older than 60 seconds)
        user_data["minute"] = [
            req_time for req_time in user_data["minute"]
            if current_time - req_time < 60
        ]
        
        # Clean hour-level requests (older than 3600 seconds)
        user_data["hour"] = [
            req_time for req_time in user_data["hour"]
            if current_time - req_time < 3600
        ]
        
        # Clean day-level requests (older than 86400 seconds)
        user_data["day"] = [
            req_time for req_time in user_data["day"]
            if current_time - req_time < 86400
        ]


class QueryDeduplicator:
    """
    Query deduplication defense mechanism.
    
    This class implements query deduplication to prevent repeated
    queries that may indicate model stealing attempts.
    """
    
    def __init__(self, max_duplicates: int = 5, time_window: int = 3600):
        """
        Initialize the query deduplicator.
        
        Args:
            max_duplicates: Maximum number of duplicate queries allowed
            time_window: Time window for duplicate detection (seconds)
        """
        self.max_duplicates = max_duplicates
        self.time_window = time_window
        
        # Track queries per user
        self.user_queries = {}
        
    def is_duplicate(self, user_id: str, query: np.ndarray) -> Tuple[bool, str]:
        """
        Check if a query is a duplicate.
        
        Args:
            user_id: User identifier
            query: Query vector
            
        Returns:
            Tuple of (is_duplicate, reason)
        """
        # Generate query hash
        query_hash = hashlib.md5(query.tobytes()).hexdigest()
        current_time = time.time()
        
        # Initialize user tracking if not exists
        if user_id not in self.user_queries:
            self.user_queries[user_id] = {}
        
        user_data = self.user_queries[user_id]
        
        # Clean old queries
        if query_hash in user_data:
            user_data[query_hash] = [
                req_time for req_time in user_data[query_hash]
                if current_time - req_time < self.time_window
            ]
        else:
            user_data[query_hash] = []
        
        # Check duplicate count
        if len(user_data[query_hash]) >= self.max_duplicates:
            return True, f"Duplicate query detected: {len(user_data[query_hash])} occurrences"
        
        # Add current query
        user_data[query_hash].append(current_time)
        
        return False, "Query is unique"


class ResponseRandomizer:
    """
    Response randomization defense mechanism.
    
    This class implements response randomization to make it harder
    for attackers to extract model information.
    """
    
    def __init__(self, noise_level: float = 0.1, random_seed: int = 42):
        """
        Initialize the response randomizer.
        
        Args:
            noise_level: Level of noise to add to responses
            random_seed: Random seed for reproducibility
        """
        self.noise_level = noise_level
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
    def randomize_response(self, response: np.ndarray) -> np.ndarray:
        """
        Randomize a model response.
        
        Args:
            response: Original model response
            
        Returns:
            Randomized response
        """
        # Add Gaussian noise
        noise = np.random.normal(0, self.noise_level, response.shape)
        randomized_response = response + noise
        
        # Clip to valid range
        randomized_response = np.clip(randomized_response, 0, 1)
        
        return randomized_response
    
    def randomize_probabilities(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Randomize probability distributions.
        
        Args:
            probabilities: Original probability distribution
            
        Returns:
            Randomized probability distribution
        """
        # Add noise to logits
        logits = np.log(probabilities + 1e-9)
        noise = np.random.normal(0, self.noise_level, logits.shape)
        randomized_logits = logits + noise
        
        # Convert back to probabilities
        randomized_probabilities = np.exp(randomized_logits)
        randomized_probabilities = randomized_probabilities / np.sum(randomized_probabilities, axis=1, keepdims=True)
        
        return randomized_probabilities


class HoneyTrap:
    """
    Honey trap defense mechanism.
    
    This class implements honey traps to detect and track
    suspicious query patterns.
    """
    
    def __init__(self, trap_queries: List[np.ndarray], trap_responses: List[np.ndarray]):
        """
        Initialize the honey trap.
        
        Args:
            trap_queries: List of trap queries
            trap_responses: List of corresponding trap responses
        """
        self.trap_queries = trap_queries
        self.trap_responses = trap_responses
        self.trap_hashes = [hashlib.md5(q.tobytes()).hexdigest() for q in trap_queries]
        
        # Track users who hit traps
        self.trap_hits = {}
        
    def is_trap_query(self, query: np.ndarray) -> bool:
        """
        Check if a query is a trap query.
        
        Args:
            query: Query vector
            
        Returns:
            True if query is a trap
        """
        query_hash = hashlib.md5(query.tobytes()).hexdigest()
        return query_hash in self.trap_hashes
    
    def get_trap_response(self, query: np.ndarray) -> Optional[np.ndarray]:
        """
        Get the trap response for a query.
        
        Args:
            query: Query vector
            
        Returns:
            Trap response if query is a trap, None otherwise
        """
        if not self.is_trap_query(query):
            return None
        
        query_hash = hashlib.md5(query.tobytes()).hexdigest()
        trap_index = self.trap_hashes.index(query_hash)
        return self.trap_responses[trap_index]
    
    def record_trap_hit(self, user_id: str, query: np.ndarray) -> None:
        """
        Record a trap hit for a user.
        
        Args:
            user_id: User identifier
            query: Query vector that hit the trap
        """
        if user_id not in self.trap_hits:
            self.trap_hits[user_id] = []
        
        self.trap_hits[user_id].append({
            "timestamp": time.time(),
            "query_hash": hashlib.md5(query.tobytes()).hexdigest()
        })
    
    def get_trap_hits(self, user_id: str) -> List[Dict]:
        """
        Get trap hits for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of trap hits
        """
        return self.trap_hits.get(user_id, [])
    
    def is_suspicious_user(self, user_id: str, threshold: int = 3) -> bool:
        """
        Check if a user is suspicious based on trap hits.
        
        Args:
            user_id: User identifier
            threshold: Number of trap hits to consider suspicious
            
        Returns:
            True if user is suspicious
        """
        trap_hits = self.get_trap_hits(user_id)
        return len(trap_hits) >= threshold


class APIDefenseSystem:
    """
    Comprehensive API defense system.
    
    This class combines multiple defense mechanisms to protect
    against model stealing attempts.
    """
    
    def __init__(
        self,
        rate_limiter: Optional[RateLimiter] = None,
        query_deduplicator: Optional[QueryDeduplicator] = None,
        response_randomizer: Optional[ResponseRandomizer] = None,
        honey_trap: Optional[HoneyTrap] = None
    ):
        """
        Initialize the API defense system.
        
        Args:
            rate_limiter: Rate limiting defense
            query_deduplicator: Query deduplication defense
            response_randomizer: Response randomization defense
            honey_trap: Honey trap defense
        """
        self.rate_limiter = rate_limiter or RateLimiter()
        self.query_deduplicator = query_deduplicator or QueryDeduplicator()
        self.response_randomizer = response_randomizer or ResponseRandomizer()
        self.honey_trap = honey_trap
        
        # Track defense actions
        self.defense_log = []
        
    def process_query(
        self,
        user_id: str,
        query: np.ndarray,
        model_response: np.ndarray
    ) -> Tuple[bool, Optional[np.ndarray], str]:
        """
        Process a query through the defense system.
        
        Args:
            user_id: User identifier
            query: Query vector
            model_response: Original model response
            
        Returns:
            Tuple of (is_allowed, response, reason)
        """
        # Check rate limiting
        is_allowed, reason = self.rate_limiter.is_allowed(user_id)
        if not is_allowed:
            self._log_defense_action(user_id, "rate_limit", reason)
            return False, None, reason
        
        # Check for duplicates
        is_duplicate, reason = self.query_deduplicator.is_duplicate(user_id, query)
        if is_duplicate:
            self._log_defense_action(user_id, "duplicate", reason)
            return False, None, reason
        
        # Check for honey traps
        if self.honey_trap and self.honey_trap.is_trap_query(query):
            self.honey_trap.record_trap_hit(user_id, query)
            trap_response = self.honey_trap.get_trap_response(query)
            self._log_defense_action(user_id, "honey_trap", "Trap query detected")
            return True, trap_response, "Trap response provided"
        
        # Randomize response
        randomized_response = self.response_randomizer.randomize_response(model_response)
        
        self._log_defense_action(user_id, "allowed", "Query processed successfully")
        return True, randomized_response, "Query processed successfully"
    
    def _log_defense_action(self, user_id: str, action: str, reason: str) -> None:
        """Log a defense action."""
        self.defense_log.append({
            "timestamp": time.time(),
            "user_id": user_id,
            "action": action,
            "reason": reason
        })
    
    def get_defense_log(self, user_id: Optional[str] = None) -> List[Dict]:
        """
        Get defense log entries.
        
        Args:
            user_id: Optional user ID to filter by
            
        Returns:
            List of defense log entries
        """
        if user_id is None:
            return self.defense_log
        
        return [entry for entry in self.defense_log if entry["user_id"] == user_id]
    
    def get_suspicious_users(self) -> List[str]:
        """
        Get list of suspicious users.
        
        Returns:
            List of suspicious user IDs
        """
        suspicious_users = set()
        
        # Check honey trap hits
        if self.honey_trap:
            for user_id in self.honey_trap.trap_hits:
                if self.honey_trap.is_suspicious_user(user_id):
                    suspicious_users.add(user_id)
        
        # Check defense log for blocked requests
        for entry in self.defense_log:
            if entry["action"] in ["rate_limit", "duplicate"]:
                suspicious_users.add(entry["user_id"])
        
        return list(suspicious_users)
