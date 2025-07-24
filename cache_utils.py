import os
import json
import threading
import re
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict

logger = logging.getLogger(__name__)

CACHE_FILE = "prompt_response_cache.json"
_LOCK = threading.Lock()
CACHE_EXPIRY_HOURS = 24

def normalize_prompt_for_cache(prompt: str) -> str:
    """Normalize prompt for consistent cache key generation"""
    # Remove extra whitespace and normalize line endings
    normalized = re.sub(r'\s+', ' ', prompt.strip())
    # Convert to lowercase for case-insensitive matching
    normalized = normalized.lower()
    return normalized

def generate_cache_key(prompt: str) -> str:
    """Generate a consistent hash-based cache key"""
    normalized = normalize_prompt_for_cache(prompt)
    return hashlib.md5(normalized.encode()).hexdigest()

def _load_cache():
    """Load cache from file with error handling"""
    if not os.path.exists(CACHE_FILE):
        return {}
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Error loading cache file: {e}")
        return {}

def _save_cache(cache_dict):
    """Save cache to file with thread safety"""
    with _LOCK:
        try:
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(cache_dict, f, indent=2, ensure_ascii=False)
        except IOError as e:
            logger.error(f"Error saving cache file: {e}")

def get_cached_response(prompt: str) -> Optional[str]:
    """Retrieve cached response with proper key generation and expiration check"""
    cache_key = generate_cache_key(prompt)
    logger.info(f"Looking for cache key: {cache_key}")
    
    cache = _load_cache()
    
    if cache_key in cache:
        cache_entry = cache[cache_key]
        
        # Check if cache entry has required fields
        if not isinstance(cache_entry, dict) or 'response' not in cache_entry:
            logger.warning(f"Invalid cache entry format for key: {cache_key}")
            return None
        
        # Check if cache is still valid
        try:
            cache_time = datetime.fromisoformat(cache_entry['timestamp'])
            if datetime.now() - cache_time < timedelta(hours=CACHE_EXPIRY_HOURS):
                logger.info("Cache hit!")
                
                # Update access count and save back
                cache_entry['access_count'] = cache_entry.get('access_count', 0) + 1
                cache_entry['last_accessed'] = datetime.now().isoformat()
                _save_cache(cache)
                
                return cache_entry['response']
            else:
                # Remove expired cache entry
                del cache[cache_key]
                _save_cache(cache)
                logger.info("Cache expired and removed")
        except (KeyError, ValueError) as e:
            logger.error(f"Error processing cache entry: {e}")
            # Remove corrupted entry
            del cache[cache_key]
            _save_cache(cache)
    
    logger.info("Cache miss!")
    return None

def set_cache_response(prompt: str, response: str):
    """Store response in cache with proper key generation"""
    cache_key = generate_cache_key(prompt)
    logger.info(f"Storing in cache with key: {cache_key}")
    
    cache = _load_cache()
    cache[cache_key] = {
        'prompt': prompt,
        'response': response,
        'timestamp': datetime.now().isoformat(),
        'last_accessed': datetime.now().isoformat(),
        'access_count': 1,
        'response_length': len(response)
    }
    _save_cache(cache)
    logger.info(f"Successfully cached response for key: {cache_key}")

def get_last_n_queries(n=5):
    """Return last n queries from cache in insertion order (newest last)."""
    cache = _load_cache()
    # Sort by timestamp to get truly last queries
    sorted_items = sorted(
        cache.items(), 
        key=lambda x: x[1].get('timestamp', ''), 
        reverse=True
    )
    return [item[0] for item in sorted_items[:n]]

def clear_cache():
    """Clear all cached responses"""
    with _LOCK:
        try:
            if os.path.exists(CACHE_FILE):
                os.remove(CACHE_FILE)
            logger.info("Cache file cleared")
        except IOError as e:
            logger.error(f"Error clearing cache file: {e}")

def get_cache_stats():
    """Get cache statistics"""
    cache = _load_cache()
    if not cache:
        return {
            "total_entries": 0,
            "cache_file_exists": os.path.exists(CACHE_FILE),
            "cache_file_size": 0
        }
    
    total_response_size = sum(
        len(entry.get('response', '')) for entry in cache.values() 
        if isinstance(entry, dict)
    )
    
    return {
        "total_entries": len(cache),
        "cache_file_exists": os.path.exists(CACHE_FILE),
        "cache_file_size": os.path.getsize(CACHE_FILE) if os.path.exists(CACHE_FILE) else 0,
        "total_response_size_chars": total_response_size,
        "average_response_size": total_response_size // len(cache) if len(cache) > 0 else 0
    }

def cleanup_expired_cache():
    """Remove expired entries from cache"""
    cache = _load_cache()
    current_time = datetime.now()
    expired_keys = []
    
    for cache_key, cache_entry in cache.items():
        if not isinstance(cache_entry, dict) or 'timestamp' not in cache_entry:
            expired_keys.append(cache_key)
            continue
            
        try:
            cache_time = datetime.fromisoformat(cache_entry['timestamp'])
            if current_time - cache_time >= timedelta(hours=CACHE_EXPIRY_HOURS):
                expired_keys.append(cache_key)
        except ValueError:
            expired_keys.append(cache_key)
    
    if expired_keys:
        for key in expired_keys:
            del cache[key]
        _save_cache(cache)
        logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    return len(expired_keys)

def search_cache_by_keyword(keyword: str):
    """Search for cached prompts containing a keyword"""
    cache = _load_cache()
    matching_entries = []
    
    for cache_key, entry in cache.items():
        if not isinstance(entry, dict) or 'prompt' not in entry:
            continue
            
        if keyword.lower() in entry['prompt'].lower():
            matching_entries.append({
                "cache_key": cache_key,
                "prompt_preview": entry['prompt'][:100] + "..." if len(entry['prompt']) > 100 else entry['prompt'],
                "timestamp": entry.get('timestamp', 'Unknown'),
                "access_count": entry.get('access_count', 0)
            })
    
    return matching_entries
