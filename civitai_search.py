#!/usr/bin/env python3
"""
Civitai Model Search Script

This script searches for models on Civitai with various filtering options,
including an optional filter for models that include training data.

Requirements:
- requests library: pip install requests
- Civitai API key (get from https://civitai.com/user/account)

Usage:
    python civitai_search.py "anime character"
    python civitai_search.py "style" --type LORA --base-model Illustrious
    python civitai_search.py "realistic" --training-data-only
    python civitai_search.py --hash 0873291ac5
"""

import requests
import json
import time
import argparse
import sys
import hashlib
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional

class CivitaiSearcher:
    def __init__(self, api_key: Optional[str] = None, cache_dir: str = "cache"):
        """
        Initialize the Civitai searcher with optional API key and caching.

        Args:
            api_key: Your Civitai API key, or None for unverified access
            cache_dir: Directory to store cache files
        """
        self.api_key = api_key
        self.base_url = "https://civitai.com/api/v1"
        self.cache_dir = cache_dir
        self.session = requests.Session()
        headers = {'Content-Type': 'application/json'}
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'
        self.session.headers.update(headers)
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_key(self, model_types, base_model_filter, query, training_data_only):
        """Generate a unique cache key for search parameters."""
        key_data = {
            'model_types': sorted(model_types) if model_types else None,
            'base_model_filter': base_model_filter,
            'query': query,
            'training_data_only': training_data_only
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_cache_file(self, cache_key):
        """Get the cache file path for a cache key."""
        return os.path.join(self.cache_dir, f"search_{cache_key}.json")
    
    def _load_cache(self, cache_key, max_age_hours=24):
        """
        Load cached results if they exist and are not too old.
        
        Args:
            cache_key: Cache key to load
            max_age_hours: Maximum age of cache in hours
            
        Returns:
            Cached data or None if not found/expired
        """
        cache_file = self._get_cache_file(cache_key)
        
        if not os.path.exists(cache_file):
            return None
        
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            
            # Check if cache is too old
            cache_time = datetime.fromisoformat(cache_data['timestamp'])
            max_age = timedelta(hours=max_age_hours)
            
            if datetime.now() - cache_time > max_age:
                print(f"Cache expired (older than {max_age_hours} hours)")
                return None
            
            print(f"Loading {len(cache_data['results'])} cached results from {cache_time.strftime('%Y-%m-%d %H:%M:%S')}")
            return cache_data
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Invalid cache file, ignoring: {e}")
            return None
    
    def _merge_results(self, cached_results, new_results):
        """
        Merge cached and new results, avoiding duplicates.
        
        Args:
            cached_results: Previously cached results
            new_results: New results from API
            
        Returns:
            Merged results list
        """
        # Create a set of existing model IDs
        existing_ids = {result['id'] for result in cached_results}
        
        # Add new results that aren't already cached
        merged = cached_results.copy()
        new_count = 0
        
        for result in new_results:
            if result['id'] not in existing_ids:
                merged.append(result)
                existing_ids.add(result['id'])
                new_count += 1
        
        if new_count > 0:
            print(f"Added {new_count} new results to {len(cached_results)} cached results")
        
        return merged
    
    def _save_cache(self, cache_key, results, search_params):
        """
        Save results to cache.
        
        Args:
            cache_key: Cache key to save under
            results: Results to cache
            search_params: Search parameters used
        """
        cache_file = self._get_cache_file(cache_key)
        
        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'search_params': search_params,
            'results': results
        }
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            print(f"Saved {len(results)} results to cache")
        except Exception as e:
            print(f"Failed to save cache: {e}")
    
    def search_by_hash(self, model_hash: str, training_data_only: bool = False) -> Optional[Dict]:
        """
        Search for a specific model by its hash.
        
        Args:
            model_hash: The model hash (e.g., "0873291ac5")
            training_data_only: Only return if model has training data
            
        Returns:
            Model information if found, None otherwise
        """
        print(f"🔍 Searching for model with hash: {model_hash}")
        if training_data_only:
            print("Filter: Training data only")
        print("-" * 50)
        
        try:
            # Use the specific hash lookup endpoint
            response = self.session.get(f"{self.base_url}/model-versions/by-hash/{model_hash}")
            response.raise_for_status()
            version_data = response.json()
            
            print(f"✅ Found model version with hash: {model_hash}")
            
            # Get the full model details using the model ID
            model_id = version_data.get('modelId')
            if not model_id:
                print("❌ Model ID not found in version data")
                return None
            
            # Fetch full model data
            model_response = self.session.get(f"{self.base_url}/models/{model_id}")
            model_response.raise_for_status()
            model_data = model_response.json()
            
            print(f"✅ Retrieved full model data: {model_data.get('name', 'Unknown')}")
            
            # Check training data
            has_training_data = self._model_has_training_data(model_data)
            
            if training_data_only and not has_training_data:
                print(f"❌ Model found but has no training data (filtered out)")
                return None
            
            # Extract model info
            model_info = self._extract_model_info(model_data)
            model_info['matched_hash'] = model_hash
            model_info['hash_type'] = 'Direct Hash Lookup'
            model_info['has_training_data'] = has_training_data
            model_info['matched_version'] = version_data
            
            if has_training_data:
                print(f"✅ Model has training data")
            else:
                print(f"⚠️  Model has no training data")
            
            return model_info
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                print(f"❌ No model found with hash: {model_hash}")
                print("   This could mean:")
                print("   - The hash is incorrect")
                print("   - The model is not publicly available")
                print("   - The hash format is not recognized by Civitai")
            else:
                print(f"❌ HTTP Error {e.response.status_code}: {e}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"❌ Error searching by hash: {e}")
            return None
    
    def search_models(
        self,
        model_types: Optional[List[str]] = None,
        base_model_filter: Optional[str] = None,
        query: Optional[str] = None,
        training_data_only: bool = False,
        limit_per_page: int = 100,
        max_pages: int = 10,
        use_cache: bool = True,
        cache_max_age_hours: int = 24,
        exclude_terms: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Search for models with optional training data filter.
        
        Args:
            model_types: List of model types to filter by (e.g., ['LORA', 'Checkpoint'])
            base_model_filter: Filter by base model (e.g., 'Illustrious', 'SDXL', 'SD 1.5')
            query: Search query string
            training_data_only: Only return models that include training data
            limit_per_page: Number of results per API request (max 100)
            max_pages: Maximum number of pages to search through
            use_cache: Whether to use cached results
            cache_max_age_hours: Maximum age of cache in hours
            exclude_terms: List of terms to exclude from results (case-insensitive)
            
        Returns:
            List of models
        """
        # Generate cache key (include exclude_terms in cache key)
        cache_key = self._get_cache_key(model_types, base_model_filter, query, training_data_only)
        if exclude_terms:
            # Add exclude terms to cache key
            exclude_str = ','.join(sorted(exclude_terms))
            cache_key = hashlib.md5(f"{cache_key}_{exclude_str}".encode()).hexdigest()
        
        search_params = {
            'model_types': model_types,
            'base_model_filter': base_model_filter,
            'query': query,
            'training_data_only': training_data_only,
            'max_pages': max_pages
        }
        
        print(f"Searching for models...")
        if model_types:
            print(f"Model types: {', '.join(model_types)}")
        if base_model_filter:
            print(f"Base model filter: {base_model_filter}")
        if query:
            print(f"Query: {query}")
        if training_data_only:
            print(f"Filter: Training data only")
        if exclude_terms:
            print(f"Excluding terms: {', '.join(exclude_terms)}")
        print("-" * 50)
        
        # Try to load from cache
        cached_data = None
        if use_cache:
            cached_data = self._load_cache(cache_key, cache_max_age_hours)
        
        # If we have cached data, start from there
        if cached_data:
            cached_results = cached_data['results']
            cached_pages = cached_data.get('pages_searched', 0)
            print(f"Found {len(cached_results)} cached results from {cached_pages} previous pages")
            
            # If we've already searched the requested number of pages, return cache
            if cached_pages >= max_pages:
                print("Cache covers requested search depth, using cached results")
                return cached_results
            
            # Otherwise, continue searching from where we left off
            start_page = cached_pages + 1
            remaining_pages = max_pages - cached_pages
            print(f"Searching {remaining_pages} additional pages (starting from page {start_page})")
        else:
            cached_results = []
            start_page = 1
            remaining_pages = max_pages
            print(f"No cache found, searching {remaining_pages} pages from the beginning")
        
        # Search for new results
        new_results = self._search_api(
            model_types=model_types,
            base_model_filter=base_model_filter,
            query=query,
            training_data_only=training_data_only,
            limit_per_page=limit_per_page,
            max_pages=remaining_pages,
            start_page=start_page,
            exclude_terms=exclude_terms
        )
        
        # Merge results
        all_results = self._merge_results(cached_results, new_results)
        
        # Save updated cache
        if use_cache:
            # Update search params with total pages searched
            search_params['pages_searched'] = max_pages
            self._save_cache(cache_key, all_results, search_params)
        
        return all_results
    
    def _search_api(
        self,
        model_types: Optional[List[str]] = None,
        base_model_filter: Optional[str] = None,
        query: Optional[str] = None,
        training_data_only: bool = False,
        limit_per_page: int = 100,
        max_pages: int = 10,
        start_page: int = 1,
        exclude_terms: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Perform the actual API search.
        
        Returns:
            List of models
        """
        found_models = []
        page_count = start_page - 1  # Will be incremented to start_page
        cursor = None
        
        # Convert exclude terms to lowercase for case-insensitive matching
        exclude_terms_lower = [term.lower() for term in exclude_terms] if exclude_terms else []
        
        while page_count < start_page + max_pages - 1:
            page_count += 1
            print(f"Searching page {page_count}...")
            
            # Build API request parameters - use cursor instead of page for search queries
            params = {
                'limit': limit_per_page,
                'primaryFileOnly': False
            }
            
            # Use cursor-based pagination for search queries
            if query:
                if cursor:
                    params['cursor'] = cursor
                # Don't use 'page' parameter with search queries
            else:
                # Use page-based pagination for non-search queries
                params['page'] = page_count
            
            if model_types:
                # Join multiple types with comma for API
                params['types'] = ','.join(model_types)
            if query:
                params['query'] = query
                
            try:
                response = self.session.get(f"{self.base_url}/models", params=params)
                response.raise_for_status()
                data = response.json()
                
                items = data.get('items', [])
                print(f"Found {len(items)} models on page {page_count}")
                
                if not items:
                    print("No more results found.")
                    break
                
                # Process each model
                found_on_page = 0
                for model in items:
                    # Check if model should be excluded based on exclude terms
                    if exclude_terms_lower:
                        model_name = model.get('name', '').lower()
                        model_tags = [tag.lower() for tag in model.get('tags', [])]
                        
                        # Check if any exclude term matches name or tags
                        should_exclude = False
                        for exclude_term in exclude_terms_lower:
                            if exclude_term in model_name or exclude_term in model_tags:
                                print(f"⊘ Excluded (matches '{exclude_term}'): {model.get('name', 'Unknown')}")
                                should_exclude = True
                                break
                        
                        if should_exclude:
                            continue
                    
                    # Check training data filter
                    has_training_data = self._model_has_training_data(model, base_model_filter)
                    
                    if training_data_only and not has_training_data:
                        continue  # Skip models without training data if filter is on
                    
                    # Extract model info
                    model_info = self._extract_model_info(model)
                    model_info['has_training_data'] = has_training_data
                    found_models.append(model_info)
                    
                    if has_training_data and training_data_only:
                        print(f"✅ Found (with training data): {model_info['name']}")
                    elif training_data_only:
                        continue  # This shouldn't happen due to filter above
                    else:
                        training_indicator = " 📚" if has_training_data else ""
                        print(f"✅ Found{training_indicator}: {model_info['name']}")
                    
                    found_on_page += 1
                
                if training_data_only:
                    print(f"Found {found_on_page} models with training data on page {page_count}")
                else:
                    print(f"Found {found_on_page} models on page {page_count}")
                
                # Handle pagination
                metadata = data.get('metadata', {})
                
                if query:
                    # Cursor-based pagination
                    next_cursor = metadata.get('nextCursor')
                    if not next_cursor:
                        print("No more pages available (no nextCursor)")
                        break
                    cursor = next_cursor
                else:
                    # Page-based pagination
                    total_pages = metadata.get('totalPages', 1)
                    print(f"Page {page_count} of {total_pages}")
                    if page_count >= total_pages:
                        break
                
                # Rate limiting - be nice to the API
                time.sleep(0.5)
                
            except requests.exceptions.RequestException as e:
                print(f"Error fetching data: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    print(f"Response content: {e.response.text[:500]}")
                break
        
        return found_models
    
    def _model_has_training_data(self, model: Dict, base_model_filter: Optional[str] = None) -> bool:
        """
        Check if a model has training data files and matches base model filter.
        
        Args:
            model: Model data from API
            base_model_filter: Base model to filter by
            
        Returns:
            True if model has training data and matches filters
        """
        has_training_data = False
        
        # Less verbose debug output
        model_name = model.get('name', 'Unknown')
        model_type = model.get('type', 'Unknown')
        
        # Check each model version
        for version in model.get('modelVersions', []):
            # Check base model filter first
            version_base_model = version.get('baseModel', '')
            
            if base_model_filter:
                if base_model_filter.lower() not in version_base_model.lower():
                    continue
            
            # Check for training data files
            files = version.get('files', [])
            
            for file in files:
                file_type = file.get('type', '').lower()
                file_name = file.get('name', '').lower()
                
                # Look for training data indicators - be more flexible
                if (file_type == 'training data' or 
                    'training' in file_name or 
                    'dataset' in file_name or
                    file_name.endswith('_training_data.zip') or
                    (file_name.endswith('.zip') and ('train' in file_name or 'data' in file_name))):
                    print(f"  ✅ {model_name} has training data: {file_name}")
                    has_training_data = True
                    break
            
            if has_training_data:
                break
        
        return has_training_data
    
    def _extract_model_info(self, model: Dict) -> Dict:
        """
        Extract relevant information from model data.
        
        Args:
            model: Model data from API
            
        Returns:
            Dictionary with model information
        """
        model_id = model.get('id')
        model_url = f"https://civitai.com/models/{model_id}" if model_id else "N/A"
        
        # Get training data files info with download links
        training_files = []
        base_models = set()
        
        for version in model.get('modelVersions', []):
            if version.get('baseModel'):
                base_models.add(version['baseModel'])
                
            for file in version.get('files', []):
                file_type = file.get('type', '').lower()
                file_name = file.get('name', '')
                
                if (file_type == 'training data' or 
                    'training' in file_name.lower() or 
                    'dataset' in file_name.lower() or
                    (file_name.lower().endswith('.zip') and ('train' in file_name.lower() or 'data' in file_name.lower()))):
                    
                    # Get download URL - try multiple possible fields
                    download_url = file.get('downloadUrl')
                    if not download_url:
                        # Construct download URL using model version ID
                        version_id = version.get('id')
                        if version_id:
                            download_url = f"https://civitai.com/api/download/models/{version_id}"
                            # If this is not the primary file, we might need to specify the file
                            if not file.get('primary', False):
                                file_id = file.get('id')
                                if file_id:
                                    download_url = f"https://civitai.com/api/download/models/{version_id}?type=Training%20Data"
                    
                    training_files.append({
                        'name': file_name,
                        'size_kb': file.get('sizeKB', 0),
                        'version': version.get('name', 'Unknown'),
                        'version_id': version.get('id'),
                        'file_id': file.get('id'),
                        'download_url': download_url,
                        'is_primary': file.get('primary', False)
                    })
        
        return {
            'id': model_id,
            'name': model.get('name', 'Unknown'),
            'type': model.get('type', 'Unknown'),
            'url': model_url,
            'creator': model.get('creator', {}).get('username', 'Unknown'),
            'base_models': list(base_models),
            'training_files': training_files,
            'download_count': model.get('stats', {}).get('downloadCount', 0),
            'rating': model.get('stats', {}).get('rating', 0),
            'nsfw': model.get('nsfw', False)
        }
    
    def print_results(self, results: List[Dict], max_results: Optional[int] = None):
        """
        Print search results in a formatted way.
        
        Args:
            results: List of model information dictionaries
            max_results: Maximum number of results to display
        """
        if not results:
            print("No models found.")
            return
        
        # Separate results by training data availability
        with_training = [r for r in results if r.get('has_training_data', False)]
        without_training = [r for r in results if not r.get('has_training_data', False)]
        
        print(f"\n{'='*60}")
        print(f"Found {len(results)} models total")
        if with_training:
            print(f"📚 {len(with_training)} with training data")
        if without_training:
            print(f"📋 {len(without_training)} without training data")
        print(f"{'='*60}")
        
        display_results = results[:max_results] if max_results else results
        
        for i, model in enumerate(display_results, 1):
            training_icon = "📚" if model.get('has_training_data', False) else "📋"
            print(f"\n{i}. {training_icon} {model['name']}")
            print(f"   Type: {model['type']}")
            print(f"   Creator: {model['creator']}")
            print(f"   Base Models: {', '.join(model['base_models'])}")
            print(f"   URL: {model['url']}")
            print(f"   Downloads: {model['download_count']:,}")
            print(f"   Rating: {model['rating']:.1f}")
            print(f"   NSFW: {model['nsfw']}")
            
            # Show training data files if available
            if model.get('has_training_data', False) and model.get('training_files'):
                print(f"   🎯 Training Data Files:")
                for tf in model['training_files']:
                    size_mb = tf['size_kb'] / 1024 if tf['size_kb'] > 0 else 0
                    print(f"     📁 {tf['name']} ({size_mb:.1f} MB)")
                    print(f"        Version: {tf['version']}")
                    if tf['download_url']:
                        print(f"        💾 Download: {tf['download_url']}")
                    print()
        
        if max_results and len(results) > max_results:
            print(f"\n... and {len(results) - max_results} more results")
        
        # Summary of download commands for training data
        training_models = [m for m in display_results if m.get('has_training_data', False) and m.get('training_files')]
        if training_models:
            print(f"\n{'='*60}")
            print("💾 Training Data Download Commands:")
            print("(You may need to add ?token=YOUR_API_KEY for private models)")
            print("="*60)
            
            for model in training_models:
                if model['training_files']:
                    for tf in model['training_files']:
                        if tf['download_url']:
                            print(f"# {model['name']} - {tf['name']}")
                            print(f"curl -L -o \"{tf['name']}\" \"{tf['download_url']}\"")
                            print(f"# Or with wget:")
                            print(f"wget --content-disposition \"{tf['download_url']}\"")
                            print()
    
    def print_hash_result(self, result: Optional[Dict]):
        """
        Print result from hash search.
        
        Args:
            result: Model information dictionary or None
        """
        if not result:
            print("No model found with that hash.")
            return
        
        print(f"\n{'='*60}")
        print(f"Model Found by Hash")
        print(f"{'='*60}")
        
        training_icon = "📚" if result.get('has_training_data', False) else "📋"
        print(f"\n{training_icon} {result['name']}")
        print(f"   Type: {result['type']}")
        print(f"   Creator: {result['creator']}")
        print(f"   Base Models: {', '.join(result['base_models'])}")
        print(f"   URL: {result['url']}")
        print(f"   Downloads: {result['download_count']:,}")
        print(f"   Rating: {result['rating']:.1f}")
        print(f"   NSFW: {result['nsfw']}")
        print(f"   Matched Hash: {result['matched_hash']} ({result.get('hash_type', 'Unknown')})")
        
        if result.get('has_training_data', False):
            if result.get('training_files'):
                print(f"\n   🎯 Training Data Files:")
                for tf in result['training_files']:
                    size_mb = tf['size_kb'] / 1024 if tf['size_kb'] > 0 else 0
                    print(f"     📁 {tf['name']} ({size_mb:.1f} MB)")
                    print(f"        Version: {tf['version']}")
                    if tf['download_url']:
                        print(f"        💾 Download: {tf['download_url']}")
                    print()
                
                print(f"\n{'='*60}")
                print("💾 Download Commands:")
                print("="*60)
                for tf in result['training_files']:
                    if tf['download_url']:
                        print(f"# {result['name']} - {tf['name']}")
                        print(f"curl -L -o \"{tf['name']}\" \"{tf['download_url']}\"")
                        print()
            else:
                print(f"\n   ⚠️  Model has training data but files not accessible")
        else:
            print(f"\n   ❌ This model does not include training data")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Search Civitai for models with various filters',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # General searches
  python civitai_search.py "anime character"
  python civitai_search.py "style" --type LORA --base-model Illustrious
  
  # Exclude terms (filter out unwanted results)
  python civitai_search.py "character" --exclude nsfw --exclude realistic
  python civitai_search.py "anime" --type LORA --exclude "pony"
  
  # Training data searches
  python civitai_search.py "realistic" --training-data-only
  python civitai_search.py "character" --type LORA --training-data-only
  
  # Hash searches
  python civitai_search.py --hash 0873291ac5
  python civitai_search.py --hash "f422db7192" --training-data-only
        """
    )
    
    # Make query optional when using hash search
    parser.add_argument(
        'query',
        nargs='?',
        help='Search query (like Civitai search box) - REQUIRED unless using --hash'
    )
    
    parser.add_argument(
        '--hash', '-H',
        help='Search by model hash (e.g., from PNG metadata: "0873291ac5")'
    )
    
    parser.add_argument(
        '--training-data-only', '-T',
        action='store_true',
        help='Only show models that include training data files'
    )
    
    parser.add_argument(
        '--type', '-t',
        action='append',
        choices=['LORA', 'Checkpoint', 'TextualInversion', 'Hypernetwork', 'AestheticGradient', 'Controlnet', 'Poses'],
        help='Model type(s) to search for (can be used multiple times)'
    )
    
    parser.add_argument(
        '--base-model', '-b',
        help='Filter by base model (e.g., Illustrious, SDXL, "SD 1.5", Flux)'
    )
    
    parser.add_argument(
        '--exclude', '-e',
        action='append',
        help='Exclude models containing this term in name or tags (case-insensitive, can be used multiple times)'
    )
    
    parser.add_argument(
        '--max-pages', '-p',
        type=int,
        default=5,
        help='Maximum number of pages to search (default: 5, each page = 100 models)'
    )
    
    parser.add_argument(
        '--max-results', '-r',
        type=int,
        help='Maximum number of results to display (default: show all)'
    )
    
    parser.add_argument(
        '--api-key', '-k',
        help='Civitai API key (or set in script)'
    )
    
    parser.add_argument(
        '--save', '-s',
        action='store_true',
        help='Save results to JSON file'
    )
    
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable caching and always fetch fresh results'
    )
    
    parser.add_argument(
        '--cache-max-age',
        type=int,
        default=24,
        help='Maximum age of cache in hours (default: 24)'
    )
    
    parser.add_argument(
        '--clear-cache',
        action='store_true',
        help='Clear all cached results before searching'
    )
    
    return parser.parse_args()


def load_api_key():
    """Load API key from CIVITAI_API_KEY env var or key.txt, or return None."""
    env_key = os.environ.get('CIVITAI_API_KEY', '').strip()
    if env_key and env_key != "YOUR_API_KEY_HERE":
        return env_key
    try:
        with open('key.txt', 'r') as f:
            api_key = f.read().strip()
            if api_key and api_key != "YOUR_API_KEY_HERE":
                return api_key
    except FileNotFoundError:
        pass
    return None


def clear_cache(cache_dir="cache"):
    """Clear all cached files."""
    if not os.path.exists(cache_dir):
        print("No cache directory found.")
        return
    
    cache_files = [f for f in os.listdir(cache_dir) if f.startswith('search_') and f.endswith('.json')]
    
    if not cache_files:
        print("No cache files found.")
        return
    
    for cache_file in cache_files:
        os.remove(os.path.join(cache_dir, cache_file))
    
    print(f"Cleared {len(cache_files)} cache files.")


def main():
    """Main function to run the search script."""
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Clear cache if requested
    if args.clear_cache:
        clear_cache()
        return
    
    # Validate arguments
    if not args.query and not args.hash:
        print("❌ Error: Must provide either a search query or --hash parameter")
        print("Examples:")
        print("  python civitai_search.py \"anime character\"")
        print("  python civitai_search.py --hash 0873291ac5")
        print("  python civitai_search.py \"style\" --training-data-only")
        sys.exit(1)
    
    # Get API key - try multiple sources, fall back to unverified access
    API_KEY = args.api_key or load_api_key()
    if API_KEY == "YOUR_API_KEY_HERE":
        API_KEY = None

    if not API_KEY:
        print("⚠️  No API key found — using unverified (free) access. Some results may be limited.")
        print("   Set CIVITAI_API_KEY env var or create 'key.txt' for full access.")

    # Initialize searcher
    searcher = CivitaiSearcher(API_KEY)
    
    # Handle hash search
    if args.hash:
        result = searcher.search_by_hash(args.hash, training_data_only=args.training_data_only)
        searcher.print_hash_result(result)
        return
    
    # Handle regular search
    # Display search parameters
    print(f"🔍 Searching for: '{args.query}'")
    if args.type:
        print(f"📋 Model types: {', '.join(args.type)}")
    if args.base_model:
        print(f"🎯 Base model: {args.base_model}")
    if args.training_data_only:
        print(f"🎯 Filter: Training data only")
    if args.exclude:
        print(f"⊘ Excluding: {', '.join(args.exclude)}")
    print(f"📄 Max pages: {args.max_pages}")
    if args.no_cache:
        print("🚫 Cache disabled")
    else:
        print(f"💾 Cache enabled (max age: {args.cache_max_age} hours)")
    print("-" * 50)
    
    # Perform search
    results = searcher.search_models(
        model_types=args.type,
        base_model_filter=args.base_model,
        query=args.query,
        training_data_only=args.training_data_only,
        max_pages=args.max_pages,
        use_cache=not args.no_cache,
        cache_max_age_hours=args.cache_max_age,
        exclude_terms=args.exclude
    )
    
    # Display results
    searcher.print_results(results, max_results=args.max_results)
    
    # Save results if requested
    if args.save and results:
        filename_suffix = "_training_data" if args.training_data_only else "_all"
        filename = f"civitai_search_{args.query.replace(' ', '_')}{filename_suffix}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to '{filename}'")
    elif results:
        print(f"\n💡 Use --save to save results to JSON file")


if __name__ == "__main__":
    main()