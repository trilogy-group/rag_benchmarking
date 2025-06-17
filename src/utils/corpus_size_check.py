def check_corpus_sizes(file_paths: list[str]) -> dict[str, dict[str, int]]:
    """
    Loads all jsonl files in the given list and returns a dictionary mapping
    file names to their minimum and maximum token sizes.
    
    Args:
        file_paths: List of paths to jsonl files to check
        
    Returns:
        Dictionary mapping file names to a dict containing min and max token sizes
    """
    import json
    from tiktoken import encoding_for_model
    import logging

    logger = logging.getLogger(__name__)
    
    # Initialize tokenizer
    enc = encoding_for_model("gpt-3.5-turbo")
    
    file_stats = {}
    
    for file_path in file_paths:
        logger.info(f"Processing {file_path}...")
        max_tokens = 0
        min_tokens = float('inf')
        max_text = ""
        min_text = ""
        
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    # Convert object to string representation
                    obj_str = json.dumps(obj)
                    # Count tokens
                    tokens = len(enc.encode(obj_str))
                    if tokens > max_tokens:
                        max_tokens = tokens
                        max_text = obj_str
                    if tokens < min_tokens:
                        min_tokens = tokens
                        min_text = obj_str
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse line in {file_path}")
                    continue
                    
        file_stats[file_path] = {
            'min_tokens': min_tokens if min_tokens != float('inf') else 0,
            'max_tokens': max_tokens,
            'min_text': min_text,
            'max_text': max_text
        }
        
        logger.info(f"Token stats for {file_path}:")
        logger.info(f"  Min tokens: {file_stats[file_path]['min_tokens']}")
        logger.debug(f"  Min text example: {min_text[:200]}...")
        logger.info(f"  Max tokens: {file_stats[file_path]['max_tokens']}")
        logger.debug(f"  Max text example: {max_text[:200]}...")
        
    return file_stats


check_corpus_sizes([
    "./data/benchmark_datasets/niah/needlebench/PaulGrahamEssays.jsonl",
    "./data/benchmark_datasets/niah/needlebench/zh_finance.jsonl",
    "./data/benchmark_datasets/niah/needlebench/zh_game.jsonl",
    "./data/benchmark_datasets/niah/needlebench/zh_general.jsonl",
    "./data/benchmark_datasets/niah/needlebench/zh_government.jsonl",
    "./data/benchmark_datasets/niah/needlebench/zh_movie.jsonl",
    "./data/benchmark_datasets/niah/needlebench/zh_tech.jsonl",
])