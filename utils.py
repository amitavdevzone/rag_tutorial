import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Any
from tqdm import tqdm

def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def get_optimal_thread_count() -> int:
    num_cores = os.cpu_count() - 2  # Leaving 2 CPUs idle
    return num_cores * 2  # Typically, 2x the number of cores for I/O-bound tasks

def process_batch(vectorstore, batch: List[Any]) -> str:
    try:
        vectorstore.add_documents(documents=batch)
        return f"Processed batch of size {len(batch)}"
    except Exception as e:
        return f"Error processing batch: {e}"

def add_chunks_to_vectorstore(vectorstore, chunks: List[Any], batch_size: int = 1000) -> List[str]:
    messages = []
    total_batches = len(chunks) // batch_size + (1 if len(chunks) % batch_size else 0)
    num_threads = get_optimal_thread_count()
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_batch = {executor.submit(process_batch, vectorstore, batch): batch for batch in chunk_list(chunks, batch_size)}
        
        for i, future in enumerate(tqdm(as_completed(future_to_batch), total=total_batches, desc="Processing Batches")):
            try:
                result = future.result()
            except Exception as e:
                messages.append(f"Error retrieving result: {e}")

    return messages
