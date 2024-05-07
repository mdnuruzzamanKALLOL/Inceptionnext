import argparse
import time
import csv
import json
import tensorflow as tf
from tensorflow.keras import applications as tf_models

# Setup argparse for command line arguments
parser = argparse.ArgumentParser(description='TensorFlow Model Benchmark')
parser.add_argument('--model', default='ResNet50', help='Model name as in tf.keras.applications')
parser.add_argument('-b', '--batch-size', default=128, type=int, help='Mini-batch size')
parser.add_argument('--num-warmup', default=5, type=int, help='Number of warmup iterations')
parser.add_argument('--num-iter', default=50, type=int, help='Number of benchmark iterations')
parser.add_argument('--results-file', default='benchmark_results.csv', type=str, help='CSV file to store the benchmark results')

def load_model(model_name):
    """Load a model from TensorFlow.keras.applications"""
    if hasattr(tf_models, model_name):
        model = getattr(tf_models, model_name)(weights=None)
        return model
    else:
        raise ValueError(f"Model {model_name} is not available in tf.keras.applications")

def benchmark(model, batch_size, num_warmup, num_iter):
    """Run benchmarking for the given model"""
    input_shape = model.input_shape[1:]  # Get input shape, discard batch size dimension
    dummy_input = tf.random.normal([batch_size] + list(input_shape))

    # Warm-up runs
    for _ in range(num_warmup):
        _ = model(dummy_input)

    # Benchmark runs
    start_time = time.time()
    for _ in range(num_iter):
        _ = model(dummy_input)
    total_time = time.time() - start_time

    # Calculate performance metrics
    avg_time_per_iter = total_time / num_iter
    throughput = batch_size / avg_time_per_iter

    return {
        'model': model.name,
        'batch_size': batch_size,
        'avg_time_per_iter': avg_time_per_iter,
        'throughput': throughput
    }

def main():
    args = parser.parse_args()
    model = load_model(args.model)
    results = benchmark(model, args.batch_size, args.num_warmup, args.num_iter)

    # Print results
    print(json.dumps(results, indent=4))

    # Write results to CSV
    with open(args.results_file, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=results.keys())
        writer.writeheader()
        writer.writerow(results)

if __name__ == '__main__':
    main()
