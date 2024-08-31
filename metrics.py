from main import predict_tomorrow_price
import json
import threading
from sortedcollections import SortedDict
import time

def run_sequential(stocks=[]):
    """
    Run the prediction algorithm without threading
    """
    future_prices = SortedDict()

    # Get the start time
    start_time = time.time()

    # Run prediction on each stock
    for stock in stocks:
        predict_tomorrow_price(stock, future_prices)

    # Get final time
    end_time = time.time()

    tot_time = end_time - start_time
    return tot_time

def run_parallel(stocks=[]):
    # Using Threading
    future_prices = SortedDict()
    threads = []
    lock = threading.Lock()

    # Get start time
    start_time = time.time()

    for symbol in stocks:
        # Create a thread for each stock
        thread = threading.Thread(target=predict_tomorrow_price, args=(symbol,future_prices,))
        threads.append(thread)
        thread.start()  # Start the thread
        

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Get end time
    end_time = time.time()

    tot_time = end_time - start_time
    return tot_time
    


# Run comparisons on each benchmark
benchmarks = json.load(open('benchmarks.json'))
trial_results = []

# Write to .txt file to save results
with open('results.txt', 'w') as file:
    for test in benchmarks:
        seq_time = run_sequential(benchmarks[test])
        print(f"Completed {test} for sequential")
        file.write(f"Completed {test} for sequential\n")

        parallel_time = run_parallel(benchmarks[test])
        print(f"Completed {test} for parallel")
        file.write(f"Completed {test} for parallel\n")

        trial_results.append([seq_time, parallel_time])


    # Recorde the average difference
    avg_diff = 0
    tot_time_seq = tot_time_parallel = 0

    for i, result in enumerate(trial_results):
        # How much extra time seq needed
        difference = result[0] - result[1]
        tot_time_seq += result[0]
        tot_time_parallel += result[1]
        avg_diff += difference
        print(f"Difference for test{i} is: {difference:.2f}")
        file.write(f"Difference for test{i} is: {difference:.2f}\n")

    print(f"Overall average difference is: {avg_diff/len(trial_results):.2f}s")
    speedup = tot_time_seq/tot_time_parallel
    print(f"The speedup for parallel is: {speedup:.2f}")
    
    file.write(f"Overall average difference is: {avg_diff/len(trial_results):.2f}s\n")
    file.write(f"The speedup for parallel is: {speedup:.2f}\n")
