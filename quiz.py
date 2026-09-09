import csv
import time
from datetime import datetime

correct_count = 0
total_count = 0

with open("data/words.csv", "r") as file:
    reader = csv.DictReader(file)
 
    for row in reader:
     print("French:", row["french"])
     total_count += 1
     
     start_time = time.perf_counter()

     answer = input("English: ").strip().lower()

     end_time = time.perf_counter()
     response_time = round(end_time - start_time, 2)

     timestamp = datetime.now().isoformat(timespec="seconds")

     if answer == row["english"]:
        print("Correct!")
        result = 1
        correct_count += 1
     else:
        print("Incorrect. The answer is:", row["english"])
        result = 0
 
     with open("data/reviews.csv", "a", newline="") as review_file:
        writer = csv.writer(review_file)

        writer.writerow([
        row["id"],
        timestamp,
        result,
        response_time
        ])

print("Quiz finished!")
print("Score:", correct_count, "/", total_count)
       