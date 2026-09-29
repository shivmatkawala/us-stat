study_hours = [2, 4, 3, 8, 6, 9, 5, 5]
marks = [90, 1, 50, 100, 0, 100, 95, 1500]

# -ve  0  +ve
# -1, -0.9, -0.8, -0.7, -0.5, -0.2, 0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 1
# If Correlation value is closer to 1 then relation is high
# If Correlatio  value is closer to 0 then there is no relation   
# If Correlation value is closer to -1 then there is very bad relation

import statistics

correlation_score = statistics.correlation(study_hours, marks)
print(correlation_score)  # There is relation but average relation => 0.55
