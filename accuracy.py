from steps.load_artifacts import load_scores

accuracy, precision, recall, f1_score = load_scores()

with open("accuracy.txt", 'w') as f:
    f.write(f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1_score}\n")
