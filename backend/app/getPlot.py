from data_loader import pushup_landmarks

# Indices you want to print
indices = [211, 228, 217]

# Load all push-up landmark data
all_lines = pushup_landmarks()

# Collect and print the 3 objects (each is a list of 99 floats)
selected = [all_lines[i] for i in indices if i < len(all_lines)]
for idx, obj in zip(indices, selected):
    print(f"Line {idx}:")
    print(obj)
    print()  # Blank line between objects

# If you want as a list of lists:
print(selected)