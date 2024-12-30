from datasets import load_dataset

dataset = load_dataset('thu-coai/AISafetyLab_Datasets', 'advbench', split='test')
print(dataset)

dataset = load_dataset("thu-coai/AISafetyLab_Datasets", "xstest", split="safe")
print(dataset)

dataset = load_dataset("thu-coai/AISafetyLab_Datasets", "JBB", split="test")
print(dataset)

dataset = load_dataset("thu-coai/AISafetyLab_Datasets", "harmbench", split="standard")
print(dataset)

dataset = load_dataset("thu-coai/AISafetyLab_Datasets", "safe-rlhf")
print(dataset['train'], dataset['test'])

dataset = load_dataset("thu-coai/AISafetyLab_Datasets", "safe-tuning")
print(dataset["train"], dataset["valid"])

dataset = load_dataset("thu-coai/AISafetyLab_Datasets", "safe-unlearning")
print(dataset["train"], dataset["valid"])

