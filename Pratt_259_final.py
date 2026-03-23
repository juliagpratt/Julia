# %%
import pandas as pd
import numpy as np
memorydata = pd.read_csv("/Users/juliapratt/Desktop/memory_transcripts259.csv")

# %%
#Confirm data look good
print(memorydata.head())
print(memorydata.dtypes)
#Select only relevant columns and rename
memorydata = memorydata[["MemoryTxt", "Frequency"]].copy()
memorydata = memorydata.rename(columns={
    "MemoryTxt": "transcript",
    "Frequency": "rumination"
})
print(memorydata.head())

# %%
#Visualize distribution of rumination scores
import matplotlib.pyplot as plt

plt.hist(memorydata["rumination"], bins=30)
plt.xlabel("Rumination Rating")
plt.ylabel("Frequency")
plt.title("Distribution of Rumination Scores")
plt.show()

#Mean rumination score
mean_rumination = memorydata["rumination"].mean()
print("Mean rumination score:", mean_rumination) #2.62
#Standard deviation
std_rumination = memorydata["rumination"].std()
print("Standard deviation of rumination scores:", std_rumination) #1.44

# %%
###MPNet 

##Get embeddings

#Load embedding model
from sentence_transformers import SentenceTransformer
#Load MPNet
embed_model = SentenceTransformer("all-mpnet-base-v2")

#Turn transcripts into list
texts = memorydata["transcript"].tolist()

#Generate embeddings!
embeddings = embed_model.encode(texts, show_progress_bar=True)
print(type(embeddings))
print(embeddings.shape) #1025 memories, 768 dimensions

##Predict rumination
#Using ridge regression bc there are so many features and we don't want to overfit
y = memorydata["rumination"].values 

#Split the data into training and testing sets
#Training on 80% of the data, testing on 20%
from sklearn.model_selection import train_test_split

X = embeddings

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print(X_train.shape, X_test.shape) #Dimensions look good!


#Fit the model
from sklearn.linear_model import Ridge

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

#Predict!
y_pred = model.predict(X_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
corr = np.corrcoef(y_test, y_pred)[0, 1]

print("R²:", r2) #0.01913
print("Correlation:", corr) #0.2105 

# %%
##Visualizations
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)
plt.xlabel("Actual rumination")
plt.ylabel("Predicted rumination")
plt.title("MPNet: Predicted vs Actual")
m, b = np.polyfit(y_test, y_pred, 1)
plt.plot(y_test, m*y_test + b)

plt.show()



# %%
###RoBERTa 

#Load model
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
roberta_model = AutoModel.from_pretrained("roberta-base")

##Get embeddings
# Function to  return a vector for each memory (768 dimensions like MPNet)
def get_roberta_embedding(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    with torch.no_grad():
        outputs = roberta_model(**inputs)

    token_embeddings = outputs.last_hidden_state
    
    #Average across tokens
    sentence_embedding = token_embeddings.mean(dim=1)
    
    return sentence_embedding.squeeze().numpy()

#Run on all transcripts (loop)
roberta_embeddings = []

for i, text in enumerate(texts):
    emb = get_roberta_embedding(text)
    roberta_embeddings.append(emb)
    
    if (i + 1) % 50 == 0:
        print(f"Processed {i+1} texts")

roberta_embeddings = np.array(roberta_embeddings)

## Split into training and testing
#**Same random state as MPNet for comparison!
X_roberta = roberta_embeddings
y = memorydata["rumination"].values #defining again, but same as before

from sklearn.model_selection import train_test_split

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_roberta, y,
    test_size=0.2,
    random_state=42
)

##Fit model
roberta_reg = Ridge(alpha=1.0)
roberta_reg.fit(Xr_train, yr_train)

yr_pred = roberta_reg.predict(Xr_test)

r2_roberta = r2_score(yr_test, yr_pred)
corr_roberta = np.corrcoef(yr_test, yr_pred)[0, 1]

print("RoBERTa R²:", r2_roberta) #0.012
print("RoBERTa correlation:", corr_roberta) #0.2879

##Visualize!
# %% Plot RoBERTa predicted vs actual
plt.scatter(yr_test, yr_pred)
plt.xlabel("Actual rumination")
plt.ylabel("Predicted rumination")
plt.title("RoBERTa: Predicted vs Actual")
m, b = np.polyfit(yr_test, yr_pred, 1)
plt.plot(yr_test, m * yr_test + b)

plt.show()

# %%
###Compare predictions!

print("MPNet R²:", r2)
print("MPNet correlation:", corr)

print("RoBERTa R²:", r2_roberta)
print("RoBERTa correlation:", corr_roberta)

##Significance test (bootstrapping)
n_boot = 1000
diffs = []

n = len(y_test)

for _ in range(n_boot):
    #Resample indices
    idx = np.random.choice(n, n, replace=True)
    
    y_true = y_test[idx]
    mpnet_pred = y_pred[idx]
    roberta_pred = yr_pred[idx]
    
    #Compute correlations
    r_mp = np.corrcoef(y_true, mpnet_pred)[0, 1]
    r_ro = np.corrcoef(y_true, roberta_pred)[0, 1]
    
    diffs.append(r_ro - r_mp)

diffs = np.array(diffs)

#Get confidence interval!
lower = np.percentile(diffs, 2.5)
upper = np.percentile(diffs, 97.5)
#Not significant

print("Mean difference (RoBERTa - MPNet):", diffs.mean())
print("95% CI:", lower, upper)

##Visualize comparison
#Bar plot of r2 and correlations
models = ["MPNet", "RoBERTa"]
r2_values = [r2, r2_roberta]
corr_values = [corr, corr_roberta]

plt.figure(figsize=(6, 4))
plt.bar(models, r2_values)
plt.ylabel("R²")
plt.title("Model comparison: R²")
plt.show()

plt.figure(figsize=(6, 4))
plt.bar(models, corr_values)
plt.ylabel("Correlation")
plt.title("Model comparison: correlation")
plt.show()
# %%
#Plot residuals vs predicted for both models
mpnet_residuals = np.asarray(y_test - y_pred).ravel()
roberta_residuals = np.asarray(yr_test - yr_pred).ravel()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(y_pred, mpnet_residuals)
axes[0].axhline(0, color="red", linestyle="--")
axes[0].set_xlabel("MPNet Predicted")
axes[0].set_ylabel("Residuals")
axes[0].set_title("MPNet Residuals vs Predicted")

axes[1].scatter(yr_pred, roberta_residuals)
axes[1].axhline(0, color="red", linestyle="--")
axes[1].set_xlabel("RoBERTa Predicted")
axes[1].set_ylabel("Residuals")
axes[1].set_title("RoBERTa Residuals vs Predicted")

plt.tight_layout()
plt.show()
