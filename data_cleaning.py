# %%

import pandas as pd
import numpy as np
import sklearn as sklearn
import matplotlib as plt
import seaborn as sns

from sklearn.preprocessing import OrdinalEncoder

# %%
#Filling unnecessary + dropping unneeded columns
airline_test = pd.read_csv('train.csv')
airline_test = airline_test.fillna(0)
airline_test = airline_test.drop(columns='Unnamed: 0')

# %%
#Scaling flight distance to be natural log to decrease right skew of data
airline_test['Flight Distance'] = airline_test['Flight Distance'].apply(np.log)

# %%
#Ordinal encoding object columns
object_cols = [col for col in airline_test.columns if airline_test[col].dtype == 'object']
ordinal_encoder = OrdinalEncoder()
airline_test_copy = airline_test.copy()
airline_test_copy[object_cols] = ordinal_encoder.fit_transform(airline_test_copy[object_cols])

# %%
corr = airline_test.drop(columns=object_cols).corr() 
corr.style.background_gradient(cmap ='coolwarm')

# %%
corr2 = airline_test_copy.corr()
corr2.style.background_gradient(cmap='coolwarm')

# %%

fig, axes = plt.pyplot.subplots(4, 4, figsize=(20, 20))
non_ordinal_int_cols = ['id', 'Age', 'Departure Delay in Minutes']
ordinal_cols = [col for col in airline_test.columns if (airline_test[col].dtype == 'int64') and (col not in non_ordinal_int_cols)]
for i, col in enumerate(ordinal_cols):
    print(col)
    row, column = divmod(i, 4)
    print(row, column)
    sns.countplot(data=airline_test, x = col, hue = 'satisfaction', ax=axes[row, column], )
    print(f"Plotted {col}")
    axes[row, column].set_title(f"{col}")
    for tick in axes[row, column].get_yticks():
        axes[row, column].axhline(y=tick, color='gray', linestyle='--', linewidth=0.5)
    axes[row, column].legend([])
    axes[row, column].set_ylabel('')
    axes[row, column].set_xlabel('')
for j in range(len(ordinal_cols), 16):
    row, column = divmod(j, 4)
    axes[row, column].axis('off')
fig.legend(['Satisfied', 'Not Satisfied'], loc='lower right', bbox_to_anchor=(0.9, 0.1), frameon=False, prop = {'size': 25})
fig.subplots_adjust(left=0.07, right=0.93, top=0.93, bottom=0.07)  # Sets margins for the figure
fig.suptitle("Ordinal Variables vs. Satisfaction", fontsize=30)
fig.supylabel("Number of Customers", fontsize=25)
fig.savefig("ordinalvariables_vs_satisfaction.png", dpi = 300)

# %%

continuous_cols = non_ordinal_int_cols[1:] + [col for col in airline_test.columns if (airline_test[col].dtype == 'float64')]
fig, axes = plt.pyplot.subplots(1, len(continuous_cols), figsize=(25, 10))
for i, col in enumerate(continuous_cols):
    print(col)
    sns.violinplot(hue=airline_test['satisfaction'], y=airline_test[col], ax = axes[i])
    print(f"Plotted {col}")
    axes[i].set_title(f"{col}", fontsize=20)
    axes[i].set_ylabel('')
    axes[i].set_xlabel('')
    axes[i].legend([])
fig.legend(['Satisfied', 'Not Satisfied'], loc='lower left', bbox_to_anchor=(0.05, 0.02), frameon=False, prop = {'size': 10})
fig.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.10)
fig.suptitle('Continuous Variables vs. Satisfaction', fontsize=30)
fig.supxlabel('Satisfaction', fontsize=25)
fig.savefig("continuousvariables_vs_satisfaction.png", dpi=300)

#%%

airline_test_copy.shape[0]
