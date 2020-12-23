# %%
import pandas as pd
import numpy as np
import seaborn as sns

# %%
df = pd.read_csv("./train.csv")
df.describe()

# %%
df.head()

# %%
def add_okay_column(df):
    """
    agrega columna con 1 si el texto tiene algo toxico, 0 si esta
    todo piola
    """
    is_not_ok = np.zeros(len(df.index))
    columns = [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    ]

    for index, row in df.iterrows():
        if any(list(row[columns])):
            is_not_ok[index] = 1

    df2 = df.assign(is_not_ok = is_not_ok)

    return df2

# %%
# dataframes por columna
toxic = df[df["toxic"] == 1]
severe_toxic = df[df["severe_toxic"] == 1]
obscene = df[df["obscene"] == 1]
threat = df[df["threat"] == 1]
insult = df[df["insult"] == 1]
identity_hate = df[df["identity_hate"] == 1]

# %%
# libreria para hacer procesamiento de lenguaje natural
import spacy

nlp = spacy.load("en_core_web_sm")

# %%
def count_upper_tokens(df):
    """
    Cuenta cantidad de palabras en mayuscula que hay
    en todas las filas de un un dataframe.
    """
    counter = 0

    for index, row in df.iterrows():
        # Esta funcion tokeniza un documento.
        doc = nlp(row["comment_text"])
        for tok in doc:
            # Por cada token sumar 1 si el mismo está en mayuscula
            if tok.is_upper:
                counter += 1

    return counter

# %%
upper_count_toxic = count_upper_tokens(toxic)
upper_count_severe_toxic = count_upper_tokens(severe_toxic)
upper_count_obscene = count_upper_tokens(obscene)
upper_count_threat = count_upper_tokens(threat)
upper_count_insult = count_upper_tokens(insult)
upper_count_identity_hate = count_upper_tokens(identity_hate)

# %%
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.barplot(
    x=[
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    ],
    y=[
        upper_count_toxic,
        upper_count_severe_toxic,
        upper_count_obscene,
        upper_count_threat,
        upper_count_insult,
        upper_count_identity_hate,
    ])

# %%
def df_difference(df1, df2):
    """
    Devuelve las filas que están en el dataframe df1 y NO en el df2.
    """
    diff = df1.merge(
        df2, how ='outer', indicator=True
    ).loc[
        lambda x : x['_merge']=='left_only'
    ]

    return diff

# %%
from matplotlib_venn import venn2, venn2_circles, venn2_unweighted
from matplotlib_venn import venn3, venn3_circles
from matplotlib import pyplot as plt

venn2(
    subsets=(
        len(insult.index), # cantidad total cosa 1
        len(identity_hate.index), # cantidad total cosa 2
        len(insult.merge(identity_hate, how='inner' ,indicator=False).index) # Cantidad en interseccion
    ),
    set_labels = ('identity_hate', 'insult')
)

# %%
venn2(
    subsets=(
        len(toxic.index),
        len(threat.index),
        len(toxic.merge(threat, how='inner' ,indicator=False).index)
    ),
    set_labels = ('toxic', 'threat')
)
