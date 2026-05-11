# %% [markdown]
# # Analiza si preturilor imobiliarelor in New York
# In acest proiect ne-am propus sa prezentam un flux complet de analiza a datelor, curatare, segmentare geografica si modelare statistica pentru piata imobiliara din New York.
# 
# ### Obiective:
# 1. **Curatarea datelor**: Identificarea si eliminarea zgomotului (outliers) si a datelor redundante.
# 2. **Analiza exploratorie**: Intelegerea distributiei variabilelor cheie.
# 3. **Segmentare Geografica**: Gruparea proprietatilor pe zone folosind alogritmul K-Means si identificarea celor mai scumpe / ieftine zone.
# 4. **Modelare Predictiva**: Construirea unui model de regresie pentru estimarea preturilor.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster.vq import kmeans2, whiten
import statsmodels.api as sm
import plotly.express as px
from IPython.display import display

# Setam formatul numerelor pentru a evita notatia stiintifica
pd.options.display.float_format = '{:.2f}'.format
plt.style.use('ggplot') # Stil vizual mai curat

# %% [markdown]
# ## 1. Incarcarea si Inspectia Initiala a Datelor
# Incepem prin a incarca setul de date si a vedea volumul de informatii disponibil.

# %%
filename = "NY-House-Dataset.csv" 
df = pd.read_csv(filename)

print(f"Numar total de inregistrari: {df.shape[0]}")
print(f"Numar total de variabile: {df.shape[1]}")
display(df.describe())

# %% [markdown]
# ## 2. Curatarea Datelor (Data Cleaning)
# In aceasta etapa, eliminam coloanele care nu pot fi folosite in calcule matematice (adrese brute), gestionam duplicatele si eliminam outlierii.
# *Nota: Nu au fost gasite valori nule in acest set de date, acestea par a fi precuratate cu imputarea mediilor in celule. De asemenea regasim in datele brute valori precum 2.37 pentru bai, care se intampla a fi exact media, motiv pentru care am tras aceasta concluzie*
# In etapa trecuta vedem si un max(price) de 2147483647, care este limita superioara al unui integer pe 32 biti. Este evident o eroare, acestea vor fi tratate in pasul urmator.

# %%
# Verificam valorile null
print("Valori null pe fiecare coloana:\n", df.isnull().sum())

# Eliminam coloanele inutile pentru analiza.
df = df.drop(columns=['BROKERTITLE', 'MAIN_ADDRESS', 'FORMATTED_ADDRESS'])

# Eliminarea duplicatelor (inregistrari identice care pot distorsiona statisticile)
print(f"Duplicate identificate: {df.duplicated().sum()}")
df.drop_duplicates(inplace=True)
print(f"Duplicate dupa curatare: {df.duplicated().sum()}")


# %% [markdown]
# ## 3. Analiza Distributiei si eliminarea de outliers
# Seturile de date contin adesea valori extreme, fie ele erori de tastare ori situatii exceptionale legate de subiectul setului de date.
# Vizualizam distributia pentru a vedea asimetria datelor.

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.histplot(df['BEDS'], bins=20, color='skyblue', kde=True, ax=axes[0])
axes[0].set_title('Distributie Dormitoare')

sns.histplot(df['BATH'], bins=20, color='salmon', kde=True, ax=axes[1])
axes[1].set_title('Distributie Bai')

sns.histplot(df['PROPERTYSQFT'], bins=20, color='green', kde=True, ax=axes[2])
axes[2].set_title('Distributie Suprafata (sqft)')

plt.tight_layout()
plt.show()

# Filtrare Outliers: Pastram datele reprezentative pentru piata generala
# Eliminam proprietatile care depasesc praguri extreme pentru a nu "pacali" modelul de regresie.
df_clean = df[
    (df['PRICE'] < 500_000_000) & 
    (df['PRICE'] > 20_000) &
    (df['BEDS'] < 13) &
    (df['BATH'] < 11) &
    (df['PROPERTYSQFT'] < 10001)
].copy()

print(f"Inregistrari ramase dupa curatare: {len(df_clean)}")
display(df_clean.describe())
# Observam ca atat media cat si deviatia standard au scazut substantial, astfel analiza noastra va putea fi mult mai precisa. Cifrele erau ridicate artificial de ceea ce pareau a fi erori de introducere a datelor.

# %% [markdown]
# ## 4. Analiza Corelatiei
# Verificam cat de puternic sunt legate variabilele intre ele prin corelatia Pearson. O corelatie aproape de 1.0 indica o legatura directa puternica.

# %%
numeric_cols = ['PRICE', 'BEDS', 'BATH', 'PROPERTYSQFT']
corr_matrix = df_clean[numeric_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matricea de Corelatie (Variabile Numerice)")
plt.show()

# %% [markdown]
# Interpretare: Se observa o corelatie puternica intre BEDS si BATH (aprox 0.7), ceea ce poate duce la multicoliniaritate.
# De asemenea, observam ca exista o corelatie mai stransa intre BATH si PRICE decat intre BEDS si PRICE. Explicam acest fenomen prin importanta gradului de confort al proprietatii: degeaba o locuinta are 5 dormitoare daca acestea nu sunt insotite de un numar corespunzator de bai.
# Corelatia dintre PROPERTYSQFT si Price este de asteptat sa fie pozitiva.

# %% [markdown]
# ## 5. Grupare Geografica (Clustering)
# Locatia este cel mai important factor in imobiliare. Deoarece nu avem cartierele normalizate, folosim **K-Means** pe coordonate (Lat/Long) pentru a crea 75 de "micro-zone".
# Astfel, scapam de povara de a lucra cu variabilele LOCALITY / SUBLOCALITY, in acelasi timp putem desena o harta a proprietatilor. Dupa aceasta clusterizare am putea folosi un serviciu
# de geocoding precum Google Places API, insa aceasta parte ar iesi din obiectivele proiecului. Vom nota totusi punctul de mijloc al fiecarui cluster pentru a putea identifica pe harta mai
# usor fiecare zona.

# %%
# folosim whiten pe coordonate pentru a le pregati de k-means clustering
cluster_features = ['LATITUDE', 'LONGITUDE']
data_whitened = whiten(df_clean[cluster_features].values)

# Rulam K-Means pentru a identifica zonele fierbinti
centroids, labels = kmeans2(data_whitened, k=75, minit='points', iter=20)
df_clean['Cluster'] = labels

plt.figure(figsize=(10, 7))
sns.scatterplot(data=df_clean, x='LONGITUDE', y='LATITUDE', hue='Cluster', palette='viridis', alpha=0.4, legend=False)
plt.title('Gruparea geografica automata (75 Clustere)')
plt.show()

cluster_summary = df_clean.groupby('Cluster').agg({
    'LATITUDE': 'mean',
    'LONGITUDE': 'mean',
    'PRICE': 'mean'
}).reset_index()
cluster_summary.columns = ['Cluster', 'Lat_Mijloc', 'Lng_Mijloc', 'Pret_Mediu_Cluster']

cluster_summary = cluster_summary.sort_values(by='Pret_Mediu_Cluster', ascending=False)

print("Rezumatul celor 75 de clustere (ordonate dupa pretul mediu):")
display(cluster_summary)

# %% [markdown]
# ## 6. Analiza Statistica Inferentiala (ANOVA)
# Folosim testul ANOVA pentru a confirma daca preturile difera semnificativ intre diverse localitati. 
# Daca P-value < 0.05, locatia are un impact cert asupra pretului.

# %%
print("--- Analiza ANOVA pentru Clustere ---")
localities_list = [df_clean[df_clean['Cluster'] == loc]['PRICE'] for loc in df_clean['Cluster'].unique() if len(df_clean[df_clean['Cluster'] == loc]) > 10]

f_stat, p_val = stats.f_oneway(*localities_list)
print(f"F-Statistic: {f_stat:.2f}, P-value: {p_val:.4e}")
print("Interpretare: Exista o diferenta semnificativa statistic intre preturile din diferite cartiere. Locatia conteaza!")
print("De obicei, pragul pentru a confirma ipoteza este de ca p < 0.05. In cazul de fata, p-ul este astronomic de mic.")

# %% [markdown]
# ## 7. Vizualizare Interactiva a Pietei
# Coloram zonele in functie de pretul mediu al fiecarui cluster. Galben/Verde deschis indica zone scumpe, mov indica zone accesibile.

# %%
cluster_prices = df_clean.groupby('Cluster')['PRICE'].mean().reset_index()
cluster_prices.columns = ['Cluster', 'Cluster_Avg_Price']
df_plot = df_clean.merge(cluster_prices, on='Cluster')

fig_clusters = px.scatter_map(
    df_plot, 
    lat='LATITUDE', lon='LONGITUDE', 
    color='Cluster_Avg_Price', 
    zoom=10, height=800,
    title='Harta Preturilor Medii pe Zone (Cluster)',
    color_continuous_scale=px.colors.sequential.Viridis,
    map_style="open-street-map"
)
fig_clusters.update_layout(
    margin={"r":0, "t":50, "l":0, "b":0} # Elimina spatiile goale de pe margini
)
fig_clusters.show()

# %% [markdown]
# ## 8. Modelarea Predictiva (Regresie OLS)
# Incercam sa prezicem pretul folosind caracteristicile casei si locatia (clusterele). 
# Transformam variabilele de tip text in variabile "Dummy" (0 sau 1).

# %%
# Transformare in variabile Dummy pentru Model
df_model = pd.get_dummies(df_clean, columns=['TYPE', 'Cluster'], drop_first=True)

X_cols = ['BEDS', 'BATH', 'PROPERTYSQFT'] + \
    [col for col in df_model.columns if 'TYPE_' in col] + \
    [col for col in df_model.columns if 'Cluster_' in col]

X = df_model[X_cols].astype(float)
y = df_model['PRICE'].astype(float)

X_with_const = sm.add_constant(X)
model = sm.OLS(y, X_with_const).fit()

print(model.summary())

# %% [markdown]
# ## 9. Optimizarea Modelului: Modelul Log-Liniar
# Preturile imobiliare cresc adesea exponential, nu liniar. 
# 1. **Log(Price)**: Preturile in NY sunt foarte asimetrice, iar logaritmarea pretului reduce efectul outlierilor care duc la anomalii precum coeficientul negativ al variabilei BEDS din prima versiune a regresiei.
# prin logaritmare, coeficientii vor fi relativi, nu absoluti. 
# 2. **BATH_PER_BED**: O variabila noua care descrie raportul de confort, eliminand multicoliniaritatea data de variabilele corelate BATH si BEDS. De asemenea, contribuie semnificativ la cresterea R^2 si la oferirea coeficientului variabilei BEDS o valoare corecta.

# %%
# Crearea variabilelor noi
df_model['LOG_PRICE'] = np.log1p(df_model['PRICE'])
df_model['BATH_PER_BED'] = df_model['BATH'] / df_model['BEDS']
df_model['BATH_PER_BED'] = df_model['BATH_PER_BED'].replace([np.inf, -np.inf], 0).fillna(0)

# Selectam noile variabile, eliminand BATH pentru a reduce multicoliniaritatea
X_new_cols = ['BEDS', 'BATH_PER_BED', 'PROPERTYSQFT'] + \
    [col for col in df_model.columns if 'TYPE_' in col] + \
    [col for col in df_model.columns if 'Cluster_' in col]

X_new = df_model[X_new_cols].astype(float)
y_log = df_model['LOG_PRICE']

X_new_const = sm.add_constant(X_new)
model_optimized = sm.OLS(y_log, X_new_const).fit()

print(model_optimized.summary())

# %% [markdown]
# ## 10. Interpretarea Performantei Modelului de Regresie (LOG_PRICE)
# Analizam calitatea modelului prin prisma indicatorilor rezultati din tabelul OLS Regression Results, concentrandu-ne pe puterea de predictie si validitatea erorilor.
#
# ### 10.1. Puterea de Predictie si Semnificatia Globala
# * **R-squared (0.726):** Modelul reuseste sa explice aproximativ 72% din variatia preturilor logaritmate. Intr-o piata volatila precum cea din New York, un scor de peste 0.7 indica o capacitate de predictie ridicata si un model bine calibrat.
# * **Adj. R-squared (0.720):** Valoarea foarte apropiata de R^2 demonstreaza ca variabilele introduse (cele 75 de clustere si tipurile de proprietati) sunt relevante si nu "umfla" artificial scorul modelului.
# * **F-statistic (131.5) & Prob(F) = 0.00:** Probabilitatea de 0.00 confirma semnificatia globala a modelului. Putem afirma cu certitudine ca variabilele independente alese au un impact real asupra pretului de vanzare.
#
# ### 10.2. Diagnosticul Erorilor (Testele de Validitate)
# * **Durbin-Watson (1.977):** Valoarea este aproape de pragul ideal de 2.0, ceea ce indica faptul ca nu exista autocorelatie in randul erorilor. Reziduurile sunt independente, o conditie esentiala pentru validitatea regresiei OLS.
# 
# * **Skew (0.046) & Kurtosis (6.651)):** Asimetria (Skew) este aproape de zero, ceea ce arata ca logaritmarea pretului a functionat corect in normalizarea distributiei. 
# * Desi valoarea Kurtosis a scazut la 6.43, aceasta ramane inca peste pragul ideal de 3. 
#   * Aceasta scadere indica faptul ca modelul a devenit mult mai robust, iar erorile extreme s-au redus considerabil. 
#   * Totusi, valoarea ramasa sugereaza ca in New York inca exista proprietati cu caracteristici unice (outliers reziduali) care produc erori de predictie mai mari decat media, dar intr-o masura mult mai mica decat inainte.
# 
# * **Jarque-Bera (2510.015) & Omnibus (388.7353), prob = 0:** Probabilitatea de 0.00 indica faptul ca reziduurile nu urmeaza o distributie perfect normala. in econometrie, pe esantioane de mii de observatii, acest rezultat este frecvent si nu anuleaza utilitatea practica a modelului, ci doar evidentiaza complexitatea datelor din realitate.
# 
#
# ### 10.3. Multicoliniaritatea si Matricea de Date
# * **Cond. No. (1.89e+05):** Observam un numar de conditionare mare, fapt explicabil prin utilizarea a 75 de variabile dummy pentru clustere. Aceasta complexitate numerica este imposibil de evitat in modelele precum acesta, care folosesc o segmentare cu atat de multe variabile dummy, dar este compensata de stabilitatea coeficientilor obtinuti prin eliminarea corelatiei directe dintre BEDS si BATH.
#
# **Concluzie:** Modelul optimizat cu pret logaritmat este mult mai robust decat varianta initiala, reusind sa ofere o imagine fidela a modului in care suprafata, confortul si locatia interactioneaza pentru a stabili valoarea imobiliara in New York.

# %% [markdown]
# ## Analiza celor mai importante variabile independente
# Analizam rezultatele regresiei pentru a identifica factorii care au cel mai mare impact asupra pretului unei proprietati in New York. Deoarece variabila dependenta este **LOG_PRICE**, interpretam coeficientii ca modificari procentuale.
#
# ### Variabilele de impact
# Acestea sunt variabilele cu cel mai mic P-value (0.000), fiind predictori extrem de siguri:
# * **BATH_PER_BED (Coef: 0.5272):** Este variabila cu cel mai mare impact unitar. O crestere cu o unitate a raportului bai per dormitor (de exemplu, trecerea de la o baie la doua bai pentru acelasi numar de paturi) genereaza o crestere a pretului de aproximativ **52.7%**. Acest lucru confirma ca luxul si confortul primeaza in fata cantitatii.
# * **BEDS (Coef: 0.1639):** Fiecare dormitor adaugat creste pretul proprietatii cu aproximativ **16.4%**, mentinand restul factorilor constanti.
# * **PROPERTYSQFT (Coef: 0.0002):** Desi coeficientul pare mic, el se aplica la fiecare square foot. O crestere de 1000 sqft in suprafata aduce o crestere de aproximativ **20%** a pretului total.
#
# ### Tipul Proprietatii (TYPE)
# Modelul foloseste o categorie de referinta (probabil 'Condo for sale' sau similar). Comparativ cu referinta:
# * **Townhouse for sale (Coef: 0.8067):** Este cel mai scump tip de proprietate, avand un pret cu **80.6%** mai mare decat categoria de baza.
# * **House for sale (Coef: 0.7805):** Casele individuale sunt cu **78%** mai scumpe decat apartamentele standard.
# * **Multi-family home (Coef: 0.6108):** Acestea aduc un plus de **61%** la pret, fiind probabil privite ca investitii generatoare de venit.
# * **Observatie:** Tipuri precum 'Coming Soon' (P=0.458) sau 'Condop' (P=0.603) au P-value mare, deci **nu sunt predictori importanti** pentru pret in acest set de date.
#
# ### 12.3. Locatia (Clusterele Geografice)
# * Toate cele 75 de clustere au **P-value = 0.000**, ceea ce demonstreaza ca locatia este un factor critic.
# * Diferenta uriasa dintre coeficientii clusterelor (de la -0.21 la -2.19) arata ca **locatia poate modifica pretul unei case cu pana la 200%**, confirmand maxima imobiliara "Location, Location, Location".
#
# ### Concluzie privind importanta:
# Cei mai importanti predictori sunt, in ordine: **Locatia (Clusterele)**, **Tipul proprietatii (Townhouse/House)** si **Gradul de confort (BATH_PER_BED)**. Suprafata (SQFT) conteaza, dar impactul ei este secundar fata de zona in care se afla imobilul.

# %% [markdown]
# ## 11. Identificarea Proprietatilor Subevaluate (Oportunitati de Investitie)
# Pentru a gasi proprietati subevaluate, comparam pretul real din setul de date cu pretul estimat de modelul nostru optimizat. 
# * **Reziduu Negativ mare**: Inseamna ca pretul real este mult sub cel pe care modelul il considera "corect" pentru acele dotari si acea zona.
# * **Nota**: O proprietate poate aparea ca fiind subevaluata si daca are defecte ascunse (ex: structura subreda) pe care modelul nu le vede din datele actuale.

# %%
# 1. Obtinem predictiile modelului in format logaritmic si le transformam inapoi in dolari
# Folosim expm1 pentru a inversa operatia log1p folosita la antrenare
df_clean['PREDICTED_PRICE'] = np.expm1(model_optimized.predict(X_new_const))

# 2. Calculam diferenta absoluta si procentuala
# Un numar negativ inseamna ca pretul real e mai mic decat cel prezis de model
df_clean['PRICE_DIFF'] = df_clean['PRICE'] - df_clean['PREDICTED_PRICE']
df_clean['UNDERVALUATION_PCT'] = (df_clean['PRICE_DIFF'] / df_clean['PREDICTED_PRICE']) * 100

# 3. Filtram proprietatile
# Cautam proprietati unde pretul real este cu cel putin 40% mai mic decat cel estimat de model
deals = df_clean[df_clean['UNDERVALUATION_PCT'] < -40].sort_values(by='UNDERVALUATION_PCT')

print(f"Am gasit {len(deals)} proprietati potential subevaluate.")
display(deals[['TYPE', 'PRICE', 'PREDICTED_PRICE', 'UNDERVALUATION_PCT', 'BEDS', 'BATH']].head(10))

# 4. Vizualizam unde se afla aceste oportunitati pe harta
fig_deals = px.scatter_map(
    deals, 
    lat='LATITUDE', lon='LONGITUDE', 
    color='UNDERVALUATION_PCT',
    size='PROPERTYSQFT',
    hover_data=['PRICE', 'PREDICTED_PRICE', 'TYPE'],
    title='Top Oportunitati: Proprietati Subevaluate (Pret Real vs Pret Model)',
    color_continuous_scale='RdYlGn_r', # Verde = Subevaluat
    zoom=10, height=800,
    map_style="open-street-map"
)
fig_deals.update_layout(margin={"r":0, "t":50, "l":0, "b":0})
fig_deals.show()
deals.to_csv("proprietati_subevaluate.csv", index=False)

# %% [markdown]
# ### Interpretare
# Daca modelul prezice un pret de 1.000.000$ (PREDICTED_PRICE) si proprietatea este listata la 600.000$ (PRICE), exista un profit potential de 400.000$ care poate fi exploatat prin revanzare la pretul pietei.
# Punctele de un verde intens reprezinta zonele unde pretul cerut este mult sub valoarea statistica a zonei si a dotarilor.

# %% [markdown]
# ## 12. Intrebari Specifice: Analiza Extremelor Geografice
# **Intrebare: Care sunt "Polii" pietei imobiliare din New York in functie de clusterele identificate?**
# 
# ### Raspuns:
# Identificam clusterele cu media de pret cea mai ridicata si cea mai scazuta pentru a intelege ierarhia zonelor.

# %%
# Calculam media pe clustere si sortam
top_clusters = df_clean.groupby('Cluster')['PRICE'].mean().sort_values(ascending=False)

print("Top 5 Cele mai scumpe Clustere (Zone de Lux):")
display(top_clusters.head(5))

print("\nTop 5 Cele mai ieftine Clustere (Zone Accesibile):")
display(top_clusters.tail(5))

# %% [markdown]
# **Interpretare:** Diferenta dintre cel mai scump si cel mai ieftin cluster poate fi de peste 14 ori. Aceasta confirma ca in New York, locatia are un impact mai puternic decat suprafata proprietatii.

# %% [markdown]
# ## 13. Intrebari Specifice: Eficienta Costului pe Suprafata
# **Intrebare: Care tip de proprietate ofera cel mai bun raport Pret / Suprafata (Sqft)?**
# 
# ### Raspuns:
# Calculam o variabila noua `PRICE_PER_SQFT` pentru a vedea cat plateste un cumparator pentru fiecare unitate de suprafata in functie de tipul casei.

# %%
df_typeclean = df_clean[df_clean["TYPE"] != "For sale"].copy() #eliminam datele cu type = "for sale", e irelevant in situatia noastra
df_typeclean['PRICE_PER_SQFT'] = df_typeclean['PRICE'] / df_typeclean['PROPERTYSQFT']
sqft_analysis = df_typeclean.groupby('TYPE')['PRICE_PER_SQFT'].mean().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
sqft_analysis.plot(kind='bar', color='teal')
plt.title('Pretul mediu per Square Foot in functie de Tipul Proprietatii')
plt.ylabel('USD / Sqft')
plt.show()

# %% [markdown]
# **Interpretare:** Observam ca tipurile de lux (ex: Townhouses) au un pret pe sqft mult mai mare. Daca un investitor cauta "volum", se va orienta catre tipurile de proprietati cu un pret per sqft mai mic, unde primeste mai mult spatiu pentru aceeasi suma de bani.

# %% [markdown]
# ## 14. Intrebari Specifice: Analiza pretului pe unitate de suprafata (Sqft) pe Clustere
# **Intrebare: Care sunt zonele cele mai scumpe raportat la suprafata oferita?**
#
# %%
# 1. Calculam pretul per square foot pentru fiecare proprietate in df_clean
df_clean['PRICE_PER_SQFT'] = df_clean['PRICE'] / df_clean['PROPERTYSQFT']

# 2. Gruparea dupa Cluster si calcularea mediei pretului pe sqft
cluster_value = df_clean.groupby('Cluster').agg({
    'PRICE_PER_SQFT': 'mean',
    'LATITUDE': 'mean',
    'LONGITUDE': 'mean'
}).reset_index()

cluster_value.columns = ['Cluster', 'Avg_Price_Per_Sqft', 'Center_Lat', 'Center_Lng']
cluster_value = cluster_value.sort_values(by='Avg_Price_Per_Sqft', ascending=False)
cluster_value.to_csv("pret_sqft_clustere.csv", index=False)

# 3. Vizualizam top 15 cele mai scumpe clustere (unde spatiul este cel mai valoros)
top_expensive_sqft = cluster_value.sort_values(by='Avg_Price_Per_Sqft', ascending=False).head(15)


plt.figure(figsize=(12, 6))
sns.barplot(
    data=top_expensive_sqft, 
    x='Cluster', 
    y='Avg_Price_Per_Sqft', 
    palette='magma', 
    order=top_expensive_sqft['Cluster'],
    hue='Avg_Price_Per_Sqft',
)
plt.title('Top 15 Clustere: Cele mai scumpe zone pe unitate de suprafata (USD/Sqft)')
plt.xlabel('Cluster')
plt.ylabel('Pret Mediu USD / Sqft')
plt.show()

# 4. Harta interactiva pentru distributia valorii pe Sqft
df_plot_val = df_clean.merge(cluster_value, on='Cluster')

fig_val = px.scatter_map(
    df_plot_val, 
    lat='LATITUDE', lon='LONGITUDE', 
    color='Avg_Price_Per_Sqft', 
    size='PRICE_PER_SQFT',
    hover_data=['Cluster', 'Avg_Price_Per_Sqft', 'PRICE'],
    title='Harta Valorii Imobiliare: Pret per Square Foot pe Clustere',
    color_continuous_scale='Inferno',
    zoom=10, height=800,
    map_style="open-street-map"
)
fig_val.update_layout(margin={"r":0, "t":50, "l":0, "b":0})
fig_val.show()

# %% [markdown]
# ### Interpretare:
# * **Diferente intre medii:** Observam discrepante uriase intre clustere. In timp ce in zonele de periferie pretul per sqft este scazut, in zonele de lux precum Manhattan Island, in jurul Central Park cumparatorii platesc sume uriase pentru locatie.
# * **Decizie de investitie:** Clusterele cu un pret total mare, dar un pret pe sqft moderat, pot reprezenta investitii mai sigure decat zonele unde platesti foarte mult pentru cativa metri patrati.
# * **Confirmarea Regresiei:** Aceasta analiza explica de ce SQFT are un coeficient stabil in regresie, dar Cluster are un impact mult mai mare asupra pretului final: suprafata conteaza, dar unde se afla acea suprafata este multiplicatorul real al valorii.

