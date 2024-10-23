import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import kagglehub
import plotly.graph_objects as go
import numpy as np

path = kagglehub.dataset_download("gregorut/videogamesales")
print(f'{path}\\vgsales.csv')

df = pd.read_csv(f'{path}\\vgsales.csv')
# df.head()

# TOTAL DE CAMPOS VAZIOS POR COLUNA
plt.figure(figsize=(10, 6))
sns.barplot(y=df.isnull().sum().index, x=df.isnull().sum().values, orient='h')

plt.xlabel('Quantidade')
plt.ylabel('Dados não preenchidos por coluna')
plt.title('Campo')

plt.show()

df['Year'] = df['Year'].fillna(df['Year'].median())
df['Publisher'] = df['Publisher'].fillna(df['Publisher'].mode()[0])

# TOTAL DE VENDAS POR PLATAFORMA
sales_per_platform = df.groupby('Platform')['Global_Sales'].sum() / 1000

sales_per_platform = sales_per_platform.reset_index()
order = sales_per_platform.sort_values(by='Global_Sales', ascending=False)['Platform']

plt.figure(figsize=(15, 6))
sns.barplot(x='Platform', y='Global_Sales', data=sales_per_platform, order=order, color='#dc6fed')

plt.title('Total de Vendas por Plataforma')
plt.ylabel('Total de Vendas (em bilhão)')

plt.grid(True, linestyle='--', alpha=0.7, color='gray')

plt.xticks(rotation=90)
plt.show()

# TOTAL DE VENDAS POR REGIÃO
regional_sales = df[['NA_Sales', 'EU_Sales', 'JP_Sales']].sum() / 1000

regional_sales.plot(kind='barh', color=['#9af252', '#edf252', '#e6844c'])

plt.title('Vendas Totais de Jogos por Região')

plt.xlabel('Total de Vendas (bilhões)')
plt.ylabel('Região', rotation=360)
plt.yticks(rotation=45)

plt.grid(True, linestyle='--', alpha=0.7, color='gray')

plt.grid(True)
plt.show()

# VENDAS DE JOGOS POR GÊNERO (NA)
plt.style.use('seaborn-v0_8-darkgrid')

sales_by_genre = df.groupby('Genre')['NA_Sales'].sum().reset_index()
sales_by_genre = sales_by_genre.sort_values(by='NA_Sales', ascending=False)

colors = sns.color_palette("coolwarm", len(sales_by_genre))

plt.figure(figsize=(12, 6))
barplot = sns.barplot(x='Genre', y='NA_Sales', data=sales_by_genre, hue='Genre', palette=colors)

plt.title('Vendas de Jogos por Gênero (NA)', fontsize=16, fontweight='bold')
plt.xlabel('Gênero', fontsize=12)
plt.ylabel('Vendas Totais na NA (Milhões)', fontsize=12)

plt.xticks(rotation=45, ha='right')

for index, value in enumerate(sales_by_genre['NA_Sales']):
    barplot.text(index, value + 0.1, f'{value:.2f}', color='black', ha="center", fontsize=10)

plt.tight_layout()

plt.show()

# VENDAS DE JOGOS POR GÊNERO (EU)
plt.style.use('seaborn-v0_8-darkgrid')

sales_by_genre = df.groupby('Genre')['EU_Sales'].sum().reset_index()
sales_by_genre = sales_by_genre.sort_values(by='EU_Sales', ascending=False)

colors = sns.color_palette("coolwarm", len(sales_by_genre))

plt.figure(figsize=(12, 6))
barplot = sns.barplot(x='Genre', y='EU_Sales', data=sales_by_genre, hue='Genre', palette=colors)

plt.title('Vendas de Jogos por Gênero (EU)', fontsize=16, fontweight='bold')
plt.xlabel('Gênero', fontsize=12)
plt.ylabel('Vendas Totais na EU (Milhões)', fontsize=12)

plt.xticks(rotation=45, ha='right')

for index, value in enumerate(sales_by_genre['EU_Sales']):
    barplot.text(index, value + 0.05, f'{value:.2f}', color='black', ha="center", fontsize=10)

plt.tight_layout()

plt.show()

# VENDAS DE JOGOS POR GÊNERO (JP)
plt.style.use('seaborn-v0_8-darkgrid')

sales_by_genre = df.groupby('Genre')['JP_Sales'].sum().reset_index()
sales_by_genre = sales_by_genre.sort_values(by='JP_Sales', ascending=False)

colors = sns.color_palette("coolwarm", len(sales_by_genre))

plt.figure(figsize=(12, 6))
barplot = sns.barplot(x='Genre', y='JP_Sales', data=sales_by_genre, hue='Genre', palette=colors)

plt.title('Vendas de Jogos por Gênero (JP)', fontsize=16, fontweight='bold')
plt.xlabel('Gênero', fontsize=12)
plt.ylabel('Vendas Totais no JP (Milhões)', fontsize=12)

plt.xticks(rotation=45, ha='right')

for index, value in enumerate(sales_by_genre['JP_Sales']):
    barplot.text(index, value + 0.05, f'{value:.2f}', color='black', ha="center", fontsize=10)

plt.tight_layout()

plt.show()

# VENDAS GLOBAIS AO LONGO DO TEMPO
sales_over_time = df.groupby('Year')['Global_Sales'].sum().reset_index()

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=sales_over_time['Year'],
    y=sales_over_time['Global_Sales'],
    mode='lines+markers',
    marker=dict(size=6),
    line=dict(width=2, color='blue')
))

fig.update_layout(
    title='Vendas Globais ao Longo do Tempo',
    xaxis_title='Ano',
    yaxis_title='Vendas Globais (em milhões)',
    hovermode='x unified',
    font=dict(size=12),
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=True, gridcolor='lightgrey')
)

fig.show()

# HISTOGRAMA DE VENDAS POR PLATAFORMA

df_grouped = df.groupby('Platform')['Global_Sales'].sum().reset_index()
df_grouped = df_grouped.sort_values(by='Global_Sales', ascending=False)
# df_grouped = df_grouped.sort_values(by='Platform', ascending=True)

plt.figure(figsize=(24, 10))
ax = sns.histplot(data=df_grouped, x='Platform', weights='Global_Sales', color='#ff0008', bins=10)
plt.title('Histograma de Vendas por Plataforma')
plt.xlabel('Plataforma')
plt.ylabel('Vendas (bilhão)')
plt.grid(True, linestyle='--', alpha=0.7, color='gray')

yticks = ax.get_yticks()
ax.set_yticks(yticks)

ax.set_yticklabels([f'{tick / 1000:.1f}' for tick in yticks])
plt.xticks(ax.get_xticks(), rotation=90, fontsize=12)

sns.set_theme(style="whitegrid")

plt.show()

# HISTOGRAMA DE VENDAS POR ANO
df_grouped = df.groupby('Year')['Global_Sales'].sum().reset_index()

plt.figure(figsize=(24, 10))

ax = sns.histplot(data=df_grouped, x='Year', weights='Global_Sales', color='#65c7a6', bins=10)
plt.title('Histograma de Vendas por Ano de Lançamento')
plt.xlabel('Ano de Lançamento')
plt.ylabel('Vendas (bilhão)')
ax.grid(True)

yticks = ax.get_yticks()
ax.set_yticks(yticks)

ax.set_yticklabels([f'{tick / 1000:.1f}' for tick in yticks])

plt.plot()