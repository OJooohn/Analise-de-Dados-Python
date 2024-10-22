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
df['Year'] = df['Year'].fillna(df['Year'].median())
df.isnull().sum()

df['Publisher'] = df['Publisher'].fillna(df['Publisher'].mode()[0])
df.isnull().sum()

# Total de Vendas por Plataforma
sales_per_platform = df.groupby('Platform')['Global_Sales'].sum() / 1000

sales_per_platform = sales_per_platform.reset_index()
order = sales_per_platform.sort_values(by='Global_Sales', ascending=False)['Platform']

plt.figure(figsize=(15, 6))
sns.barplot(x='Platform', y='Global_Sales', data=sales_per_platform, order=order)

plt.title('Total de Vendas por Plataforma')
plt.ylabel('Total de Vendas (em bilhão)')

plt.xticks(rotation=90)
plt.show()

# Vendas Totais por regiao
regional_sales = df[['NA_Sales', 'EU_Sales', 'JP_Sales']].sum() / 1000

regional_sales.plot(kind='bar')

plt.title('Vendas Totais de Jogos por Região')

plt.xlabel('Região')
plt.ylabel('Total de Vendas (bilhões)')
plt.xticks(rotation=360)

plt.grid(True)
plt.show()

# Total de vendas conforme os anos

sales_over_time = df.groupby('Year')['Global_Sales'].sum().reset_index()

# Criando a figura
fig = go.Figure()

# Adicionando o gráfico de linhas com marcadores
fig.add_trace(go.Scatter(
    x=sales_over_time['Year'],
    y=sales_over_time['Global_Sales'],
    mode='lines+markers',
    marker=dict(size=6),
    line=dict(width=2, color='blue')
))

# Atualizando o layout para um design simples
fig.update_layout(
    title='Vendas Globais ao Longo do Tempo',
    xaxis_title='Ano',
    yaxis_title='Vendas Globais (em milhões)',
    hovermode='x unified',
    font=dict(size=12),
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=True, gridcolor='lightgrey')
)

# Exibindo o gráfico
fig.show()

# Histograma de vendas por plataforma

df_grouped = df.groupby('Platform')['Global_Sales'].sum().reset_index()
df_grouped = df_grouped.sort_values(by='Global_Sales', ascending=False)

plt.figure(figsize=(24, 10))
ax = sns.histplot(data=df_grouped, x='Platform', weights='Global_Sales', bins=10)
plt.title('Histograma de Vendas por Plataforma')
plt.xlabel('Plataforma')
plt.ylabel('Vendas (milhão)')
plt.grid(True)

yticks = ax.get_yticks()
ax.set_yticks(yticks)
ax.set_yticklabels([f'{tick / 1000:.1f}' for tick in yticks])

plt.xticks(ax.get_xticks(), rotation=90)

plt.show()

# Histograma de vendas por ano

df_grouped = df.groupby('Year')['Global_Sales'].sum().reset_index()

plt.figure(figsize=(24, 10))

ax2 = sns.histplot(data=df_grouped, x='Year', weights='Global_Sales', bins=10)
plt.title('Histograma de Vendas por Ano de Lançamento')
plt.xlabel('Ano de Lançamento')
plt.ylabel('Vendas (milhão)')
ax2.grid(True)

yticks = ax2.get_yticks()
ax2.set_yticks(yticks)
ax2.set_yticklabels([f'{tick / 1000:.1f}' for tick in yticks])

plt.plot()
plt.show()
