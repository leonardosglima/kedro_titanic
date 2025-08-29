import pandas as pd

df_check = pd.read_csv('data/01_raw/train.csv')

print(df_check)
print('')

porcentagem_sobreviventes = (len(df_check[df_check['Survived'] == 1])/len(df_check)) * 100
porcentagem_nao_sobreviventes = (len(df_check[df_check['Survived'] == 0])/len(df_check)) * 100

print(f'Sobreviventes: {round(porcentagem_sobreviventes, 2)}%')
print(f'Não Sobreviventes: {round(porcentagem_nao_sobreviventes, 2)}%')

print('')