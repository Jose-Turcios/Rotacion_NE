import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import datetime
import pymongo
import hmac



##############################################################################################################################


st.set_page_config(layout="wide")
 
#################################################################
def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        # Verifica si la contraseña existe antes de comparar
        if "password" not in st.session_state:
            st.session_state["password_correct"] = False
            return

        # Compara la contraseña ingresada con el secreto
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # No almacenar la contraseña.
        else:
            st.session_state["password_correct"] = False

    # Si ya está autenticado, retorna True
    if st.session_state.get("password_correct", False):
        return True

    # Mostrar campo de contraseña
    st.text_input(
        "Password", 
        type="password", 
        on_change=password_entered, 
        key="password"
    )

    # Mostrar error si la contraseña es incorrecta
    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("😕 Password incorrect")

    return False

if not check_password():
    st.stop()  # Detener la ejecución si la contraseña no es correcta
 
###################################################################
 
# Function to cache data
@st.cache_resource
def get_data():
    # Load MongoDB URI from Streamlit secrets
    # Ensure you have defined 'mongouri' in your secrets.toml
    mongo_uri = st.secrets["mongouri"]
 
    # Connect to your MongoDB cluster
    client = pymongo.MongoClient(mongo_uri)
 
    # Select your database
    db = client.Rotacion_NE
 
 
    # Select your collection
    collection = db.Ventas
   
    # Fetch data from the collection
    # Converting cursor to list then to DataFrame
    data = list(collection.find())
    df = pd.DataFrame(data)
 
   
 
 
    return df
 
# Get data
df = get_data()
#drop index column
df = df.drop(columns=['_id'])
 
# Function to cache data 2
@st.cache_resource
def get_data2():
    # Load MongoDB URI from Streamlit secrets
    # Ensure you have defined 'mongouri' in your secrets.toml
    mongo_uri = st.secrets["mongouri"]
 
    # Connect to your MongoDB cluster
    client = pymongo.MongoClient(mongo_uri)
 
    # Select your database
    db = client.Rotacion_NE
 
 
    # Select your collection
    collection2  = db.STOCK
 
    # Fetch data from the collection
    # Converting cursor to list then to DataFrame
    data = list(collection2.find())
    df_stock = pd.DataFrame(data)
 
    # Fetch data from the collection2
    # Converting cursor to list then to DataFrame
    data2 = list(collection2.find())
    df_stock = pd.DataFrame(data2)
 
 
    return df_stock
 
#get data for df2
df_stock = get_data2()
#drop index column
df_stock = df_stock.drop(columns=['_id'])


###########################################################################################################################

# Convert Codigo_SAP to string
df['Codigo_SAP'] = df['Codigo_SAP'].astype(str)

# Remove trailing spaces.0 
df['Codigo_SAP'] = df['Codigo_SAP'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()

######################################################################################################################################

# Create a column named Fecha and for this column use Dia, Mes and Anio
# Supongamos que tus columnas se llaman Anio, Mes, Dia
df = df.rename(columns={'Anio': 'year', 'Mes': 'month', 'Dia': 'day'})

# Crear la columna de fecha
df['Fecha'] = pd.to_datetime(df[['year', 'month', 'day']])

# Formatearla como 'dd-mm-yyyy' sin hora
df['Fecha'] = df['Fecha'].dt.strftime('%d-%m-%Y')

##################################################################################################################################

# Drop the columns Anio, Mes and Dia
df = df.drop(columns=['year', 'month', 'day'])

###################################################################################################################################

# NOW make a grouby by Pais,Bodega,Codigo_SAP, U_Estilo,U_Silueta,U_Coleccion_NE,Talla, U_Liga, U_Team, Fecha
df_grouped = df.groupby(['Pais','Codigo_SAP','Bodega', 'U_Estilo', 'U_Silueta', 'U_Coleccion_NE', 'Talla', 'U_Liga', 'U_Team','U_Segmento']).agg({'Cantidad': 'sum'}).reset_index()



##################################################################################################################################

#Using markdown h1 title movimientos NEW ERA
st.markdown("<h1 style='text-align: center; color: black;'>🧢 MOVIMIENTOS NEW ERA</h1>", unsafe_allow_html=True)


# Create multisectbox for Pais, Bodega, U_Estilo, U_Silueta, U_Coleccion_NE, Talla, U_Liga, U_Team

col1, col2, col3,col4,col5,col6 = st.columns(6)

with col1:
    pais = st.multiselect('Pais', df_grouped['Pais'].unique())
    bodega = st.multiselect('Bodega', df_grouped['Bodega'].unique())
with col2:
    estilo = st.multiselect('U_Estilo', df_grouped['U_Estilo'].unique())
    silueta = st.multiselect('U_Silueta', df_grouped['U_Silueta'].unique())
with col3:
    coleccion = st.multiselect('U_Coleccion_NE', df_grouped['U_Coleccion_NE'].unique())
    talla = st.multiselect('Talla', df_grouped['Talla'].unique())
with col4:
    liga = st.multiselect('U_Liga', df_grouped['U_Liga'].unique())
    team = st.multiselect('U_Team', df_grouped['U_Team'].unique())
with col5:
    # Create a multiselect for Codigo_SAP
    codigo_sap = st.multiselect('Codigo_SAP', df_grouped['Codigo_SAP'].unique())
    # Create a multiselect for U_Segmento
    u_segmento = st.multiselect('U_Segmento', df_grouped['U_Segmento'].unique())


with col6:
    # Convierte 'Fecha' a datetime si no lo es
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    # Set the minimum and maximum dates for the date range filter
    min_date = df['Fecha'].min()
    max_date = df['Fecha'].max()
    # Ask the user to choose a start and end date
    start_date = st.date_input('Start date', min_value=min_date, max_value=max_date, value=min_date)
    end_date = st.date_input('End date', min_value=min_date, max_value=max_date, value=max_date)

# Apply filters
if pais:
    df_grouped = df_grouped[df_grouped['Pais'].isin(pais)]
    df= df[df['Pais'].isin(pais)]
    df_stock = df_stock[df_stock['Pais'].isin(pais)]
if bodega:
    df_grouped = df_grouped[df_grouped['Bodega'].isin(bodega)]
    df= df[df['Bodega'].isin(bodega)]
    df_stock = df_stock[df_stock['Bodega'].isin(bodega)]
if estilo:
    df_grouped = df_grouped[df_grouped['U_Estilo'].isin(estilo)]
    df= df[df['U_Estilo'].isin(estilo)]
    df_stock = df_stock[df_stock['U_Estilo'].isin(estilo)]
if silueta:
    df_grouped = df_grouped[df_grouped['U_Silueta'].isin(silueta)]
    df= df[df['U_Silueta'].isin(silueta)]
    df_stock = df_stock[df_stock['U_Silueta'].isin(silueta)]
if coleccion:
    df_grouped = df_grouped[df_grouped['U_Coleccion_NE'].isin(coleccion)]
    df= df[df['U_Coleccion_NE'].isin(coleccion)]
    df_stock = df_stock[df_stock['U_Coleccion_NE'].isin(coleccion)]
if talla:
    df_grouped = df_grouped[df_grouped['Talla'].isin(talla)]
    df= df[df['Talla'].isin(talla)]
    df_stock = df_stock[df_stock['Talla'].isin(talla)]
if liga:
    df_grouped = df_grouped[df_grouped['U_Liga'].isin(liga)]
    df= df[df['U_Liga'].isin(liga)]
    df_stock = df_stock[df_stock['U_Liga'].isin(liga)]
if team:
    df_grouped = df_grouped[df_grouped['U_Team'].isin(team)]
    df= df[df['U_Team'].isin(team)]
    df_stock = df_stock[df_stock['U_Team'].isin(team)]
if codigo_sap:
    df_grouped = df_grouped[df_grouped['Codigo_SAP'].isin(codigo_sap)]
    df= df[df['Codigo_SAP'].isin(codigo_sap)]
    df_stock = df_stock[df_stock['Codigo_SAP'].isin(codigo_sap)]
if u_segmento:
    df_grouped = df_grouped[df_grouped['U_Segmento'].isin(u_segmento)]
    df= df[df['U_Segmento'].isin(u_segmento)]
    df_stock = df_stock[df_stock['U_Segmento'].isin(u_segmento)]



# Filter the DataFrame based on the selected date range
df = df[(df['Fecha'] >= pd.to_datetime(start_date)) & (df['Fecha'] <= pd.to_datetime(end_date))]


############################################################################################################


df_stock['Codigo_SAP'] = df_stock['Codigo_SAP'].astype(str)

# Make a groupby by Pais,Bodega,Codigo_SAP, U_Estilo,U_Silueta,U_Coleccion_NE,Talla, U_Liga, U_Team
df_stock_grouped = df_stock.groupby(['Pais','Codigo_SAP','Bodega', 'U_Estilo', 'U_Silueta', 'U_Coleccion_NE', 'Talla', 'U_Liga', 'U_Team','U_Segmento']).agg({'Stock_Actual': 'sum'}).reset_index()

# Drop the columns U_Estilo, U_Silueta, U_Coleccion_NE, Talla, U_Liga, U_Team from df_stock_grouped
df_stock_grouped = df_stock_grouped.drop(columns=['U_Estilo', 'U_Silueta', 'U_Coleccion_NE', 'Talla', 'U_Liga', 'U_Team','U_Segmento'])

############################################################################################################
# Make a copy of df named df_fechas
df_fechas = df.copy()
# Make a groupyby by Pais,Bodega,Codigo_SAP and Fecha
df_fechas = df_fechas.groupby(['Pais','Codigo_SAP','Bodega', 'U_Estilo', 'Fecha']).agg({'Cantidad': 'sum'}).reset_index()

# Ahora en este df encontraras la ultima fecha que se vendio El Codigo_SAP por Pais y Bodega 
# y dejaras unicamente la ultima fecha y le restaras la fecha de hoy 
# menos esa fehca para encontar hace cuantos dias se Vendio el producto
# Asegurar formato de fecha
df_fechas['Fecha'] = pd.to_datetime(df_fechas['Fecha'], dayfirst=True, errors='coerce')

# Filtrar solo cantidades positivas (ventas reales)
df_fechas = df_fechas[df_fechas['Cantidad'] > 0]

# Ordenar por fecha y agrupar por Codigo_SAP y Bodega para quedarte con la última venta real
df_ult_fecha = (
    df_fechas
    .sort_values('Fecha')
    .groupby(['Codigo_SAP', 'Bodega'], as_index=False)
    .last()
)

# Calcular días desde la última venta
hoy = pd.Timestamp.today().normalize()
df_ult_fecha['dias_sin_venta'] = (hoy - df_ult_fecha['Fecha']).dt.days

#############################################################################################################################

# Make a groupy by Pais, Bodega, Codigo_SAP, dias_sin_venta and agg cantidad
df_ult_fecha = df_ult_fecha.groupby(['Pais', 'Bodega', 'Codigo_SAP', 'dias_sin_venta']).agg({'Cantidad': 'sum'}).reset_index()
# Drop Cantidad 

df_ult_fecha = df_ult_fecha.drop(columns=['Cantidad'])
#st.dataframe(df_ult_fecha)
#############################################################################################################################
# Merge on Bodega and Codigo_SAP and Pais to df_grouped and df_ult_fecha
df_grouped = df_grouped.merge(df_ult_fecha, on=['Bodega', 'Codigo_SAP', 'Pais'], how='left')
#############################################################################################################################

# Drop the columns U_Estilo, U_Silueta, U_Coleccion_NE, Talla, U_Liga, U_Team from df_grouped
df_grouped = df_grouped.drop(columns=['U_Estilo', 'U_Silueta', 'U_Coleccion_NE', 'Talla', 'U_Liga', 'U_Team'])

# NOW using outer join merge df_grouped and df_stock_grouped on Bodega and Codigo_SAP and Pais
df_grouped = df_grouped.merge(df_stock_grouped, on=['Bodega', 'Codigo_SAP', 'Pais'], how='outer')

# None values to 0
df_grouped['Cantidad'] = df_grouped['Cantidad'].fillna(0)
df_grouped['Stock_Actual'] = df_grouped['Stock_Actual'].fillna(0)
df_grouped['dias_sin_venta'] = df_grouped['dias_sin_venta'].fillna(0)
########################################################################################################################################
# Now make a copy of df and df_stock
df_ventas_cat = df.copy()
df_stock_cat = df_stock.copy()


# Make a groupby Codigo_SAP, U_Estilo, U_Silueta, U_Coleccion_NE, Talla, U_Liga, U_Team in both dataframes
df_ventas_cat = df_ventas_cat.groupby(['Codigo_SAP', 'U_Estilo', 'U_Silueta', 'U_Coleccion_NE', 'Talla', 'U_Liga', 'U_Team','U_Descripcion']).agg({'Cantidad': 'sum'}).reset_index()
df_stock_cat = df_stock_cat.groupby(['Codigo_SAP', 'U_Estilo', 'U_Silueta', 'U_Coleccion_NE', 'Talla', 'U_Liga', 'U_Team','U_Descripcion']).agg({'Stock_Actual': 'sum'}).reset_index()

# Drop the Cantidad and Stock_Actual columns from df_ventas_cat and df_stock_cat
df_ventas_cat = df_ventas_cat.drop(columns=['Cantidad'])
df_stock_cat = df_stock_cat.drop(columns=['Stock_Actual'])

#####################################################################################################################################

# Concatenar los dos DataFrames
df_concat = pd.concat([df_ventas_cat, df_stock_cat], ignore_index=True)

# Eliminar duplicados por Codigo_SAP, dejando la primera ocurrencia
df_concat = df_concat.drop_duplicates(subset='Codigo_SAP', keep='first')

#st.dataframe(df_concat)
#####################################################################################################################################

# Merge df_grouped and df_concat on Codigo_SAP
df_grouped = df_grouped.merge(df_concat, on=['Codigo_SAP'], how='left')

# U_Estilo = NO APLICA DROP THEM
df_grouped = df_grouped[df_grouped['U_Estilo'] != 'NO APLICA']
# U_Silueta = NO APLICA DROP THEM
df_grouped = df_grouped[df_grouped['U_Silueta'] != 'NO APLICA']
# U_Coleccion_NE = NO APLICA DROP THEM
df_grouped = df_grouped[df_grouped['U_Coleccion_NE'] != 'NO APLICA']
# Talla = NO APLICA DROP THEM
df_grouped = df_grouped[df_grouped['Talla'] != 'NO APLICA']
# U_Liga = NO APLICA DROP THEM
df_grouped = df_grouped[df_grouped['U_Liga'] != 'NO APLICA']
# U_Team = NO APLICA DROP THEM
df_grouped = df_grouped[df_grouped['U_Team'] != 'NO APLICA']
# U_Descripcion = NO APLICA DROP THEM
df_grouped = df_grouped[df_grouped['U_Descripcion'] != 'NO APLICA']

# Reset index
df_grouped = df_grouped.reset_index(drop=True)

# Now order by Pais first GT,SV,PA,CR,HN
df_grouped['Pais'] = pd.Categorical(df_grouped['Pais'], categories=['GT', 'SV', 'PA', 'CR', 'HN'], ordered=True)
df_grouped = df_grouped.sort_values(by=['Pais', 'Bodega', 'Codigo_SAP'], ascending=[True, True, True])
# Reset index
df_grouped = df_grouped.reset_index(drop=True)
# DROP ALL BODEGAS that contains Mayoreo case insensitive
df_grouped = df_grouped[~df_grouped['Bodega'].str.contains('mayoreo', case=False, na=False)]
# ORDER BY Cantidad mayor a menor and reset index
df_grouped = df_grouped.sort_values(by=['Cantidad'], ascending=False)
# Reset index
df_grouped = df_grouped.reset_index(drop=True)

#################################################################################################################################

# Reorder columns
df_grouped = df_grouped[['Pais', 'Bodega', 'Codigo_SAP', 'U_Estilo','U_Descripcion', 'U_Silueta', 'U_Coleccion_NE', 'Talla', 'U_Liga', 'U_Team', 'dias_sin_venta', 'Cantidad', 'Stock_Actual']]
# Rename columns

##################################################################################################################################
# Ahora en una nueva comlumna saca el SellsThrough Cantidad/(Cantidad + Stock_Actual) * 100
df_grouped['SellsThrough'] = (df_grouped['Cantidad'] / (df_grouped['Cantidad'] + df_grouped['Stock_Actual'])) * 100
# Round to 2 decimals
df_grouped['SellsThrough'] = df_grouped['SellsThrough'].round(2)
# Percentage format
df_grouped['SellsThrough'] = df_grouped['SellsThrough'].astype(str) + '%'
###########################################################################################################################################
# Saca promedio el promedio mensual Cantidad/12
df_grouped['Promedio_Mensual'] = (df_grouped['Cantidad'] / 12).round(2)
# Round to 2 decimals
df_grouped['Promedio_Mensual'] = df_grouped['Promedio_Mensual'].round(2)

### Stock_Meses column Stock_Actual/Promedio_Mensual
df_grouped['Stock_Meses'] = (df_grouped['Stock_Actual'] / df_grouped['Promedio_Mensual']).round(0)




st.dataframe(df_grouped)

####################################################################################################################################
####################################################################################################################################

import numpy as np
import pandas as pd

# 1. Agrupar ventas y stock incluyendo Codigo_SAP, U_Coleccion_NE y Talla
cols_group = ['Pais', 'U_Coleccion_NE', 'Bodega', 'Codigo_SAP', 'U_Estilo', 'Talla']

df_ventas_cat = df.groupby(cols_group).agg({'Cantidad': 'sum'}).reset_index()
df_stock_cat = df_stock.groupby(cols_group).agg({'Stock_Actual': 'sum'}).reset_index()

# 2. Limpieza: filtrar NO APLICA y bodegas mayoreo
for df_tmp in [df_ventas_cat, df_stock_cat]:
    df_tmp = df_tmp[df_tmp['U_Estilo'] != 'NO APLICA']
    df_tmp = df_tmp[~df_tmp['Bodega'].str.contains('mayoreo', case=False, na=False)]

# 3. Merge ventas + stock
df_merge = pd.merge(df_ventas_cat, df_stock_cat, on=cols_group, how='outer')
df_merge['Cantidad'] = df_merge['Cantidad'].fillna(0)
df_merge['Stock_Actual'] = df_merge['Stock_Actual'].fillna(0)

# 4. Cálculos principales a nivel Codigo_SAP y Talla
df_merge['SellThrough'] = (df_merge['Cantidad'] / (df_merge['Cantidad'] + df_merge['Stock_Actual'])).fillna(0) * 100
df_merge['SellThrough'] = df_merge['SellThrough'].round(2)
df_merge['Promedio_Mensual'] = (df_merge['Cantidad'] / 12).round(2)
df_merge['Stock_Meses'] = (df_merge['Stock_Actual'] / df_merge['Promedio_Mensual']).replace([np.inf, -np.inf], 0).fillna(0).round(0)

# 5. Flag SellThrough >= 50 a nivel Colección
df_merge['SellThrough_flag'] = df_merge['SellThrough'] >= 50

# 6. Elegir bodegas destino a nivel U_Coleccion_NE y Pais (max Promedio_Mensual)
df_good = df_merge[df_merge['SellThrough_flag']]
idx = df_good.groupby(['U_Coleccion_NE', 'Pais'])['Promedio_Mensual'].idxmax()
df_best = df_good.loc[idx][['U_Coleccion_NE', 'Pais', 'Bodega']].rename(columns={'Bodega': 'Bodega_Destino'})

# 7. Merge para sugerir traslados (excluir bodega destino igual a origen)
df_trash = df_merge.merge(df_best, on=['U_Coleccion_NE', 'Pais'], how='inner')
df_trash = df_trash[df_trash['Bodega'] != df_trash['Bodega_Destino']]

# 8. Stock en bodega destino (a nivel U_Coleccion_NE, Pais, Bodega_Destino)
stock_destino = df_merge.groupby(['Pais', 'U_Coleccion_NE', 'Bodega'])['Stock_Actual'].sum().reset_index()
stock_destino = stock_destino.rename(columns={'Bodega': 'Bodega_Destino', 'Stock_Actual': 'Stock_Destino'})

df_trash = df_trash.merge(stock_destino, on=['Pais', 'U_Coleccion_NE', 'Bodega_Destino'], how='left')
df_trash['Stock_Destino'] = df_trash['Stock_Destino'].fillna(0)

# 9. Limitar máximo 12 unidades a recibir en destino por colección y país
stock_total_destino = df_trash.groupby(['Pais', 'U_Coleccion_NE', 'Bodega_Destino'])['Stock_Destino'].max().reset_index()
stock_total_destino['Limite_Recepcion'] = 12 - stock_total_destino['Stock_Destino']
stock_total_destino['Limite_Recepcion'] = stock_total_destino['Limite_Recepcion'].clip(lower=0)

df_trash = df_trash.merge(stock_total_destino[['Pais', 'U_Coleccion_NE', 'Bodega_Destino', 'Limite_Recepcion']],
                          on=['Pais', 'U_Coleccion_NE', 'Bodega_Destino'], how='left')

# 10. Calcular unidades traslado: mínimo entre stock actual de origen y limite de recepcion en destino
df_trash['Unidades_Traslado'] = df_trash[['Stock_Actual', 'Limite_Recepcion']].min(axis=1).round(0).astype(int)
df_trash = df_trash[df_trash['Unidades_Traslado'] > 0]

# 11. Agregar datos de la bodega origen
df_origen = df_merge[[
    'Pais', 'U_Coleccion_NE', 'Bodega', 'Codigo_SAP', 'U_Estilo', 'Talla',
    'Stock_Actual', 'SellThrough', 'Promedio_Mensual'
]].rename(columns={
    'Bodega': 'Bodega_Origen',
    'Stock_Actual': 'Stock_Origen',
    'SellThrough': 'SellThrough_Origen',
    'Promedio_Mensual': 'Prom_Mensual_Origen'
})

df_trash = df_trash.merge(
    df_origen,
    left_on=['Pais', 'U_Coleccion_NE', 'Bodega', 'Codigo_SAP', 'U_Estilo', 'Talla'],
    right_on=['Pais', 'U_Coleccion_NE', 'Bodega_Origen', 'Codigo_SAP', 'U_Estilo', 'Talla'],
    how='left'
)

# 12. Agregar datos de la bodega destino
df_destino = df_merge[[
    'Pais', 'U_Coleccion_NE', 'Bodega', 'Codigo_SAP', 'U_Estilo', 'Talla',
    'SellThrough', 'Promedio_Mensual'
]].rename(columns={
    'Bodega': 'Bodega_Destino',
    'SellThrough': 'SellThrough_Destino',
    'Promedio_Mensual': 'Prom_Mensual_Destino'
})

df_trash = df_trash.merge(
    df_destino,
    on=['Pais', 'U_Coleccion_NE', 'Bodega_Destino', 'Codigo_SAP', 'U_Estilo', 'Talla'],
    how='left'
)

# 13. Resultado final con todo lo solicitado
traslados = df_trash[[
    'Pais', 'U_Coleccion_NE', 'Codigo_SAP', 'U_Estilo', 'Talla',
    'Bodega_Origen', 'Stock_Origen', 'SellThrough_Origen', 'Prom_Mensual_Origen',
    'Bodega_Destino', 'Stock_Destino', 'SellThrough_Destino', 'Prom_Mensual_Destino',
    'Unidades_Traslado'
]].copy()

traslados['Pais'] = pd.Categorical(traslados['Pais'], categories=['GT', 'SV', 'PA', 'CR', 'HN'], ordered=True)
traslados = traslados.sort_values(['Pais', 'U_Coleccion_NE', 'Codigo_SAP', 'Talla']).reset_index(drop=True)

# Drop in the Bodega colum all Bodegas that contains mayoreo case insensitive
traslados = traslados[~traslados['Bodega_Origen'].str.contains('mayoreo', case=False, na=False)]
traslados = traslados[~traslados['Bodega_Destino'].str.contains('mayoreo', case=False, na=False)]
# Solo rellena las columnas numéricas con 0
for col in traslados.select_dtypes(include=[np.number]).columns:
    traslados[col] = traslados[col].fillna(0)


#Using markdown h1 title movimientos NEW ERA
st.markdown("<h1 style='text-align: center; color: black;'>🌎 MOVIMIENTOS NEW ERA POR COLECCIÓN</h1>", unsafe_allow_html=True)
  # Multiselects para traslados: 2 por columna, organizados en 4 columnas visuales

col1, col2, col3, col4 = st.columns(4)

with col1:
    pais_sel = st.multiselect('Pais', traslados['Pais'].unique(), max_selections=2)
    coleccion_sel = st.multiselect('Colección', traslados['U_Coleccion_NE'].unique(), max_selections=2)
with col2:
    codigo_sap_sel = st.multiselect('Codigo SAP', traslados['Codigo_SAP'].unique(), max_selections=2)
    estilo_sel = st.multiselect('Estilo', traslados['U_Estilo'].unique(), max_selections=2)
with col3:
    talla_sel = st.multiselect('Talla', traslados['Talla'].unique(), max_selections=2)
    bodega_origen_sel = st.multiselect('Bodega Origen', traslados['Bodega_Origen'].unique(), max_selections=2)
with col4:
    bodega_destino_sel = st.multiselect('Bodega Destino', traslados['Bodega_Destino'].unique(), max_selections=2)

# Aplica los filtros seleccionados
if pais_sel:
    traslados = traslados[traslados['Pais'].isin(pais_sel)]
if coleccion_sel:
    traslados = traslados[traslados['U_Coleccion_NE'].isin(coleccion_sel)]
if codigo_sap_sel:
    traslados = traslados[traslados['Codigo_SAP'].isin(codigo_sap_sel)]
if estilo_sel:
    traslados = traslados[traslados['U_Estilo'].isin(estilo_sel)]
if talla_sel:
    traslados = traslados[traslados['Talla'].isin(talla_sel)]
if bodega_origen_sel:
    traslados = traslados[traslados['Bodega_Origen'].isin(bodega_origen_sel)]
if bodega_destino_sel:
    traslados = traslados[traslados['Bodega_Destino'].isin(bodega_destino_sel)]
st.dataframe(traslados)

# ADD A BUTTON TO DOWNLOAD THE DATAFRAME AS A CSV FILE separated by semicolon
csv = traslados.to_csv(sep=';', index=False)
st.download_button(
    label="Download Traslados",
    data=csv,
    file_name='traslados.csv',
    mime='text/csv',
    key='download-csv'
)