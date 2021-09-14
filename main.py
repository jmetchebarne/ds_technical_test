import streamlit as st
import pandas as pd

#################
### Funciones ###
#################
# @st.cache(allow_output_mutation=True)
# def load_data(file_path):
#     """
#     Carga la data asociada al dataset del archivo de configuracion. Se asume que
#     el archivo a leer es un tipo csv separado por ";".
#
#     Parametros
#     ----------
#     Retorno
#     --------
#     df: DataFrame
#         DataFrame que contiene el dataset del archivo de configuracion
#         ya cargado.
#     """
#     try:
#         df = pd.read_csv(file_path, sep=';', low_memory=False)
#         return df
#     except Exception as e:
#         # Cargamos en la pagina que la data no se puedo cargar correctamente
#         st.title('⭕️No se pudo cargar la data correctamente')
#         st.write('Para mayor informacion sobre el error, puede revisar lo siguiente: ', e)
#
#
# def image_to_base64(image_full_path):
#     """
#     Procesa una imagen para poder ser cargada a Streamlit para extensiones que no son
#     .jpg o .png.
#
#     Parametros
#     ----------
#     Retorno
#     --------
#     """
#     file_ = open(image_full_path, "rb")
#     contents = file_.read()
#     data_url = base64.b64encode(contents).decode("utf-8")
#     file_.close()
#
#     st.markdown(
#         f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
#         unsafe_allow_html=True,
#     )
#
#
# def estudio_de_datos(app_mode, df):
#     """
#     Procesa una imagen para poder ser cargada a Streamlit para extensiones que no son
#     .jpg o .png.
#
#     Parametros
#     ----------
#     Retorno
#     --------
#     """
#     this_function = False
#     if app_mode == 'Estudio de Datos':
#         this_function = True
#     if this_function:
#         st.write('Lo primero es ver los datos, tal cual vienen, para poder ver que se necesita hacer con ellos.')
#         st.dataframe(df.head(25))
#         st.write('El tamaño del dataset es de: ', df.shape)
#         col1, col2 = st.beta_columns(2)
#         col1.write(' '
#                    ' '
#                    ' ')
#         col1.write('Vemos cuantos valores nulos hay en el dataset por columna, como se ve en la siguiente tabla.')
#         col1.write('Como se ve en la tabla, la mayoria de las columnas presentan todos sus datos, pero una tiene '
#                    'muchos datos faltantes. En este caso, eliminaremos dichas filas, pero esto se puede rellenar '
#                    'con algun valor que el cliente desee.')
#         col2.write(df.isnull().sum())
#
#     df = df.dropna()
#     df = cyt.parseQuantityColumn(df, 'Cant Real', 'cantRealParsed')
#     df['newProduct'] = df.Marca + '--' + df.Cepa + '--' + df.Formato
#
#     if this_function:
#         st.write('Lo siguiente, es revisar las columnas que utilizaremos para nuestros calculos y estimaciones.')
#         st.write('Para esto, usaremos la correlacion entre los datos. Una alta correlacion nos da indicio de que '
#                  'ambas variables estan representando algo similar.')
#         s, corr = cyt.matriz_correlacion(df)
#         st.dataframe(s)
#         st.write('En base a esto, las siguientes columnas serian dependientes entre si: ')
#         dup_list = cyt.eliminar_columnas_alta_correlacion(df, corr)
#         if len(dup_list) > 0:
#             dicc_list = []
#             for dicc in dup_list:
#                 dicc_list.append(dicc)
#             # Sacamos las filas pares para que no se repitan
#             dicc_list = dicc_list[::2]
#             list_df = pd.DataFrame(columns=['Columna 1', 'Columna 2'], data=dicc_list)
#             st.dataframe(list_df)
#         else:
#             st.write('No hay columnas que tengan una correlacion lo suficientemente alta para poder decir que ambas '
#                      'representan la misma informacion.')
#         st.write('Recordar que mientras mas pequeño sea el dataset, mayor error puede haber en esta parte, debido a '
#                  'que puede haber informacion insuficiente sobre el comportamiento de cada columna. '
#                  'Este paso es opcional, pero suele ser necesario.')
#
#     if not this_function:
#         # Si tenemos otra pagina cargada, devolvemos el DataFrame modificado
#         return df
#
#
# def prediccion_proxima_compra(app_mode, df):
#     """
#     Codigo utilizado en la seccion para predecir la proxima compra.
#
#     Parametros
#     ----------
#     Retorno
#     --------
#     """
#     this_function = False
#     if app_mode == 'Prediccion Proxima Compra':
#         this_function = True
#     # Primero, ejecutamos el codigo anterior
#     df = estudio_de_datos(app_mode, df)
#     # Dropeamos las otras columnas de cantidad
#     df = df.drop(['Cant', 'Cant Real'], axis=1)
#     # Pasamos la columna Año/mes para que sea en tiempo
#     df['Año/mes'] = df['Año/mes'].astype(int)
#     df['Año/mes'] = df['Año/mes'].astype(str)
#     # st.write(df['Año/mes'])
#     df['Año/mes'] = [x[:4] + '-' + x[-2:] for x in df['Año/mes']]
#     # st.write(df['Año/mes'])
#     df['Año/mes'] = pd.to_datetime(df['Año/mes']).dt.date
#     if this_function:
#         st.write('En esta seccion, utilizaremos el historial de compras del usuario para predecir su proxima compra '
#                  'de un producto determinado.')
#         st.write('Para este punto, daremos la opcion al usuario de agrupar como guste los datos, ya sea filtrando '
#                  'por lugar, vendedor, u otra columna.')
#         # Agrupamos por lo que el usuario quiera
#         col_name, col_value = st.beta_columns(2)
#         col_name = st.sidebar.selectbox('Por que columna desea agrupar?', df.columns[1:])
#         col_value = st.sidebar.selectbox('Por que valor desea filtrar?', df[col_name].unique())
#         df = df[df[col_name] == col_value]
#         st.write('Con esto, obtenemos el siguiente set de datos: ')
#         st.dataframe(df.head(5))
#         cliente = st.sidebar.selectbox('Elija un Cliente', df['Cliente'].unique())
#         cliente_df = df[df['Cliente'] == cliente]
#         cliente_df.sort_values(['Año/mes'], inplace=True)
#         # Vemos el resultado y metricas precision and recall
#         cliente_df = cyt.prediccion_proxima_compra_1(cliente_df)
#         producto = st.sidebar.selectbox('Elija un producto', cliente_df.columns[1:])
#         st.write('El producto elegido para la prediccion es: ', producto[11:])
#         percent = st.sidebar.slider('Porcentaje de Entrenamiento',min_value=0.5,max_value=0.9,value=0.8,step=0.1)
#         yhat, future_values, ber_rmse, cm, precision, recall, algorithm = cyt.metricas_prediccion_temporal(cliente_df, producto, percent)
#         st.write('yhat',yhat)
#         st.write('futureval',future_values)
#         st.write('cm',cm)
#         # st.write('cm1',cm1)
#         # Guardamos las metricas en mlflow
#         fmt = ".2f"
#         thresh = cm.max() / 2.
#         descriptions = np.array([["True Positive", "False Negative"], ["False Positive", "True Negatives"]])
#         colors = np.array([["green", "red"], ["red", "green"]])
#         plt.imshow([[0, 0], [0, 0]], interpolation='nearest', cmap=plt.cm.Greys)
#         for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#             plt.text(j, i, format(cm[i, j], fmt) + '%\n' + descriptions[i, j],
#                      horizontalalignment="center",
#                      color=colors[i, j])
#         plt.axhline(y=0.5, xmin=0, xmax=1, color="black", linewidth=0.75)
#         plt.axvline(x=0.5, ymin=0, ymax=1, color="black", linewidth=0.75)
#         plt.ylabel('True')
#         plt.xlabel('Predicted')
#         plt.title("Confusion Matrix")
#         plt.xticks([0, 1], [1, 0], rotation=45)
#         plt.yticks([0, 1], [1, 0])
#         img = plt.show()
#         st.pyplot(img)
#         st.write('rmse', ber_rmse)
#         st.write('precision',precision)
#         st.write('recall',recall)
#         st.write('algorithm',algorithm)
#         cliente_df = cliente_df[[producto]]
#         recom_df = cyt.prediccion_proxima_compra_2(cliente_df, producto)
#         st.dataframe(recom_df.style.applymap(cyt.highlight_first))
#     elif not this_function:
#         return df
#
#
# def seleccion_de_variables(app_mode, df):
#     """
#     Codigo utilizado en la seccion para predecir la proxima compra.
#
#     Parametros
#     ----------
#     Retorno
#     --------
#     """
#     this_function = False
#     if app_mode == 'Seleccion de Variables':
#         this_function = True
#     # Primero, ejecutamos los codigos anteriores
#     df = prediccion_proxima_compra(app_mode, df)
#     # Le damos la opcion al cliente de elegir las columnas
#     desea_elegir = st.sidebar.selectbox('Desea elegir las variables?', ['No', 'Si'])
#     if desea_elegir == 'Si':
#         # Ahora dejamos que el cliente elija las columnas
#         if this_function:
#             st.write('Como usted eligio que va a elegir las variables, le mostraremos nuevamente el set de datos para '
#                      'que vea con que columnas le gustari a quedarse.')
#             st.dataframe(df.head(10))
#             st.write('Por favor, seleccione las columnas en el panel izquierdo.')
#         columns_list = list(df.columns)
#         # Sacamos cantRealParsed ya que ira si o si
#         columns_list.remove('cantRealParsed')
#         columnas_seleccionadas = st.sidebar.multiselect('Seleccione las columnas',columns_list)
#         if len(columnas_seleccionadas) > 0:
#             if this_function:
#                 st.write('Usted eligio quedarse con las columnas: ')
#             columnas_write = 'cantRealParsed'
#             for column in columnas_seleccionadas:
#                 columnas_write = columnas_write + ', ' + column
#                 # st.write(column)
#             if this_function:
#                 st.write('Las columnas elegidas son: ', columnas_write)
#             # st.write('Usted eligio quedarse con las columnas: ',columnas_seleccionadas)
#             if this_function:
#                 st.write('Ya con las columnas elegidas, filtramos nuestro dataset: ')
#             columnas_seleccionadas.append('cantRealParsed')
#             df = df[columnas_seleccionadas]
#             if this_function:
#                 st.dataframe(df.head(10))
#             return df
#     elif desea_elegir == 'No':
#         # Ejecutamos el codigo de seleccion de variable, para esto usaremos Lasso
#         if this_function:
#             st.write('Este procedimiento requiere pasar todas las columnas a columnas numericas.')
#         # Pasamos las columnas a numericas, primero buscamos las columnas
#         filtered_columns = df.dtypes[df.dtypes == np.object]
#         list_of_column_names = list(filtered_columns.index)
#         all_cols_dict = {}
#         for col in list_of_column_names:
#             col_dict = cyt.df_col_string_dict(df[col])
#             all_cols_dict[col] = col_dict
#         # Reemplazamos para cada columna
#         for col in list_of_column_names:
#             df[col] = df[col].map(all_cols_dict[col])
#         if this_function:
#             st.write(df.head(10))
#         # Feature Selection
#         df = df.reset_index(drop=True)
#         X = df.drop(['cantRealParsed'], axis=1)
#         y = df['cantRealParsed'].to_numpy()
#         # No usaremos la chi2 ya que requiere que sean positivos sus valores
#         num_k = st.sidebar.slider('Elija el numero de columnas a elegir.', min_value=1, max_value=len(X.columns),
#                                   value=5, step=1)
#         select = SelectKBest(f_regression, k=num_k)
#         select.fit_transform(X, y)
#         mask = select.get_support(indices=True)
#         # Forzamos que este newProduct
#         # mask = np.append(mask, [df.shape[1]])
#         df = df.iloc[:, mask]
#         df['cantRealParsed'] = y
#         # Reemplazamos por los valores originales
#         for col in df.columns:
#             if col in all_cols_dict:
#                 temp_dict = all_cols_dict[col]
#                 temp_dict = {v: k for k, v in temp_dict.items()}
#                 df[col] = df[col].map(temp_dict)
#         if this_function:
#             st.write('En este caso, la columna cantRealParsed siempre estara, por lo que si selecciona, por ejemplo, '
#                      '4 variables, el nuevo set de datos tendra 5 columnas.')
#             st.write(df.head(10))
#             not_in_common = list(set(list(X.columns)) ^ set(list(df.columns)))
#             st.write('Por otra parte, las columnas que fueron descartadas son: ', not_in_common)
#         return df
#
#
# def clustering(app_mode, df):
#     """
#     Codigo utilizado en la seccion para predecir la proxima compra.
#
#     Parametros
#     ----------
#     Retorno
#     --------
#     """
#     st.write('hola')
#
# def recomendaciones(df):
#     """
#     Codigo utilizado en la seccion para predecir la proxima compra.
#
#     Parametros
#     ----------
#     Retorno
#     --------
#     """
#     st.write('sola')

##############################
### Funciones de Streamlit ###
##############################
# def cargar_pagina_seleccionada(app_mode, df):
#     if app_mode == 'Pagina de Inicio':
#         cargar_pagina_inicio()
#     elif app_mode == "Estudio de Datos":
#         cargar_estudio_de_datos(app_mode, df)
#     elif app_mode == "Prediccion Proxima Compra":
#         cargar_prediccion_proxima_compra(app_mode, df)
#     elif app_mode == "Seleccion de Variables":
#         cargar_seleccion_de_variables(app_mode, df)
#     elif app_mode == "Clustering":
#         cargar_clustering(app_mode, df)
#     elif app_mode == "Sugerencia de Ventas a Clientes":
#         cargar_recomendaciones(df)
#
#
# def cargar_pagina_inicio():
#     """
#     Breve introduccion al problema a revisar. La idea es explicar el problema en general, explicando que
#     estos problemas pueden ser vistos tanto de la oferta como de la demanda. Mencionar otras cosas que
#     puedan ser de interes al cliente.
#
#     Parametros
#     ----------
#     Retorno
#     --------
#     df: DataFrame
#         DataFrame que contiene el dataset del archivo de configuracion
#         ya cargado.
#     """
#     # cyt_logo_full_path = images_path + 'logo-Concha-y-Toro.jpg'
#     # image_cyt = Image.open(cyt_logo_full_path)
#     # st.image(image_cyt, use_column_width=False)
#     # logo_cyt_full_path = images_path + 'logo-cyt.png'
#     # logo_cyt = Image.open(logo_cyt_full_path)
#     # st.image(logo_cyt, use_column_width=False)
#     st.title('Motor de Recomendaciones para Concha y Toro')
#     # '<p class="big-font"> </p>', unsafe_allow_html=True
#     st.markdown(
#         'Motor de recomendaciones. Con esto se busca un sistema interactivo para probar distintos metodos '
#         'para probar y revisar las configuraciones que obtienen los mejores resultados.')
#     st.markdown('Primero partiremos con el problema irrestricto, luego, a medida que se vaya avanzando con el cliente, '
#                 'se le pueden ir agregando las restricciones que ellos estimen adecuadas.')
#     st.markdown('Finalmente, recordar que la idea de esto tambien es darle un enfoque distinto a lo que se esta '
#                 'realizando actualmente, ya que el sistema no se basara en reglas de negocios, mas bien se '
#                 'va a permitir mostrar algoritmos que tengan resultados cuantificables, que a futuro pueden irse '
#                 'modificando para lograr mejores resultados.')
#     st.subheader(":wine_glass: Estudio de Datos :wine_glass:")
#     st.markdown("* En este punto se realizara un estudio basico de los datos. Ya sea estudiar la "
#                 "cantidad de datos, datos faltantes, entre otras cosas.")
#     # st.subheader(":wine_glass: Prediccion Proxima Compra :wine_glass:")
#     # st.markdown("* En esta seccion se va a predecir la proxima compra del Cliente en base a un "
#     #             "historial netamente propio, sin tomar en cuenta a los demas clientes.")
#     st.subheader(":wine_glass: Seleccion de Variables :wine_glass:")
#     st.markdown("* La idea de esta seccion es reducir la cantidad de informacion del dataset, lo cual agilizara "
#                 "los calculos y hara que las predicciones sean mas certeras. Se dara la opcion de hacerlo de manera "
#                 "manual,y tambien de manera automarica.")
#     st.subheader(":wine_glass: Clustering :wine_glass:")
#     st.markdown("* El proposito de esta seccion es desarrollar clusters de la data, ya sea clusters de usuarios, "
#                 "clusters filtrador por ciertas columnas elegidas por el usuario, o columnas elegidas "
#                 "por el mismo programa.")
#     st.subheader(":wine_glass: Sugerencia de Ventas a Clientes :wine_glass:")
#     st.markdown("* Este ultimo punto busca entregar a cada vendedor un esquema de los clientes basados en el "
#                 "Clustering realizado anteriormente para poder sugerir no solamente que producto sugerirle, "
#                 "si no tambien su cantidad y cuando se deberia hacer.")
#
#
# def cargar_estudio_de_datos(app_mode, df):
#     st.title('Estudio de Datos')
#     st.markdown('En este capitulo, nos enfocaremos en estudiar los datos recibidos, tanto limpieza y procesamiento '
#                 'como estudios estadisticos que serviran para las secciones futuras.')
#     # st.sidebar.title('Configuraciones')
#     # st.sidebar.markdown('Estas configuraciones alteraran los calculos de las recomendaciones. Evite editar las '
#     #                     'configuraciones asociadas a nombres o programas que no conoce ya que pueden reducir la '
#     #                     'precision de la recomendacion.')
#     estudio_de_datos(app_mode, df)
#
#
# def cargar_prediccion_proxima_compra(app_mode, df):
#     st.title('Prediccion Proxima Compra')
#     st.markdown('En esta seccion, se va a predecir si el cliente comprara o no un producto que el ya ha comprado '
#                 'anteriormente en los proximos meses. Para tener una mejor prediccion en este punto es importante '
#                 'tener la mayor cantidad de datos posibles para dicho cliente.')
#     st.sidebar.title('Configuraciones')
#     st.sidebar.markdown('Estas configuraciones alteraran los calculos de las recomendaciones. Evite editar las '
#                         'configuraciones asociadas a nombres o programas que no conoce ya que pueden reducir la '
#                         'precision de la recomendacion.')
#     prediccion_proxima_compra(app_mode, df)
#
#
# def cargar_seleccion_de_variables(app_mode, df):
#     st.title('Seleccion de Variables')
#     st.markdown('La idea de este punto es filtrar el dataset original, y quedandonos con menos columnas de este, '
#                 'lo cual no solamente va a hacer que los calculos sean mas rapidos, tambien suele ser bastante '
#                 'mas preciso el calculo.')
#     st.markdown('Esto se debe a que la mayoria de las veces, los datasets contienen cierta informacion que es util '
#                 'para la empresa, pero que no tiene mucho que ver con la informacion que nos gustaria poder predecir. '
#                 'En nuestro caso, usamos la columna cantRealParsed como la columna que nos gustaria poder predecir, '
#                 'ya que esta columna nos entrega las ventas mensuales para cada combinacion de producto y cliente.')
#     st.sidebar.title('Configuraciones')
#     st.sidebar.markdown('Estas configuraciones alteraran los calculos de las recomendaciones.')
#     seleccion_de_variables(app_mode, df)
#
#
# def cargar_clustering(app_mode, df):
#     st.title('Clustering')
#     st.markdown('Este punto muestra como las variables se pueden juntar en un mismo cluster, es decir, un conjunto de '
#                 'variables que comparten algun tipo de similitud.')
#     st.sidebar.title('Configuraciones')
#     st.sidebar.markdown('Estas configuraciones alteraran los calculos de las recomendaciones.')
#     clustering(app_mode, df)
#
#
# def cargar_recomendaciones(df):
#     st.title('Recomendaciones')
#     st.markdown('')
#     st.sidebar.title('Configuraciones')
#     st.sidebar.markdown('Estas configuraciones alteraran los calculos de las recomendaciones.')
#     recomendaciones(df)


def configuraciones_streamlit():
    """
    Configuraciones de Streamlit.

    Parametros
    ----------
    Retorno
    --------
    """
    # Para que no muestre el mensaje de warning de Pyplot
    st.set_option('deprecation.showPyplotGlobalUse', False)
    # st.set_page_config(layout="wide")

    # Configuraciones de resolucion
    max_width = '1200'
    padding_top = 0
    padding_right = 0
    padding_left = 0
    padding_bottom = 0
    big_font = 50
    st.markdown(
        """
    <style>
        .big-font {font-size:24px}
        .reportview-container .main .block-container{max-width:1200px}
    </style>
    """,
        unsafe_allow_html=True,
    )


# .reportview - container .main .block-container{{
#     max-width: {max_width}px;
#     padding-top: {padding_top}rem;
#     padding-right: {padding_right}rem;
#     padding-left: {padding_left}rem;
#     padding-bottom: {padding_bottom}rem;
# }}


##############
### Codigo ###
##############
# Ejecutamos configuraciones basicas, como de resolucion, entre otras
configuraciones_streamlit()

# Logo de Jooycar
# logo_cyt_full_path = images_path + 'logo-cyt.png'
# logo_cyt = Image.open(logo_cyt_full_path)
# st.image(logo_cyt, width=400)

# Cargamos el df, de no poder cargarlo tirara error
# is_loaded_header = st.sidebar.subheader(":x:️ Data not loaded")  # Revisar como se ve esto
# df = load_data(file_path)

# Si  esta cargada, mostramos una representacion visual de esto y cargamos el menu inicial
# is_loaded_header.subheader("✔️Data loaded successfully")
st.sidebar.title("Menu")
app_mode = st.sidebar.selectbox("Seleccione una pagina", ["Pagina de Inicio",
                                                          "Estudio de Datos",
                                                          "Prediccion Proxima Compra",
                                                          "Seleccion de Variables",
                                                          "Clustering",
                                                          "Sugerencia de Ventas a Clientes"])

# Vemos que pagina hay que cargar
# cargar_pagina_seleccionada(app_mode, df)
