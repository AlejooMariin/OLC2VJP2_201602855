from logging import exception
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier ,plot_tree,export_graphviz
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error, r2_score
import re



import requests, os
mpl.use("agg")

from matplotlib.backends.backend_agg import RendererAgg
_lock = RendererAgg.lock

# -- Set page config
apptitle = 'OLC2 Proyecto 2'
st.set_page_config(page_title=apptitle, page_icon=":chart_with_upwards_trend:")

# -- Default detector list
Algoritmolist = ['Regresión lineal.',
                'Regresión polinomial.', 
                'Clasificador Gaussiano.', 
                'Clasificador de árboles de decisión.', 
                'Redes neuronales.'
                ]
Valx=[]
Valy=[]
# Title the app
st.title('Machine Learning')

st.markdown("""
 * Utilice el menú de la izquierda para seleccionar datos y establecer parámetros de trazado.
 * En la parte de abajo se mostraran sus resultados.
""")
try:
 uploaded_file = st.file_uploader("Subir Archivo", type=(".csv", ".xls", ".xlsx", ".json"))

 if uploaded_file is None:
      st.sidebar.warning("Debe Subir un Archivo para poder habilitar los componentes y poder realizar los algoritmos.")
 elif uploaded_file is not None:
     # To read file as bytes:
    #  bytes_data = uploaded_file.getvalue()
    #  st.write(bytes_data)

    #  # To convert to a string based IO:
    #  stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    #  st.write(stringio)

    #  # To read file as string:
    #  string_data = stringio.read()
    #  st.write(string_data)

     # Can be used wherever a "file-like" object is accepted:
     #Aqui se leera los tipos de archivos
     Filname=uploaded_file.name
     vext=Filname.split('.')
     ext= vext[1]
     #st.write(ext)
     if ext.lower()=='csv':
      dataframe = pd.read_csv(uploaded_file)
     if ext.lower()=='xls':
      dataframe = pd.read_excel(uploaded_file)
     if ext.lower()=='xlsx':
      dataframe = pd.read_excel(uploaded_file)
     if ext.lower()=='json':
      dataframe = pd.read_json(uploaded_file)

     st.write(dataframe)
     
     Valx=dataframe.columns
     Valy=dataframe.columns
     st.sidebar.markdown("## Parametros de ejecucion")
    #**** Inicio de la barra **
     select_Algoritmo = st.sidebar.selectbox('Algoritmo a Ejecutar', Algoritmolist)
    #**** Fin de la barra **

            #Algotirmo
     if select_Algoritmo=='Regresión lineal.':
        st.subheader('Regresión lineal.')
        with st.sidebar.expander("Informacion Sobre el Algoritmo"):
          st.markdown("""
                      Que es?
                     * La regresión lineal permite predecir el comportamiento de una variable (dependiente o predicha) a partir de otra (independiente o predictora).
                      
                      Para que sirve?
                     * Tiene presunciones como la linearidad de la relación, la normalidad, la aleatoridad de la muestra y homogeneidad de las varianzas.
            """) 
        try:
        #Combo1
         select_valx = st.sidebar.selectbox('Valores "X"',Valx)
        #Combo2
         select_valy = st.sidebar.selectbox('Valores "Y"',Valy)
        #Valor para el calculo de prediccion
         max_Ngrams = st.sidebar.number_input(
            "Valor de prediccion",
             value=0,
            #min_value=1,
            #max_value=10,
             help="""The maximum value for the keyphrase_ngram_range.
                    *Keyphrase_ngram_range* sets the length of the resulting keywords/keyphrases.
                    To extract keyphrases, simply set *keyphrase_ngram_range* to (1, 2) or higher depending on the number of words you would like in the resulting keyphrases.""",
          )
         var_x = select_valx
         var_y = select_valy
         x = np.asarray(dataframe[var_x]).reshape(-1, 1)
         y = dataframe[var_y]
        #Regresion Lineal
         regr = LinearRegression()

         regr.fit(x, y)
         y_pred = regr.predict(x)
         r2 = r2_score(y, y_pred)
        #Print
        #st.write('x: ', x)
        #st.write('y_pred: ', y_pred)
        #st.write('coef_: ', regr.coef_)
        # st.write('intercept: ', regr.intercept_)
        # st.write('r^2: ', r2)
        # st.write('Error cuadrático: ', mean_squared_error(y, y_pred))
        #Plot
         plt.scatter(x, y, color='green')
         plt.plot(x, y_pred, color='blue')
         st.pyplot(plt)
        #Funcion Lineal
         Fun="Y = "+str(regr.intercept_) +" + ("+str(regr.coef_).replace("[","").replace("]","")+")*X "
         st.write('Funcion Lineal: ')
         st.write(Fun)
        #Prediccion
         st.write('Prediccion: ',str(regr.predict([[max_Ngrams]])).replace("[","").replace("]",""))
         #st.write(str(regr.predict([[max_Ngrams]])).replace("[","").replace("]",""))
         
                 
        except Exception as e:
         print(e) 
         ##########################################################################################################################
     if select_Algoritmo=='Regresión polinomial.':
        st.subheader('Regresión polinomial.')
        with st.sidebar.expander("Informacion Sobre el Algoritmo"):
          st.markdown("""
                      Que es?
                     * Los modelos de regresión polinomial suelen ajustarse mediante el método de mínimos cuadrados. El método de mínimos cuadrados minimiza la varianza de los estimadores insesgados de los coeficientes.
                      
                      Para que sirve?
                     * El objetivo de la regresión polinomial es modelar una relación no lineal entre las variables independientes y dependientes (técnicamente, entre la variable independiente y la media condicional de la variable dependiente).
            """) 
        try:
         Vdegree = st.sidebar.number_input(
             "Grado",
             value=2,
             #min_value=1,
             #max_value=10,
             help=""" Valor de grado de la funcion.""",
          )
        #Combo1
         select_valx = st.sidebar.selectbox('Valores "X"',Valx)
         #Combo2
         select_valy = st.sidebar.selectbox('Valores "Y"',Valy)
         max_Ngrams = st.sidebar.number_input(
             "Valor de prediccion",
             value=0,
             #min_value=1,
             #max_value=10,
             help="""The maximum value for the keyphrase_ngram_range.
                    *Keyphrase_ngram_range* sets the length of the resulting keywords/keyphrases.
                    To extract keyphrases, simply set *keyphrase_ngram_range* to (1, 2) or higher depending on the number of words you would like in the resulting keyphrases.""",
          )
         var_x = select_valx
         var_y = select_valy
         x = np.asarray(dataframe[var_x]).reshape(-1, 1)
         y = dataframe[var_y]
        
         pf = PolynomialFeatures(degree = Vdegree)
         x_trans = pf.fit_transform(x)
         regr = LinearRegression()
         regr.fit(x_trans, y)
         y_pred = regr.predict(x_trans)
         rmse = np.sqrt(mean_squared_error(y, y_pred))
         r2 = r2_score(y, y_pred)
         st.write("Funcion Polinomial:")
         #st.write(regr.coef_)
         BodyPol=""
         ContPol=1

         for vx in regr.coef_ :
            if ContPol>1 :
               if vx<0 :
                  if(ContPol-1)==1:
                   BodyPol=BodyPol+" - "+str(vx).replace("-","")+" X"
                  elif (ContPol-1)>1:
                   BodyPol=BodyPol+" - "+str(vx).replace("-","")+" X^"+str(ContPol-1)
               elif vx>0:
                  if (ContPol-1)==1:
                   BodyPol=BodyPol+" + "+str(vx)+" X"
                  elif (ContPol-1)>1:    
                   BodyPol=BodyPol+" + "+str(vx)+" X^"+str(ContPol-1)
            ContPol=ContPol+1

         #Plot
         plt.scatter(x, y, color='green')
         plt.plot(x, y_pred, color='blue')
         st.pyplot(plt)
         Funstr= "Y = "+str(regr.intercept_)+BodyPol
         st.write(Funstr)
         st.write('RMSE: ', rmse)
         st.write('R^2: ', r2)
        #Prediccion
         pred = max_Ngrams
         x_new_min = pred
         x_new_max = pred
         x_new = np.linspace(x_new_min, x_new_max, 1)
         x_new = x_new[:, np.newaxis]
         x_trans = pf.fit_transform(x_new)
         st.write('Prediccion: ',"     "+str(regr.predict(x_trans)).replace("[","").replace("]",""))
         #st.write()
        
       ##########################################################################################################################
                
        except Exception as e:
         print(e) 

     if select_Algoritmo=='Clasificador Gaussiano.':
        st.subheader('Clasificador Gaussiano.')
        with st.sidebar.expander("Informacion Sobre el Algoritmo"):
          st.markdown("""
                      Que es?
                      * El Clasificador de Procesos Gaussianos es un algoritmo de aprendizaje de la máquina de clasificación.
                      
                      Para que sirve? 
                      * Los procesos gausianos son una generalización de la distribución de probabilidad gausiana y pueden utilizarse como base de sofisticados algoritmos no paramétricos de aprendizaje automático para la clasificación y la regresión.
                      * Son un tipo de modelo de núcleo, como los SVM, y a diferencia de éstos, son capaces de predecir probabilidades de pertenencia a una clase altamente calibradas, aunque la elección y configuración del núcleo utilizado en el núcleo del método puede ser un reto.
                      """) 
        try:
         select_valxlist = st.sidebar.multiselect('Valores de Columnas de interes',Valx)
         select_valylist = st.sidebar.multiselect('Valores de Columna de respuesta',Valy)

         x = np.asarray(dataframe[select_valxlist])#.reshape(-1, 1)
         y = np.asarray(dataframe[select_valylist])
         st.write("Valores X")
         st.write(x)
         st.write("Valores Y")
         st.write(y)

         st.sidebar.subheader("Valores de Interes")
         vals = st.sidebar.text_input("Los valores se separan por coma")
         List_Val = lambda x : [int(i) for i in re.split(",",x) if i!=""]

         arraylist = List_Val(vals)
         valInit= list(map(float,arraylist))
        
         clf = GaussianNB()
        # Adaptación de datos
         clf.fit(x,y)
         st.write("Prediccion:")
         st.write(clf.predict([valInit]))
        #st.write("==Predict result by predict_proba==")
        # print(clf.predict_proba([[-0.8, -1]]))
        #st.write("==Predict result by predict_log_proba==")
        # print(clf.predict_log_proba([[-0.8, -1]]))
        ##########################################################################################################################      
        except Exception as e:
                  print(e) 
         
     if select_Algoritmo=='Clasificador de árboles de decisión.':
        st.subheader('Clasificador de árboles de decisión.')
        with st.sidebar.expander("Informacion Sobre el Algoritmo"):
          st.markdown("""
                      Que es?
                     * Un árbol de decisión o un árbol de clasificación es un árbol en el que cada nodo interno (no hoja) está etiquetado con una función de entrada. Los arcos procedentes de un nodo etiquetado con una característica están etiquetados con cada uno de los posibles valores de la característica.
                      
                      Para que sirve?
                     * Un árbol puede ser "aprendido" mediante el fraccionamiento del conjunto inicial en subconjuntos basados en una prueba de valor de atributo. Este proceso se repite en cada subconjunto derivado de una manera recursiva llamada particionamiento recursivo. La recursividad termina cuando el subconjunto en un nodo tiene todo el mismo valor de la variable objetivo, o cuando la partición ya no agrega valor a las predicciones.
            """) 
        select_valxlist = st.sidebar.multiselect('Valores de Columnas de interes',Valx)
        select_valylist = st.sidebar.multiselect('Valores de Columna de respuesta',Valy)

        try:
         x = np.asarray(dataframe[select_valxlist])#.reshape(-1, 1)
         y = np.asarray(dataframe[select_valylist])
         st.write("Valores X")
         st.write(x)
         st.write("Valores Y")
         st.write(y)

         st.sidebar.subheader("Valores de Interes")
         vals = st.sidebar.text_input("Separar por coma")
         List_Val = lambda x : [int(i) for i in re.split(",",x) if i!=""]

         arraylist = List_Val(vals)
         valInit= list(map(float,arraylist))
        
         clf = DecisionTreeClassifier().fit(x,y)
         # plot_tree(clf,filled=True)
         # st.pyplot(plt)
         Treedot= export_graphviz(clf,out_file=None,filled=True,rounded=True,special_characters=True)
         st.graphviz_chart(Treedot)
        # Adaptación de datos
         
         st.write("Prediccion: ")
         st.write(clf.predict([valInit]))
         #st.write("==Predict result by predict_proba==")
         # print(clf.predict_proba([[-0.8, -1]]))
         #st.write("==Predict result by predict_log_proba==")
         # print(clf.predict_log_proba([[-0.8, -1]]))
         #vtree= tree.export_graphviz(clf,out_file=None)
         #st.#.graphviz_chart(vtree)
         
         #plt.savefig("arbolDecision.png")
         #st.image("./arbolDecision.png")
         #st.pyplot(plt)

         
         # create dataset for lightgbm
         # lgb_train = lgb.Dataset(x, y)

         # # specify your configurations as a dict
         # params = {
         #    'num_leaves': 5,
         #    'metric': ('l1', 'l2'),
         #    'verbose': 0
         # }

         # evals_result = {}  # to record eval results for plotting

         # # train
         # gbm = lgb.train(params,
         #                lgb_train,
         #                num_boost_round=100,
         #                # valid_sets=[lgb_train, lgb_test],
         #                # feature_name=['f' + str(i + 1) for i in range(X_train.shape[-1])],
         #                categorical_feature=[21],
         #                evals_result=evals_result,
         #                verbose_eval=10)

         # graph = lgb.create_tree_digraph(gbm, tree_index=53, name='Tree54')
         # st.graphviz_chart(graph)
        except Exception as e:
         print(e) 
        
        
        
   #      viz = dtreeviz(clf
   #          ,x,x
   #       )
   #      svg = viz.svg()
   #      b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")

   #  # Add some CSS on top
   #      #css_justify = "center" if center else "left"
   #      css = f'<p style="text-align:center; display: flex; ">'#justify-content: {css_justify};
   #      html = f'{css}<img src="data:image/svg+xml;base64,{b64}"/>'
   #      print(html)
   #  # Write the HTML
   #      st.write(html, unsafe_allow_html=True)


        ##########################################################################################################################
    
     if select_Algoritmo=='Redes neuronales.':
        st.subheader('Redes neuronales.')
        with st.sidebar.expander("Informacion Sobre el Algoritmo"):
          st.markdown("""
                      Que es?
                      * Una red neuronal es un método de la inteligencia artificial que enseña a las computadoras a procesar datos de una manera que está inspirada en la forma en que lo hace el cerebro humano.
                      
                      Para que sirve? 
                      * las redes neuronales artificiales intentan resolver problemas complicados, como la realización de resúmenes de documentos o el reconocimiento de rostros, con mayor precisión.
                      """)       
        try:
         select_valxlist = st.sidebar.multiselect('Valores de Columnas de interes',Valx)
         select_valylist = st.sidebar.multiselect('Valores de Columna de respuesta',Valy)

         x = np.asarray(dataframe[select_valxlist])#.reshape(-1, 1)
         y = np.asarray(dataframe[select_valylist])
         st.write("Valores X")
         st.write(x)
         st.write("Valores Y")
         st.write(y)

         st.sidebar.subheader("Valores de Interes")
         vals = st.sidebar.text_input("Separar por coma")
         List_Val = lambda x : [int(i) for i in re.split(",",x) if i!=""]
         arraylist = List_Val(vals)
         valInit= list(map(float,arraylist))
        
         clf = MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(5,2),random_state=1)
        # Adaptación de datos
         clf.fit(x,y)
         st.write("Prediccion: ")
         st.write(clf.predict([valInit]))
        #st.write("==Predict result by predict_proba==")
        # print(clf.predict_proba([[-0.8, -1]]))
        #st.write("==Predict result by predict_log_proba==")
        # print(clf.predict_log_proba([[-0.8, -1]]))
        ##########################################################################################################################
        except Exception as e:
            print(e) 
              
    # -- checñ box 
except Exception as e:
       print(e) 

# fs = 4096
# maxband = 2000
# high_fs = st.sidebar.checkbox('Mantener Ejecutando')
# if high_fs:
#    fs = 16384
#    maxband = 8000


# -- Notes on whitening
with st.expander("Sobre la aplicación"):
    st.markdown("""
 * Brindar una aplicación web que tenga capacidad ejecutar algoritmos de Machine Learning
      * Regresión lineal.
      * Regresión polinomial. 
      * Clasificador Gaussiano.
      * Clasificador de árboles de decisión. 
      * Redes neuronales.

 * [Ver Codigo](https://github.com/AlejooMariin/OLC2VJP2_201602855.git)
""")
with st.expander("Informacion Desarrollador"):
    st.markdown("""
    Universidad de San Carlos de Guatemala\n
    Ingenieria en Ciencias y Sistemas 
 * José Alejandro Grande Marín 
 * 201602855
""")




