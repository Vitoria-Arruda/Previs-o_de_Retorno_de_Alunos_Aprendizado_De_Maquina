import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
import warnings
import os

st.set_page_config(layout="wide", page_title="PI1: Previsão de Retorno (Análise Final)")
warnings.filterwarnings('ignore')
sns.set(style="whitegrid")

@st.cache_data
def load_data():
    """
    Busca automaticamente por arquivos de dados na pasta local.
    Retorna os dados e uma lista de mensagens para serem exibidas na interface.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    arquivos_tentativa = [
        'base_alunos_trancamento.csv', 
        'base_alunos_trancamento.xlsx',
        'base_alunos_trancamento.xlsx - Sheet1.csv'
    ]
    
    df = None
    arquivo_carregado = None
    caminho_final = None
    messages = [] 

    for nome in arquivos_tentativa:
        caminho = os.path.join(script_dir, nome)
        if os.path.exists(caminho):
            caminho_final = caminho
            arquivo_carregado = nome
            break
    
    if caminho_final is None:
        try:
            arquivos_na_pasta = os.listdir(script_dir)
            for f in arquivos_na_pasta:
                if ('alunos' in f.lower() or 'trancamento' in f.lower()) and (f.endswith('.csv') or f.endswith('.xlsx')):
                    caminho_final = os.path.join(script_dir, f)
                    arquivo_carregado = f
                    break
        except Exception:
            pass

    if caminho_final:
        try:
            if caminho_final.endswith('.xlsx'):
                df = pd.read_excel(caminho_final)
            else:
                df = pd.read_csv(caminho_final)
            
            messages.append(("toast", f"Base de dados carregada: {arquivo_carregado}"))
        except Exception as e:
            messages.append(("error", f"Erro ao ler o arquivo {arquivo_carregado}: {e}"))

    if df is not None:
        required_cols = ['media_anterior', 'semestre_trancamento', 'divida_pendente', 'idade', 'campus_origem']
        df.columns = [str(c).lower().strip() for c in df.columns]
        
        if not all(col in df.columns for col in required_cols):
            messages.append(("error", f"Arquivo incompleto. Colunas esperadas: {required_cols}"))
            df = None
        else:
            if 'retornou' not in df.columns:
                messages.append(("warning", "Coluna alvo 'retornou' não encontrada. Gerando dados sintéticos para demonstração."))
                df['retornou'] = np.random.randint(0, 2, df.shape[0])

    # --- GERAÇÃO DE DADOS FICTÍCIOS ---
    if df is None:
        messages.append(("warning", f"Base de dados não encontrada na pasta '{script_dir}'. Usando DADOS FICTÍCIOS."))
        np.random.seed(42)
        num_alunos = 1000
        data = {
            'id_aluno': range(1, num_alunos + 1),
            'media_anterior': np.random.uniform(4.0, 10.0, num_alunos),
            'semestre_trancamento': np.random.randint(1, 11, num_alunos),
            'divida_pendente': np.random.randint(0, 2, num_alunos),
            'idade': np.random.randint(18, 45, num_alunos),
            'campus_origem': np.random.choice(['Capital', 'Interior', 'EAD', np.nan], num_alunos, p=[0.4, 0.3, 0.25, 0.05])
        }
        df = pd.DataFrame(data)
        prob_retorno = ((df['media_anterior']**2/100) - (df['divida_pendente']*0.4) + ((df['idade']<25)&(df['divida_pendente']==0))*0.2) + np.random.normal(0,0.05,num_alunos)
        df['retornou'] = (prob_retorno > np.median(prob_retorno)).astype(int)

    if 'campus_origem' in df.columns:
        if df['campus_origem'].mode().empty:
             df['campus_origem'].fillna("Desconhecido", inplace=True)
        else:
            df['campus_origem'].fillna(df['campus_origem'].mode()[0], inplace=True)
    
    df_model = pd.get_dummies(df, columns=['campus_origem'], drop_first=True)
    df_model.columns = [c.lower().strip() for c in df_model.columns]

    for col in ['campus_origem_ead', 'campus_origem_interior']:
        if col not in df_model.columns:
            df_model[col] = 0

    return df, df_model, messages

@st.cache_resource
def train_models(df_model):
    if df_model is None: return None

    features = ['media_anterior', 'semestre_trancamento', 'divida_pendente', 'idade', 'campus_origem_ead', 'campus_origem_interior']
    
    if not all(col in df_model.columns for col in features):
        st.error("Erro de colunas no treinamento.")
        return None

    X = df_model[features]
    y = df_model['retornou']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Modelos
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    
    gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    gb_model.fit(X_train_scaled, y_train)

    preds = {
        'rf_pred': rf_model.predict(X_test_scaled),
        'rf_proba': rf_model.predict_proba(X_test_scaled)[:, 1],
        'gb_pred': gb_model.predict(X_test_scaled),
        'gb_proba': gb_model.predict_proba(X_test_scaled)[:, 1]
    }

    return {
        'rf_model': rf_model,
        'gb_model': gb_model,
        'scaler': scaler,
        'features': features,
        'test_data': (X_test_scaled, y_test),
        'predictions': preds
    }

st.title("🎓 PI1: Projeto Individual - Mineração de Dados")

with st.container():
    st.markdown("### Problema:")
    st.markdown("""
    Este projeto tem como objetivo aplicar técnicas de Aprendizado de Máquina Supervisionado para mitigar o problema da evasão no ensino superior.
  
    O trancamento de matrículas e a evasão representam um desafio crítico para as instituições de ensino. 
    * Perda direta de receita recorrente (mensalidades) e aumento do Custo de Aquisição de Clientes (CAC) para repor a vaga.
    * Descontinuidade do aprendizado e redução das taxas de formação da instituição.

    O sistema preditivo é capaz de calcular a Probabilidade de Retorno de um aluno que trancou o curso. 
    Ao analisar padrões históricos (como desempenho acadêmico anterior, situação financeira e idade), a instituição pode classificar os alunos em grupos de risco e direcionar esforços de retenção de forma mais eficiente, focando naqueles com maior chance de recuperação.
    """)

df_display, df_model, log_messages = load_data()

if log_messages:
    for msg_type, msg_text in log_messages:
        if msg_type == 'toast':
            st.toast(msg_text, icon="✅")
        elif msg_type == 'warning':
            st.warning(msg_text)
        elif msg_type == 'error':
            st.error(msg_text)

if df_model is not None:
    models = train_models(df_model)
    
    if models is not None:
        scaler = models['scaler']
        rf_model = models['rf_model']
        gb_model = models['gb_model']
        features_list = models['features']
        X_test_scaled, y_test = models['test_data']
        preds = models['predictions']

        st.sidebar.header("🔍 Filtros de Simulação")
        st.sidebar.info("Defina o perfil do aluno abaixo:")
        
        s_media = st.sidebar.slider("Média Anterior", 0.0, 10.0, 7.5, 0.1)
        s_semestre = st.sidebar.slider("Semestre Trancamento", 1, 10, 3)
        s_divida = st.sidebar.selectbox("Possui Dívida?", ["Não", "Sim"])
        s_idade = st.sidebar.number_input("Idade", 17, 60, 22)
        s_campus = st.sidebar.selectbox("Campus", ["Capital", "Interior", "EAD"])

        input_data = pd.DataFrame({
            'media_anterior': [s_media],
            'semestre_trancamento': [s_semestre],
            'divida_pendente': [1 if s_divida == "Sim" else 0],
            'idade': [s_idade],
            'campus_origem_ead': [1 if s_campus == "EAD" else 0],
            'campus_origem_interior': [1 if s_campus == "Interior" else 0]
        })
        input_scaled = scaler.transform(input_data)
        
        rf_prob = rf_model.predict_proba(input_scaled)[0][1]
        gb_prob = gb_model.predict_proba(input_scaled)[0][1]

        st.markdown("### Análise Exploratória e Visualização (ETL)")
        with st.expander("📊 Visualizar Dados e Gráficos", expanded=True):
            st.dataframe(df_display, use_container_width=True, height=250)
            
            fig1, ax1 = plt.subplots(1, 2, figsize=(14, 5))
            sns.boxplot(x='retornou', y='media_anterior', data=df_display, ax=ax1[0], palette='viridis')
            ax1[0].set_title('Relação: Média Anterior vs. Retorno')
            
            sns.countplot(x='divida_pendente', hue='retornou', data=df_display, ax=ax1[1], palette='magma')
            ax1[1].set_title('Relação: Dívida Pendente vs. Retorno')
            st.pyplot(fig1)

        st.markdown("---")
        st.markdown("### Probabilidade de Retorno (Simulação)")
        st.markdown(f"Com base no perfil selecionado (**Idade:** {s_idade}, **Média:** {s_media}, **Dívida:** {s_divida}, **Campus:** {s_campus}):")

        col_sim1, col_sim2, col_sim3 = st.columns([1, 1, 2])
        
        with col_sim1:
            st.metric("Random Forest", f"{rf_prob*100:.1f}%", delta_color="normal")
        
        with col_sim2:
            st.metric("Gradient Boosting", f"{gb_prob*100:.1f}%", delta_color="normal")
            
        with col_sim3:
            if gb_prob > 0.7:
                st.success("✅ **Alta Chance de Retorno:** Este aluno tem fortes indícios de que voltará aos estudos.")
            elif gb_prob > 0.4:
                st.warning("⚠️ **Chance Moderada de Retorno:** O aluno está indeciso. Ações de engajamento são necessárias.")
            else:
                st.error("🚨 **Baixa Chance de Retorno:** Retorno improvável com o perfil atual.")

        st.markdown("---")
        st.markdown("### Avaliação dos Modelos (Dados de Teste)")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🌲 Random Forest")
            st.metric("Acurácia Global", f"{accuracy_score(y_test, preds['rf_pred']):.2%}")
            st.text("Matriz de Confusão:")
            st.write(pd.DataFrame(confusion_matrix(y_test, preds['rf_pred']), columns=['Pred: Não', 'Pred: Sim'], index=['Real: Não', 'Real: Sim']))
            st.caption("Importância das Variáveis:")
            st.bar_chart(pd.Series(rf_model.feature_importances_, index=features_list))

        with col2:
            st.subheader("⚡ Gradient Boosting")
            st.metric("Acurácia Global", f"{accuracy_score(y_test, preds['gb_pred']):.2%}")
            st.text("Matriz de Confusão:")
            st.write(pd.DataFrame(confusion_matrix(y_test, preds['gb_pred']), columns=['Pred: Não', 'Pred: Sim'], index=['Real: Não', 'Real: Sim']))
            st.caption("Importância das Variáveis:")
            st.bar_chart(pd.Series(gb_model.feature_importances_, index=features_list))

        st.markdown("---")
        st.markdown("### Conclusão")
        
        melhor_modelo = "Gradient Boosting" if accuracy_score(y_test, preds['gb_pred']) >= accuracy_score(y_test, preds['rf_pred']) else "Random Forest"
        acc_melhor = max(accuracy_score(y_test, preds['gb_pred']), accuracy_score(y_test, preds['rf_pred']))

        st.info(f"""
        **Análise e Interpretação dos Resultados:**

        1.  **Escolha do Modelo:** Neste estudo comparativo, o algoritmo **{melhor_modelo}** demonstrou desempenho superior (ou equivalente), com uma acurácia de **{acc_melhor:.2%}** nos dados de teste. Isso indica que o modelo consegue generalizar bem os padrões aprendidos para novos alunos.

        2.  **Interpretação das Variáveis (Feature Importance):**
            Ao analisar os gráficos de importância acima, observamos que variáveis como **'media_anterior'** e **'divida_pendente'** são consistentemente as mais relevantes. 
            * Isso sugere que o fator financeiro (dívida) é um bloqueador crítico para o retorno.
            * O desempenho acadêmico prévio (média) atua como um motivador: alunos com notas melhores tendem a valorizar mais o curso interrompido.

        3.  **Conclusão de Negócio:**
            A utilização deste modelo permite que a instituição saia de uma abordagem reativa para uma proativa. Ao identificar que um aluno com dívida e média baixa tem probabilidade de retorno inferior a 30% (como visto no simulador), a instituição pode evitar gastos de marketing ineficientes e focar recursos em alunos com probabilidade mediana (40-70%), onde uma intervenção (ex: desconto na dívida) pode efetivamente mudar a decisão do aluno.
        """)