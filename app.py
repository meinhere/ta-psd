# Data Processing
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

# Preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif

# Modelling
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from xgboost import XGBClassifier

# Evaluation
from sklearn.metrics import classification_report, roc_curve, auc, accuracy_score, confusion_matrix

# Visualization
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree

def main():
    st.sidebar.title("Navigasi")
    page = st.sidebar.radio("Pilih Halaman", ["Home", "Analysis", "Preprocessing", "Modeling", "Testing"])

    if page == "Home":
        show_home()
    elif page == "Analysis":
        show_analysis()
    elif page == "Preprocessing":
        show_preprocessing()
    elif page == "Modeling":
        show_model()
    elif page == "Testing":
        show_testing()

def show_home():
    st.title("Pengolahan Data Dampak Psikologi dengan Metode Random Forest, Adaboost, dan XGBoost")

    # Explain what is Random Forest
    st.header("Apa itu Random Forest?")
    st.write("Random Forest adalah salah satu algoritma yang terbaik dalam machine learning. Random Forest adalah kumpulan dari decision tree atau pohon keputusan. Algoritma ini merupakan kombinasi masing-masing tree dari decision tree yang kemudian digabungkan menjadi satu model. Biasanya, Random Forest dipakai untuk masalah regresi dan klasifikasi dengan kumpulan data yang berukuran besar.")

    # Explain what is Adaboost
    st.header("Apa itu Adaboost?")
    st.write("Adaboost adalah algoritma yang digunakan untuk meningkatkan performa dari model machine learning. Adaboost adalah singkatan dari Adaptive Boosting. Algoritma ini bekerja dengan cara menggabungkan beberapa model machine learning yang lemah menjadi satu model yang kuat. Adaboost bekerja dengan cara memberikan bobot yang berbeda pada setiap model machine learning yang lemah.")

    # Explain what is XGBoost
    st.header("Apa itu XGBoost?")
    st.write("XGBoost adalah algoritma yang digunakan untuk masalah regresi dan klasifikasi. XGBoost merupakan singkatan dari Extreme Gradient Boosting. Algoritma ini merupakan pengembangan dari algoritma Gradient Boosting. XGBoost bekerja dengan cara menggabungkan beberapa model machine learning yang lemah menjadi satu model yang kuat.")

    # Explain the purpose of this website
    st.header("Tujuan Website")
    st.write("Website ini bertujuan untuk memberikan pemahaman mengenai tahapan proses pengolahan data Dampak Psikologi dengan menggunakan metode Random Forest, Adaboost, dan XGBoost.")

    # Explain the data
    st.header("Data")
    st.write("Data yang digunakan diambil dari dataset public Psycological Effects of COVID.")

    # Explain the process of 
    st.header("Tahapan Proses")
    st.write("1. **Data Preparation**")
    st.write("2. **Preprocessing Data**")
    st.write("3. **Modeling**")
    st.write("4. **Evaluation**")
    st.write("5. **Deploy App**")

def show_analysis():
    st.title("Analysis Data")

    # --------------- Load Data -----------------
    df = pd.DataFrame(pd.read_csv("dataset/psyco.csv", delimiter=","))

    st.write("Data yang tersedia:")
    st.write(df.head())

    # --------------- Missing Value -----------------
    st.write("### Missing Value")
    st.write(df.isnull().sum())

    # --------------- Balance Data -----------------
    st.write("### Balance Data")
    fig, ax = plt.subplots()
    count = sns.countplot(x='prefer', data=df, ax=ax)
    plt.title('Count Plot of Preference')
    total = len(df)
    for p in count.patches:
      height = p.get_height()
      percentage = "{:.1f}%".format(100 * height / total)
      x = p.get_x() + p.get_width() / 2
      y = height + 10
      count.annotate(percentage, (x, y))
    st.pyplot(fig)

    # --------------- Correlation -----------------
    st.write("### Korelasi Fitur")
    corr = df.select_dtypes(include=['float', 'int']).corr()
    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(corr, annot=True)
    plt.title('Correlation Matrix')
    st.pyplot(fig)

    
def show_preprocessing():
    st.title("Preprocessing Data")

    # --------------- Load Data -----------------
    df = pd.read_csv("dataset/psyco.csv", delimiter=",")

    st.write("Data yang tersedia:")
    st.write(df.head())

    # --------------- Penghapusan Fitur -----------------
    st.write("### Drop Fitur")
    df.drop(['line_of_work', 'like_hw', 'dislike_hw', 'Unnamed: 19','time_bp.1','travel+work'],axis=1,inplace=True)

    st.write("Data Setelah Drop Fitur:")
    st.write(df)

    # --------------- Transformasi Data -----------------
    st.write("### Transformasi Data")
    object_columns = df.select_dtypes(include=['object']).columns.tolist()
    st.write("Kolom yang berisi data kategorikal:")
    for obj in object_columns:
        st.write(f"- {obj}")

    col1, col2 = st.columns(2)

    # --------------- Sebelum Transformasi -----------------
    with col1:
        st.write("#### Sebelum Transformasi")
        for col in object_columns:
            st.write(col, df[col].unique())

    # --------------- Setelah Transformasi -----------------
    with col2:
        st.write("#### Setelah Transformasi")

        le = LabelEncoder()
        
        for col in object_columns:
            df[col] = le.fit_transform(df[col])
            st.write(col, df[col].unique())

    # # --------------- Seleksi Fitur dengan Anova -----------------
    st.write("### Seleksi Fitur dengan Anova")
    X = df.drop('prefer',axis=1)
    y = df['prefer']

    selector = SelectKBest(score_func=f_classif, k=10)
    selector.fit_transform(X, y)
    feature_mask = selector.get_support()
    selected_features = X.columns[feature_mask]
    st.write("Fitur yang dipilih:")
    for feature in selected_features:
      st.write(f"- {feature}")

    st.write("Data Setelah Seleksi Fitur:")
    df = pd.concat([X[selected_features], y], axis=1)
    st.write(df.head())

    # # --------------- Oversampling Data -----------------
    st.write("### Oversampling Data")
    X = df.drop('prefer',axis=1)
    
    s=SMOTE(random_state=42)
    X_i,y_i=s.fit_resample(X,y)
    st.write("Jumlah Data setelah Oversampling:")
    st.write(y_i.value_counts())

    # ------------ Normalisasi Data -----------------
    st.write("### Normalisasi Data")

    col1, col2 = st.columns(2)

    # --------------- Sebelum Normalisasi -----------------
    with col1:
        st.write("#### Sebelum Normalisasi")
        st.write(X_i.head())

    # --------------- Setelah Normalisasi -----------------
    with col2:
        st.write("#### Setelah Normalisasi")

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_i)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        st.write(X_scaled.head())

    # --------------- Final data -----------------
    st.write("### Data Akhir")
    df = pd.concat([X_scaled, y_i], axis=1)
    st.write(df.head())
    st.write("Jumlah Baris: ", df.shape[0], "Jumlah Kolom: ", df.shape[1])
    st.session_state['preprocessed_data'] = df

def show_model():
    st.title("Modeling")
    
    if 'preprocessed_data' not in st.session_state:
        st.write("Silakan lakukan preprocessing data terlebih dahulu.")
        return
    
    # load data
    df = st.session_state['preprocessed_data']
    
    st.write("Data yang telah dipreproses:")
    st.write(df.head())

    # Memisahkan fitur dan label
    X = df.drop('prefer', axis=1)
    y = df['prefer']

    # Membagi data menjadi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1)

    st.write(X_train.shape,y_train.shape,X_test.shape,y_test.shape)

    # Membuat model Decision Tree dengan kriteria 'entropy' untuk C4.5
    rf=RandomForestClassifier(random_state=42)
    ab=AdaBoostClassifier(random_state=42)
    xgb=XGBClassifier(random_state=42)

    models=[rf,ab,xgb]
    for model in models:
        model_name = model.__class__.__name__
        st.subheader(f"Modeling dengan {model_name}")

        model.fit(X_train,y_train)
        
        y_pred=model.predict(X_test)

        ac = accuracy_score(y_pred, y_test)
        st.subheader("Accuracy")
        st.write(f"Accuracy Score: {ac:.2f}%")
        
        cr = classification_report(y_pred, y_test, output_dict=True)
        st.subheader("Classification Report")
        st.dataframe(cr)

    st.subheader("Final Model")
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    st.write(rf)

    st.subheader("Accuracy")
    ac = accuracy_score(y_test, y_pred)
    st.write(f"Accuracy Score: {ac:.2f}%")

    # Menampilkan Confusion Matrix sebagai tabel
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, 
                                    index=[f'Actual {i}' for i in range(len(cm))], 
                                    columns=[f'Predicted {i}' for i in range(len(cm))])
    st.subheader("Confusion Matrix")
    st.table(cm_df)

    # Menampilkan pohon keputusan
    st.subheader("Plot Random Forest")
    plt.figure(figsize=(20,10))
    plot_tree(rf.estimators_[0], feature_names=X.columns, filled=True, rounded=True)
    plt.title("Pohon Keputusan")
    plt.show()
    st.pyplot(plt)

def show_testing():
    features_encode = {
        "gender": {
            "Female": 0,
            "Male": 1,
            "Prefer not to say": 2
        },
        "certaindays_hw": {
            "Maybe": 0,
            "No": 1,
            "Yes": 2
        }
    }

    labels_encode = {
        0: "Complete Physical Attendance",
        1: "Work/study from home",
    }

    st.title("Prediksi Lokasi Anatomi TBC")

    gender = st.selectbox("Gender", ["Male", "Female", "Prefer not to say"])
    time_bp = st.number_input("Time Before Pandemic", min_value=1, max_value=10)
    easeof_online = st.number_input("Ease of Online Learning", min_value=1, max_value=5)
    home_env = st.number_input("Home Environment", min_value=1, max_value=5)
    prod_inc = st.selectbox("Productivity Increase", [-1, -0.5, 0, 0.5, 1])
    sleep_bal = st.selectbox("Sleep Balance", [-1, -0.5, 0, 0.5, 1])
    new_skill = st.selectbox("New Skill Earned", [-1, -0.5, 0, 0.5, 1])
    relaxed = st.selectbox("Relaxed", [-1, -0.5, 0, 0.5, 1])
    self_time = st.selectbox("Self Time", [-1, -0.5, 0, 0.5, 1])
    certaindays_hw = st.selectbox("Certain Days Homework", ["Yes", "No", "Maybe"])

    gender_encoded = features_encode["gender"][gender]
    certaindays_hw_encoded = features_encode["certaindays_hw"][certaindays_hw]

    # Create a feature vector
    input_data = pd.DataFrame({
        "gender": [gender_encoded],
        "time_bp": [time_bp],
        "easeof_online": [easeof_online],
        "home_env": [home_env],
        "prod_inc": [prod_inc],
        "sleep_bal": [sleep_bal],
        "new_skill": [new_skill],
        "relaxed": [relaxed],
        "self_time": [self_time],
        "certaindays_hw": [certaindays_hw_encoded]
    })

    check= st.button("Prediksi")

    # Normalisasi Data
    scaler_file = "models/minmaxscaler.sav"
    scaler = pickle.load(open(scaler_file, "rb"))
    input_scaled = scaler.transform(input_data)

    # Predict Data
    model_file = "models/rf_model.pkl"
    model = pickle.load(open(model_file,'rb'))
    prediction = model.predict(input_scaled)

    if  check:
        st.write(f"##### Hasil prediksi : :red[{labels_encode[int(prediction[0])]}]")

if __name__ == "__main__":
    st.set_page_config(page_title="Phsycology Effect COVID-19", page_icon="💗")
    main()