"""
app.py — AgriPredict Burundi
Application Streamlit complète — TP IA Appliquée — Bac4 Génie Logiciel
Université Polytechnique de Gitega
"""

import os, warnings, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import streamlit as st
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, roc_auc_score, roc_curve
)

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════
# CONFIG PAGE
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AgriPredict Burundi",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

:root {
  --green-dark:  #1a3a2a;
  --green-mid:   #2d6a4f;
  --green-light: #52b788;
  --green-soft:  #b7e4c7;
  --gold:        #f4a261;
  --gold-light:  #ffd166;
  --cream:       #faf7f0;
  --brown:       #7c4f2a;
  --text-dark:   #1c1c1c;
  --text-muted:  #6b7280;
  --card-bg:     #ffffff;
  --border:      #e8e0d4;
  --shadow:      0 4px 24px rgba(26,58,42,0.10);
  --red-dark:    #7c1d1d;
  --red-mid:     #b91c1c;
}

html, body, [data-testid="stAppViewContainer"] {
  background: var(--cream) !important;
  font-family: 'DM Sans', sans-serif;
  color: var(--text-dark);
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: var(--green-dark) !important;
}
[data-testid="stSidebar"] * { color: #e8f5e9 !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.15) !important; }
[data-testid="stSidebar"] .stRadio > label { color: var(--green-soft) !important; font-size: 0.82rem !important; }

/* ── Header banner ── */
.app-header {
  background: linear-gradient(135deg, var(--green-dark) 0%, var(--green-mid) 60%, var(--green-light) 100%);
  border-radius: 16px;
  padding: 2.2rem 2.5rem 1.8rem;
  margin-bottom: 1.6rem;
  position: relative;
  overflow: hidden;
}
.app-header::before {
  content: '';
  position: absolute; top: -40px; right: -40px;
  width: 200px; height: 200px;
  border-radius: 50%;
  background: rgba(255,255,255,0.06);
}
.app-header::after {
  content: '';
  position: absolute; bottom: -30px; left: 30%;
  width: 120px; height: 120px;
  border-radius: 50%;
  background: rgba(255,255,255,0.04);
}
.app-header h1 {
  font-family: 'Syne', sans-serif !important;
  font-size: 2.1rem !important;
  font-weight: 800 !important;
  color: #ffffff !important;
  margin: 0 0 0.4rem !important;
  line-height: 1.15 !important;
}
.app-header p {
  color: var(--green-soft) !important;
  font-size: 0.95rem !important;
  margin: 0 !important;
  font-weight: 300;
}

/* ── Cards ── */
.card {
  background: var(--card-bg);
  border-radius: 14px;
  border: 1px solid var(--border);
  padding: 1.4rem 1.6rem;
  margin-bottom: 1.1rem;
  box-shadow: var(--shadow);
}
.card-title {
  font-family: 'Syne', sans-serif;
  font-size: 1.02rem;
  font-weight: 700;
  color: var(--green-dark);
  margin-bottom: 0.75rem;
}
.card ul { margin: 0.4rem 0 0 1.2rem; }
.card li { margin-bottom: 0.3rem; font-size: 0.93rem; }
.card p  { margin: 0.3rem 0; font-size: 0.93rem; }

/* ── Section title ── */
.section-title {
  font-family: 'Syne', sans-serif;
  font-size: 1.25rem;
  font-weight: 700;
  color: var(--green-dark);
  border-left: 4px solid var(--green-light);
  padding-left: 0.75rem;
  margin: 0.4rem 0 1.2rem;
}

/* ── Prediction badges ── */
.pred-good {
  background: linear-gradient(135deg, #1b4332, #2d6a4f);
  color: white;
  border-radius: 14px;
  padding: 1.4rem 2rem;
  font-family: 'Syne', sans-serif;
  font-size: 1.5rem;
  font-weight: 800;
  text-align: center;
  box-shadow: 0 6px 20px rgba(45,106,79,0.35);
  margin: 1rem 0;
}
.pred-bad {
  background: linear-gradient(135deg, #7c1d1d, #b91c1c);
  color: white;
  border-radius: 14px;
  padding: 1.4rem 2rem;
  font-family: 'Syne', sans-serif;
  font-size: 1.5rem;
  font-weight: 800;
  text-align: center;
  box-shadow: 0 6px 20px rgba(185,28,28,0.35);
  margin: 1rem 0;
}
.pred-sub { font-size: 1rem; font-weight: 400; opacity: 0.85; display: block; margin-top: 0.3rem; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  background: var(--green-dark) !important;
  border-radius: 10px !important;
  padding: 4px !important;
  gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
  color: var(--green-soft) !important;
  font-family: 'Syne', sans-serif !important;
  font-weight: 600 !important;
  border-radius: 7px !important;
  font-size: 0.88rem !important;
}
.stTabs [aria-selected="true"] {
  background: var(--green-light) !important;
  color: var(--green-dark) !important;
}

/* ── Buttons ── */
.stButton > button {
  background: linear-gradient(135deg, var(--green-mid), var(--green-light)) !important;
  color: white !important;
  border: none !important;
  border-radius: 10px !important;
  font-family: 'Syne', sans-serif !important;
  font-weight: 700 !important;
  font-size: 1rem !important;
  padding: 0.65rem 2rem !important;
  box-shadow: 0 4px 12px rgba(45,106,79,0.35) !important;
  transition: all 0.2s !important;
}
.stButton > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 20px rgba(45,106,79,0.45) !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
  background: white;
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 0.8rem 1rem;
  box-shadow: var(--shadow);
}
[data-testid="stMetricLabel"] { font-size: 0.8rem !important; color: var(--text-muted) !important; }
[data-testid="stMetricValue"] { font-family: 'Syne', sans-serif !important; color: var(--green-dark) !important; font-size: 1.4rem !important; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# CONSTANTES
# ═══════════════════════════════════════════════════════════════════
FEATURES_CAT = ['saison', 'province', 'culture']
FEATURES_NUM = ['annee', 'altitude_m', 'pluviometrie_mm', 'temperature_moy_C',
                'superficie_ha', 'utilisation_engrais', 'acces_irrigation', 'nb_menages']
TARGET = 'bonne_recolte'

PROVINCES = ['Bubanza','Bujumbura Rural','Bururi','Cankuzo','Cibitoke',
             'Gitega','Kayanza','Kirundo','Makamba','Muramvya',
             'Muyinga','Mwaro','Ngozi','Rutana','Ruyigi']
CULTURES  = ['Bananier','Haricot','Manioc','Maïs','Patate douce','Sorgho']

MODEL_LABELS = {
    'decision_tree':       'Arbre de Décision',
    'random_forest':       'Forêt Aléatoire',
    'logistic_regression': 'Régression Logistique',
}
COLORS = {
    'decision_tree':       '#2d6a4f',
    'random_forest':       '#52b788',
    'logistic_regression': '#f4a261',
}

FIG_BG = '#faf7f0'

def fig_style(fig, ax_list=None):
    """Apply consistent background to figure and axes."""
    fig.patch.set_facecolor(FIG_BG)
    axes = ax_list if ax_list else (fig.axes if hasattr(fig, 'axes') else [])
    for ax in axes:
        ax.set_facecolor(FIG_BG)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    return fig

# ═══════════════════════════════════════════════════════════════════
# CHARGEMENT DONNÉES & MODÈLES
# ═══════════════════════════════════════════════════════════════════
@st.cache_data
def load_raw():
    return pd.read_csv('agriculture_burundi.csv')

@st.cache_data
def prepare_data(raw: pd.DataFrame):
    df = raw.copy()
    df.dropna(subset=[TARGET], inplace=True)
    for col in ['pluviometrie_mm','temperature_moy_C','altitude_m',
                'superficie_ha','nb_menages','rendement_t_ha','production_totale_t',
                'utilisation_engrais']:
        df[col] = df.groupby('culture')[col].transform(lambda x: x.fillna(x.median()))
    df.fillna(df.median(numeric_only=True), inplace=True)
    return df

@st.cache_resource
def build_preprocessor():
    return ColumnTransformer([
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), FEATURES_CAT),
        ('num', StandardScaler(), FEATURES_NUM),
    ], remainder='drop')

@st.cache_resource
def train_all_models(_df):
    X = _df[FEATURES_CAT + FEATURES_NUM]
    y = _df[TARGET].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    prep = build_preprocessor()
    specs = {
        'decision_tree':       DecisionTreeClassifier(max_depth=4, criterion='gini', random_state=42),
        'random_forest':       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
    }

    models, results = {}, {}
    for name, clf in specs.items():
        pipe = Pipeline([('prep', build_preprocessor()), ('clf', clf)])
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)
        y_prob = pipe.predict_proba(X_te)[:, 1]
        models[name] = pipe
        results[name] = {
            'accuracy': accuracy_score(y_te, y_pred),
            'report':   classification_report(y_te, y_pred, output_dict=True),
            'cm':       confusion_matrix(y_te, y_pred),
            'auc':      roc_auc_score(y_te, y_prob),
            'roc':      roc_curve(y_te, y_prob),
            'cv':       cross_val_score(pipe, X, y, cv=5, scoring='accuracy'),
        }
    return models, results, X_tr, X_te, y_tr, y_te

# ─── charge & entraîne ───
df_raw = load_raw()
df     = prepare_data(df_raw)
models, results, X_tr, X_te, y_tr, y_te = train_all_models(df)

# ═══════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🌱 AgriPredict")
    st.markdown("**Université Polytechnique de Gitega**  \nBac4 — Génie Logiciel")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        [
            "🏠  Accueil",
            "📊  Exploration des Données",
            "⚙️  Prétraitement",
            "🌳  Arbre de Décision",
            "🌲  Forêt Aléatoire",
            "📈  Régression Logistique",
            "📉  Comparaison ROC",
            "🔮  Prédiction Scénarios",
            "🎯  Prédiction Personnalisée",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("TP IA Appliquée — 2025/2026")

# ═══════════════════════════════════════════════════════════════════
# PAGE : ACCUEIL
# ═══════════════════════════════════════════════════════════════════
if page == "🏠  Accueil":
    st.markdown("""
    <div class="app-header">
      <h1>🌱 AgriPredict Burundi</h1>
      <p>Prédiction des bonnes et mauvaises récoltes par Machine Learning · TP Bac4 Génie Logiciel</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📋 Observations",      f"{len(df):,}")
    c2.metric("🗺️ Provinces",         df['province'].nunique())
    c3.metric("🌾 Cultures",           df['culture'].nunique())
    c4.metric("✅ Bonnes récoltes",    f"{df[TARGET].mean()*100:.1f}%")

    st.markdown("---")
    st.markdown("""
    <div class="card">
      <div class="card-title">📚 À propos de cette application</div>
      <p>Ce projet couvre le pipeline ML complet appliqué à l'agriculture burundaise :</p>
      <ul>
        <li><b>Ex. 1</b> — Exploration, statistiques et visualisations</li>
        <li><b>Ex. 2</b> — Prétraitement, encodage, normalisation, split</li>
        <li><b>Ex. 3</b> — Arbre de Décision + analyse overfitting</li>
        <li><b>Ex. 4</b> — Forêt Aléatoire + validation croisée</li>
        <li><b>Ex. 5</b> — Régression Logistique + courbe ROC</li>
        <li><b>Ex. 6</b> — Prédiction sur 4 scénarios réels du Burundi</li>
        <li><b>Ex. 7</b> — Prédiction personnalisée interactive</li>
      </ul>
    </div>
    """, unsafe_allow_html=True)

    st.success(f"✅ 3 modèles entraînés et prêts. Dataset : {len(df_raw)} lignes × {df_raw.shape[1]} colonnes.")

# ═══════════════════════════════════════════════════════════════════
# PAGE : EXPLORATION (Ex. 1)
# ═══════════════════════════════════════════════════════════════════
elif page == "📊  Exploration des Données":
    st.markdown('<div class="section-title">📊 Exercice 1 — Exploration et Qualité des Données</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Aperçu & Types", "❓ Valeurs manquantes", "📈 Statistiques", "📉 Visualisations"])

    # ── Tab 1 : Aperçu ──
    with tab1:
        st.markdown('<div class="card-title">Q1 — Structure du dataset</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Lignes",    df_raw.shape[0])
        c2.metric("Colonnes",  df_raw.shape[1])
        c3.metric("Années",    f"{int(df_raw['annee'].min())}–{int(df_raw['annee'].max())}")
        c4.metric("Provinces", df_raw['province'].nunique())

        st.dataframe(df_raw.head(10), width='stretch')

        st.markdown(f"**Provinces :** {', '.join(sorted(df_raw['province'].unique()))}")
        st.markdown(f"**Cultures :** {', '.join(sorted(df_raw['culture'].unique()))}")

        st.markdown('<div class="card-title" style="margin-top:1rem">Q2 — Types de données</div>', unsafe_allow_html=True)
        dtype_df = pd.DataFrame({
            'Colonne': df_raw.columns,
            'Type':    df_raw.dtypes.astype(str).values,
            'Exemple': df_raw.iloc[0].astype(str).values,
        })
        st.dataframe(dtype_df, width='stretch', hide_index=True)
        st.info("Pas de types incohérents majeurs. `utilisation_engrais` et `acces_irrigation` sont int (binaires) — cohérent. `bonne_recolte` est float car il contient des NaN (sera converti en int après nettoyage).")

    # ── Tab 2 : Valeurs manquantes ──
    with tab2:
        st.markdown('<div class="card-title">Q3 — Valeurs manquantes par colonne</div>', unsafe_allow_html=True)
        miss = pd.DataFrame({
            'Colonne':       df_raw.columns,
            'Manquants':     df_raw.isnull().sum().values,
            '% manquants':   (df_raw.isnull().mean() * 100).round(2).values,
        }).query('Manquants > 0').sort_values('% manquants', ascending=False)

        st.dataframe(miss, width='stretch', hide_index=True)

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.barh(miss['Colonne'], miss['% manquants'], color='#2d6a4f')
        ax.set_xlabel("% de valeurs manquantes")
        ax.set_title("Taux de valeurs manquantes par colonne")
        fig_style(fig, [ax])
        st.pyplot(fig)
        plt.close()

        st.markdown('<div class="card-title" style="margin-top:1rem">Q4 — Stratégie de traitement</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
          <ul>
            <li><b>bonne_recolte</b> (cible, 2.7% NaN) → <b>suppression des lignes</b> : on ne peut pas imputer la variable à prédire.</li>
            <li><b>pluviometrie_mm</b> (3.9%) → <b>imputation par médiane groupée par culture</b> : la pluie dépend de la zone géographique liée à la culture.</li>
            <li><b>utilisation_engrais</b> (2.8%) → <b>imputation par médiane groupée par culture</b>.</li>
            <li><b>rendement_t_ha / production_totale_t</b> → imputés mais exclus des features (data leakage).</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)
        st.success(f"✅ Après nettoyage : {len(df)} lignes, 0 valeur manquante.")

    # ── Tab 3 : Statistiques ──
    with tab3:
        st.markdown('<div class="card-title">Q5 — Statistiques descriptives</div>', unsafe_allow_html=True)
        desc = df.describe().round(3)
        st.dataframe(desc.astype(str), width='stretch')

        col1, col2 = st.columns(2)
        with col1:
            rend = df.groupby('culture')['rendement_t_ha'].mean().sort_values(ascending=False).round(2)
            st.markdown("**Rendement moyen par culture (t/ha)**")
            fig, ax = plt.subplots(figsize=(6, 3.5))
            colors_bar = ['#1b4332','#2d6a4f','#52b788','#95d5b2','#f4a261','#ffd166']
            rend.plot(kind='barh', ax=ax, color=colors_bar)
            ax.set_xlabel("t/ha")
            fig_style(fig, [ax])
            st.pyplot(fig); plt.close()

        with col2:
            pluv = df.groupby('province')['pluviometrie_mm'].mean().sort_values(ascending=False).round(1)
            st.markdown("**Pluviométrie moyenne par province (mm)**")
            fig, ax = plt.subplots(figsize=(6, 5))
            pluv.plot(kind='barh', ax=ax, color='#52b788')
            ax.set_xlabel("mm")
            fig_style(fig, [ax])
            st.pyplot(fig); plt.close()

        st.markdown('<div class="card-title" style="margin-top:1rem">Q6 — Distribution de la variable cible</div>', unsafe_allow_html=True)
        vc = df[TARGET].value_counts(normalize=True) * 100
        c1, c2 = st.columns(2)
        c1.metric("✅ Bonnes récoltes (1)", f"{vc.get(1.0, 0):.1f}%")
        c2.metric("❌ Mauvaises récoltes (0)", f"{vc.get(0.0, 0):.1f}%")
        diff = abs(vc.get(1.0, 50) - 50)
        if diff < 15:
            st.success("Dataset relativement équilibré — pas de rééchantillonnage critique nécessaire.")
        else:
            st.warning("Déséquilibre de classes → risque de biais vers la classe majoritaire. Considérer `class_weight='balanced'`.")

    # ── Tab 4 : Visualisations ──
    with tab4:
        st.markdown('<div class="card-title">Q7 — 4 Visualisations commentées</div>', unsafe_allow_html=True)

        # 1. Boxplot rendement par culture
        fig, ax = plt.subplots(figsize=(10, 4))
        cultures_order = df.groupby('culture')['rendement_t_ha'].median().sort_values().index.tolist()
        data_bp = [df[df['culture'] == c]['rendement_t_ha'].dropna().values for c in cultures_order]
        bp = ax.boxplot(data_bp, labels=cultures_order, patch_artist=True,
                        boxprops=dict(facecolor='#b7e4c7', color='#2d6a4f'),
                        medianprops=dict(color='#f4a261', linewidth=2.5),
                        whiskerprops=dict(color='#2d6a4f'),
                        capprops=dict(color='#2d6a4f'),
                        flierprops=dict(marker='o', color='#f4a261', alpha=0.5))
        ax.set_title("Distribution du rendement par culture (t/ha)", fontsize=13, pad=10)
        ax.set_xlabel("Culture"); ax.set_ylabel("Rendement (t/ha)")
        fig_style(fig, [ax]); st.pyplot(fig); plt.close()
        st.caption("→ Le Manioc et le Bananier ont les rendements les plus élevés. Le Haricot et le Sorgho présentent des rendements faibles avec une forte variabilité.")

        st.markdown("---")

        # 2. Lineplot production totale par année
        fig, ax = plt.subplots(figsize=(10, 4))
        prod_yr = df.groupby('annee')['production_totale_t'].sum() / 1000
        ax.plot(prod_yr.index, prod_yr.values, marker='o', color='#2d6a4f',
                linewidth=2.5, markersize=8, zorder=3)
        ax.fill_between(prod_yr.index, prod_yr.values, alpha=0.15, color='#52b788')
        ax.set_title("Production totale par année (milliers de tonnes)", fontsize=13, pad=10)
        ax.set_xlabel("Année"); ax.set_ylabel("Production (×1 000 t)")
        ax.grid(axis='y', alpha=0.25); ax.set_xticks(prod_yr.index)
        fig_style(fig, [ax]); st.pyplot(fig); plt.close()
        st.caption("→ La production fluctue selon les années, reflétant la variabilité climatique et des aléas agricoles saisonniers.")

        st.markdown("---")

        # 3. Barplot bonnes récoltes vs engrais
        fig, ax = plt.subplots(figsize=(7, 4))
        eng_stats = df.groupby('utilisation_engrais')[TARGET].mean() * 100
        bars = ax.bar(['Sans engrais', 'Avec engrais'],
                      [eng_stats.get(0, 0), eng_stats.get(1, 0)],
                      color=['#b7e4c7', '#2d6a4f'], edgecolor='white', width=0.45)
        ax.bar_label(bars, fmt='%.1f%%', padding=5, fontsize=13, fontweight='bold', color='#1a3a2a')
        ax.set_title("% de bonnes récoltes selon l'utilisation d'engrais", fontsize=13, pad=10)
        ax.set_ylabel("% bonnes récoltes"); ax.set_ylim(0, 110)
        fig_style(fig, [ax]); st.pyplot(fig); plt.close()
        st.caption("→ L'utilisation d'engrais améliore significativement la probabilité d'une bonne récolte — message fort pour les décideurs agricoles.")

        st.markdown("---")

        # 4. Heatmap corrélation
        num_cols = ['altitude_m','pluviometrie_mm','temperature_moy_C','superficie_ha',
                    'utilisation_engrais','acces_irrigation','nb_menages','rendement_t_ha',TARGET]
        corr = df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(corr, annot=True, fmt='.2f', ax=ax,
                    cmap=sns.diverging_palette(145, 300, s=60, as_cmap=True),
                    linewidths=0.5, square=True, annot_kws={'size': 9})
        ax.set_title("Matrice de corrélation — variables numériques", fontsize=13, pad=10)
        fig_style(fig)
        st.pyplot(fig); plt.close()
        st.caption("→ Le rendement est très corrélé à bonne_recolte (normal — c'est sa définition). La pluviométrie et l'engrais montrent des corrélations positives modérées avec la bonne récolte.")

# ═══════════════════════════════════════════════════════════════════
# PAGE : PRÉTRAITEMENT (Ex. 2)
# ═══════════════════════════════════════════════════════════════════
elif page == "⚙️  Prétraitement":
    st.markdown('<div class="section-title">⚙️ Exercice 2 — Prétraitement et Préparation</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
      <div class="card-title">Q8 — Variables catégorielles & encodage</div>
      <p><b>Variables catégorielles identifiées :</b> <code>saison</code>, <code>province</code>, <code>culture</code></p>
      <p>Les algorithmes ML exigent des entrées numériques. On ne peut pas passer des chaînes directement.</p>
      <ul>
        <li><b>LabelEncoder</b> : encode en entiers (0, 1, 2…) → introduit un ordre artificiel non souhaité pour des variables nominales.</li>
        <li><b>OneHotEncoder / get_dummies</b> : crée une colonne binaire par catégorie → adapté aux variables sans ordre naturel.</li>
        <li><b>saison</b> (2 valeurs : A/B) → OneHotEncoder (drop='first') suffit.</li>
        <li><b>province</b> (15 valeurs) → OneHotEncoder (drop='first') pour éviter la dummy variable trap.</li>
        <li><b>culture</b> (6 valeurs) → OneHotEncoder (drop='first').</li>
      </ul>
    </div>
    """, unsafe_allow_html=True)

    enc_demo = pd.get_dummies(df[['saison','province','culture']], drop_first=True)
    st.markdown(f"**Q9 — Après encodage :** {enc_demo.shape[1]} colonnes issues des variables catégorielles (sur {len(FEATURES_CAT + FEATURES_NUM) + enc_demo.shape[1]} au total).")
    st.dataframe(enc_demo.head(3).astype(str), width='stretch')

    st.markdown("""
    <div class="card">
      <div class="card-title">Q10 — Features X et cible y — Data leakage</div>
      <p><b>Colonnes exclues :</b> <code>rendement_t_ha</code> et <code>production_totale_t</code></p>
      <p><b>Pourquoi ?</b> <code>bonne_recolte</code> est définie directement depuis <code>rendement_t_ha</code>
      (bonne si rendement > 85% du seuil moyen). Garder cette colonne revient à donner la réponse au modèle → <b>data leakage total</b>.
      En production, on ne connaît pas encore le rendement au moment de prédire.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
      <div class="card-title">Q11 — Normalisation (StandardScaler)</div>
      <p>StandardScaler centre (moyenne=0) et réduit (écart-type=1) chaque variable numérique.</p>
      <ul>
        <li><b>Régression logistique</b> : indispensable — les coefficients sont directement comparables seulement si les variables sont sur la même échelle. Sans normalisation, une variable en milliers (altitude) dominerait artificiellement.</li>
        <li><b>Arbres / forêts</b> : moins critique — les splits sont basés sur des seuils relatifs, pas des distances.</li>
        <li>Dans notre pipeline : StandardScaler appliqué <i>uniquement aux variables numériques</i> via ColumnTransformer (les OHE restent 0/1).</li>
      </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
      <div class="card-title">Q12 — Split Train / Test 80% / 20%</div>
      <ul>
        <li><b>stratify=y</b> : garantit que la proportion bonnes/mauvaises récoltes est identique dans train et test. Sans stratification, le hasard pourrait créer un test quasi-vide de mauvaises récoltes.</li>
        <li><b>random_state=42</b> : assure la reproductibilité totale. Sans cela, chaque exécution donne des résultats légèrement différents, rendant la comparaison de modèles impossible.</li>
      </ul>
    </div>
    """, unsafe_allow_html=True)

    total = len(df)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total (après nettoyage)", total)
    c2.metric("Train (80%)", int(total * 0.8))
    c3.metric("Test (20%)", total - int(total * 0.8))

    # Vérifier l'équilibre des splits
    prop_train = y_tr.mean() * 100
    prop_test  = y_te.mean() * 100
    st.markdown(f"**Proportion bonnes récoltes — Train : {prop_train:.1f}%  |  Test : {prop_test:.1f}%** ✅ (stratification réussie)")

# ═══════════════════════════════════════════════════════════════════
# PAGE : ARBRE DE DÉCISION (Ex. 3)
# ═══════════════════════════════════════════════════════════════════
elif page == "🌳  Arbre de Décision":
    st.markdown('<div class="section-title">🌳 Exercice 3 — Arbre de Décision</div>', unsafe_allow_html=True)

    dt   = models['decision_tree']
    res  = results['decision_tree']

    tab1, tab2, tab3 = st.tabs(["📊 Métriques & Confusion", "🌳 Visualisation & Importance", "📉 Analyse Overfitting"])

    # ── Tab 1 : Métriques ──
    with tab1:
        st.markdown('<div class="card-title">Q13 — Performance (max_depth=4, criterion=gini)</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy",  f"{res['accuracy']:.4f}")
        c2.metric("AUC",       f"{res['auc']:.4f}")
        f1_1 = res['report'].get('1', {}).get('f1-score', 0)
        c3.metric("F1 (bonne récolte)", f"{f1_1:.4f}")

        rpt = pd.DataFrame(res['report']).T.round(3)
        rpt_display = rpt.copy()
        rpt_display['support'] = rpt_display['support'].astype(int).astype(str)
        for col in ['precision','recall','f1-score']:
            if col in rpt_display.columns:
                rpt_display[col] = rpt_display[col].astype(str)
        st.dataframe(rpt_display, width='stretch')

        st.markdown('<div class="card-title" style="margin-top:1rem">Q14 — Matrice de confusion</div>', unsafe_allow_html=True)
        cm = res['cm']
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(cm, display_labels=['Mauvaise', 'Bonne']).plot(
            ax=ax, colorbar=False, cmap='Greens')
        ax.set_title("Matrice de confusion — Arbre de Décision")
        fig_style(fig, [ax]); st.pyplot(fig); plt.close()

        tn, fp, fn, tp = cm.ravel()
        c1, c2 = st.columns(2)
        c1.metric("Faux positifs (fausses alertes)", int(fp))
        c2.metric("Faux négatifs (mauvaises récoltes manquées)", int(fn))

        st.markdown("""
        <div class="card">
          <div class="card-title">⚠️ Quel type d'erreur est le plus coûteux ?</div>
          <p>Dans un contexte agricole réel, les <b>faux négatifs</b> sont les plus coûteux : ne pas alerter
          un agriculteur d'une mauvaise récolte potentielle l'empêche de réagir à temps (irrigation d'urgence,
          application d'engrais, souscription d'une assurance récolte, diversification). Un faux positif
          est moins grave : l'agriculteur se prépare pour rien mais ne subit pas de perte majeure.</p>
        </div>
        """, unsafe_allow_html=True)

    # ── Tab 2 : Visualisation ──
    with tab2:
        st.markdown('<div class="card-title">Q15 — Visualisation de l\'arbre (profondeur ≤ 3 affichée)</div>', unsafe_allow_html=True)
        clf_dt = dt.named_steps['clf']
        prep_dt = dt.named_steps['prep']
        try:
            fn_out = prep_dt.get_feature_names_out()
        except Exception:
            fn_out = [f'f{i}' for i in range(clf_dt.n_features_in_)]

        fig, ax = plt.subplots(figsize=(22, 9))
        plot_tree(clf_dt, ax=ax,
          feature_names=fn_out,
          class_names=['Mauvaise', 'Bonne'],
          filled=True, max_depth=3,
          rounded=True, fontsize=8,
          proportion=False)
        ax.set_title("Arbre de Décision — max_depth=4 (affiché jusqu'à 3)", fontsize=13, pad=12)
        fig_style(fig, [ax]); st.pyplot(fig); plt.close()

        # Identifier la variable racine
        root_feature = fn_out[clf_dt.tree_.feature[0]]
        root_threshold = clf_dt.tree_.threshold[0]
        st.markdown(f"""
        <div class="card">
          <div class="card-title">Interprétation du premier nœud (racine)</div>
          <p><b>Variable :</b> <code>{root_feature}</code> — Seuil : <code>{root_threshold:.3f}</code></p>
          <p>Cela signifie que c'est le facteur le plus discriminant pour prédire la qualité de la récolte.
          Agronomiquement, ce split reflète la variable climatique ou agronomique qui crée la plus grande
          hétérogénéité dans les rendements observés au Burundi.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="card-title">Q16 — Importance des variables</div>', unsafe_allow_html=True)
        imp = pd.Series(clf_dt.feature_importances_, index=fn_out).sort_values(ascending=True)
        imp_top = imp.tail(15)
        fig, ax = plt.subplots(figsize=(9, 5))
        colors_imp = ['#f4a261' if i >= len(imp_top)-3 else '#2d6a4f' for i in range(len(imp_top))]
        imp_top.plot(kind='barh', ax=ax, color=colors_imp)
        ax.set_title("Importance des variables — Arbre de Décision", fontsize=12)
        ax.set_xlabel("Importance (réduction Gini)")
        fig_style(fig, [ax]); st.pyplot(fig); plt.close()

        top3 = imp.tail(3).index.tolist()[::-1]
        st.markdown(f"**Top 3 variables :** `{top3[0]}`, `{top3[1]}`, `{top3[2]}`")
        st.info("L'utilisation d'engrais dans le top des variables envoie un message clair aux décideurs : l'accès aux intrants agricoles est un levier majeur d'amélioration des récoltes au Burundi.")

    # ── Tab 3 : Overfitting ──
    with tab3:
        st.markdown('<div class="card-title">Q17 — Accuracy Train vs Test selon max_depth</div>', unsafe_allow_html=True)
        train_scores, test_scores = [], []
        for d in range(1, 21):
            pipe_d = Pipeline([
                ('prep', build_preprocessor()),
                ('clf',  DecisionTreeClassifier(max_depth=d, random_state=42))
            ])
            pipe_d.fit(X_tr, y_tr)
            train_scores.append(pipe_d.score(X_tr, y_tr))
            test_scores.append(pipe_d.score(X_te, y_te))

        best_d = int(np.argmax(test_scores)) + 1
        fig, ax = plt.subplots(figsize=(11, 5))
        depths = list(range(1, 21))
        ax.plot(depths, train_scores, 'o-', color='#f4a261', lw=2.5, markersize=7, label='Accuracy Train')
        ax.plot(depths, test_scores,  'o-', color='#2d6a4f', lw=2.5, markersize=7, label='Accuracy Test')
        ax.axvline(4, color='#52b788', ls='--', alpha=0.8, label='max_depth=4 (modèle utilisé)')
        ax.axvline(best_d, color='#b91c1c', ls=':', alpha=0.8, label=f'Meilleure profondeur test (={best_d})')
        ax.fill_between(depths, train_scores, test_scores,
                         where=[tr > te for tr, te in zip(train_scores, test_scores)],
                         alpha=0.08, color='#b91c1c', label='Zone overfitting')
        ax.set_xlabel("Profondeur max de l'arbre"); ax.set_ylabel("Accuracy")
        ax.set_title("Overfitting : Accuracy Train vs Test selon max_depth", fontsize=13)
        ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.25); ax.set_xticks(depths)
        fig_style(fig, [ax]); st.pyplot(fig); plt.close()

        st.info(f"""
**Analyse overfitting :**
- La courbe train monte vers 100% dès depth ≥ 8–10 → l'arbre mémorise le jeu d'entraînement.
- La courbe test plafonne puis redescend → le modèle généralise moins bien.
- **Meilleure profondeur test : {best_d}** | Depth=4 est un bon compromis biais/variance.
- **En contexte agricole :** un arbre trop profond apprend des règles ultra-spécifiques (ex: "si province=Kayanza ET année=2018 ET engrais=1 ET superficie=2.3 ha → bonne récolte") qui sont inutilisables pour une nouvelle saison ou province.
        """)

# ═══════════════════════════════════════════════════════════════════
# PAGE : FORÊT ALÉATOIRE (Ex. 4)
# ═══════════════════════════════════════════════════════════════════
elif page == "🌲  Forêt Aléatoire":
    st.markdown('<div class="section-title">🌲 Exercice 4 — Forêt Aléatoire</div>', unsafe_allow_html=True)

    rf  = models['random_forest']
    res = results['random_forest']
    res_dt = results['decision_tree']

    tab1, tab2, tab3 = st.tabs(["📊 Métriques & Comparaison", "🔄 Validation croisée", "📊 Importance & n_estimators"])

    # ── Tab 1 ──
    with tab1:
        st.markdown('<div class="card-title">Q18 — Comparaison Forêt Aléatoire vs Arbre de Décision</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("RF — Accuracy",  f"{res['accuracy']:.4f}")
        c2.metric("DT — Accuracy",  f"{res_dt['accuracy']:.4f}",
                  delta=f"{res['accuracy']-res_dt['accuracy']:+.4f}")
        c3.metric("RF — AUC",       f"{res['auc']:.4f}")
        c4.metric("DT — AUC",       f"{res_dt['auc']:.4f}",
                  delta=f"{res['auc']-res_dt['auc']:+.4f}")

        cm = res['cm']
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(cm, display_labels=['Mauvaise', 'Bonne']).plot(
            ax=ax, colorbar=False, cmap='Greens')
        ax.set_title("Matrice de confusion — Forêt Aléatoire")
        fig_style(fig, [ax]); st.pyplot(fig); plt.close()

        st.markdown("""
        <div class="card">
          <div class="card-title">Q19 — Pourquoi la forêt est-elle plus performante ?</div>
          <ul>
            <li><b>Bagging (Bootstrap Aggregating) :</b> chaque arbre est entraîné sur un sous-ensemble
            aléatoire des données (tirage avec remise). La prédiction finale est le vote majoritaire des
            100 arbres → la variance est fortement réduite.</li>
            <li><b>max_features :</b> à chaque nœud de chaque arbre, seul un sous-ensemble aléatoire
            de features est considéré pour le split → les arbres sont décorrélés entre eux. Un arbre
            seul qui voit toutes les features tendra à utiliser les mêmes variables à chaque fois.</li>
            <li><b>Overfitting de la forêt :</b> possible si max_depth est illimité ET n_estimators très
            grand, mais beaucoup moins prononcé qu'un seul arbre car l'agrégation lisse les extrêmes.</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

    # ── Tab 2 : CV ──
    with tab2:
        st.markdown('<div class="card-title">Q20 — Validation croisée 5 folds</div>', unsafe_allow_html=True)
        cv = res['cv']
        c1, c2, c3 = st.columns(3)
        c1.metric("CV Moyenne",     f"{np.mean(cv):.4f}")
        c2.metric("CV Écart-type",  f"{np.std(cv):.4f}")
        c3.metric("Test Accuracy",  f"{res['accuracy']:.4f}",
                  delta=f"{res['accuracy']-np.mean(cv):+.4f}")

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(range(1, 6), cv, color='#2d6a4f', edgecolor='white', width=0.6)
        ax.axhline(np.mean(cv), color='#f4a261', lw=2.5, ls='--',
                   label=f'Moyenne CV = {np.mean(cv):.4f}')
        ax.axhline(res['accuracy'], color='#52b788', lw=2, ls=':',
                   label=f'Test simple = {res["accuracy"]:.4f}')
        ax.set_xlabel("Fold"); ax.set_ylabel("Accuracy")
        ax.set_title("Validation croisée à 5 folds — Forêt Aléatoire", fontsize=12)
        ax.legend(); ax.set_ylim(0.6, 1.0); ax.set_xticks(range(1, 6))
        fig_style(fig, [ax]); st.pyplot(fig); plt.close()

        st.info("La validation croisée est plus fiable : elle évalue le modèle sur **l'intégralité des données** successivement, réduisant la variance liée au tirage aléatoire du jeu de test. L'écart-type indique la stabilité du modèle.")

    # ── Tab 3 : Importance & n_estimators ──
    with tab3:
        st.markdown('<div class="card-title">Q21 — Importance des variables (RF vs DT)</div>', unsafe_allow_html=True)
        clf_rf = rf.named_steps['clf']
        prep_rf = rf.named_steps['prep']
        try:
            fn_rf = prep_rf.get_feature_names_out()
        except Exception:
            fn_rf = [f'f{i}' for i in range(clf_rf.n_features_in_)]

        imp_rf = pd.Series(clf_rf.feature_importances_, index=fn_rf).sort_values(ascending=True).tail(15)

        clf_dt = models['decision_tree'].named_steps['clf']
        prep_dt = models['decision_tree'].named_steps['prep']
        try:
            fn_dt = prep_dt.get_feature_names_out()
        except Exception:
            fn_dt = [f'f{i}' for i in range(clf_dt.n_features_in_)]
        imp_dt = pd.Series(clf_dt.feature_importances_, index=fn_dt).sort_values(ascending=False).head(5)

        col1, col2 = st.columns([3, 2])
        with col1:
            fig, ax = plt.subplots(figsize=(9, 5))
            imp_rf.plot(kind='barh', ax=ax, color='#52b788')
            ax.set_title("Importance des variables — Forêt Aléatoire (Top 15)", fontsize=12)
            ax.set_xlabel("Importance moyenne (MDI)")
            fig_style(fig, [ax]); st.pyplot(fig); plt.close()
        with col2:
            st.markdown("**Top 5 — Arbre de Décision**")
            st.dataframe(
                imp_dt.reset_index().rename(columns={'index':'Feature', 0:'Importance'}).astype(str),
                width='stretch', hide_index=True
            )
            st.markdown("**Top 5 — Forêt Aléatoire**")
            st.dataframe(
                imp_rf.sort_values(ascending=False).head(5).reset_index()
                       .rename(columns={'index':'Feature', 0:'Importance'}).astype(str),
                width='stretch', hide_index=True
            )

        st.markdown('<div class="card-title" style="margin-top:1rem">Q22 — Impact du nombre d\'arbres</div>', unsafe_allow_html=True)
        n_list = [10, 25, 50, 75, 100, 150, 200, 300, 500]
        scores_n = []
        for n in n_list:
            pipe_n = Pipeline([
                ('prep', build_preprocessor()),
                ('clf', RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1))
            ])
            pipe_n.fit(X_tr, y_tr)
            scores_n.append(pipe_n.score(X_te, y_te))

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(n_list, scores_n, 'o-', color='#2d6a4f', lw=2.5, markersize=8)
        ax.set_xlabel("Nombre d'arbres (n_estimators)"); ax.set_ylabel("Accuracy Test")
        ax.set_title("Accuracy Test vs n_estimators", fontsize=12)
        ax.grid(axis='y', alpha=0.25)
        fig_style(fig, [ax]); st.pyplot(fig); plt.close()
        st.info("La performance se stabilise généralement à partir de **100–150 arbres**. Au-delà, le gain en accuracy est marginal (<0.1%) mais le temps de calcul et la mémoire augmentent linéairement avec n_estimators.")

# ═══════════════════════════════════════════════════════════════════
# PAGE : RÉGRESSION LOGISTIQUE (Ex. 5)
# ═══════════════════════════════════════════════════════════════════
elif page == "📈  Régression Logistique":
    st.markdown('<div class="section-title">📈 Exercice 5 — Régression Logistique</div>', unsafe_allow_html=True)

    lr  = models['logistic_regression']
    res = results['logistic_regression']

    tab1, tab2 = st.tabs(["📊 Métriques & Coefficients", "💡 Interprétation"])

    with tab1:
        st.markdown('<div class="card-title">Q23 — Coefficients (max_iter=1000)</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{res['accuracy']:.4f}")
        c2.metric("AUC",      f"{res['auc']:.4f}")
        f1_lr = res['report'].get('1', {}).get('f1-score', 0)
        c3.metric("F1 (bonne récolte)", f"{f1_lr:.4f}")

        clf_lr = lr.named_steps['clf']
        prep_lr = lr.named_steps['prep']
        try:
            fn_lr = prep_lr.get_feature_names_out()
        except Exception:
            fn_lr = [f'f{i}' for i in range(clf_lr.n_features_in_)]

        coefs = pd.Series(clf_lr.coef_[0], index=fn_lr).sort_values()
        # Afficher top positifs et négatifs
        coef_display = pd.concat([coefs.head(10), coefs.tail(10)]).round(4)
        colors_c = ['#b91c1c' if v < 0 else '#2d6a4f' for v in coef_display.values]

        fig, ax = plt.subplots(figsize=(10, 7))
        coef_display.plot(kind='barh', ax=ax, color=colors_c)
        ax.axvline(0, color='black', lw=0.9)
        ax.set_title("Coefficients Régression Logistique (Top/Bottom 10)", fontsize=12)
        ax.set_xlabel("Coefficient (impact sur log-probabilité de bonne récolte)")
        # Légende
        pos_patch = mpatches.Patch(color='#2d6a4f', label='Coefficient positif → favorise bonne récolte')
        neg_patch = mpatches.Patch(color='#b91c1c', label='Coefficient négatif → défavorise bonne récolte')
        ax.legend(handles=[pos_patch, neg_patch], fontsize=9)
        fig_style(fig, [ax]); st.pyplot(fig); plt.close()

        top_pos = coefs.tail(3).index.tolist()[::-1]
        top_neg = coefs.head(3).index.tolist()
        st.markdown(f"**Top 3 coefficients positifs :** `{'`, `'.join(top_pos)}`")
        st.markdown(f"**Top 3 coefficients négatifs :** `{'`, `'.join(top_neg)}`")

    with tab2:
        st.markdown("""
        <div class="card">
          <div class="card-title">Q24 — Performances comparées & hypothèse de linéarité</div>
          <p>La régression logistique est souvent <b>légèrement moins performante</b> que les arbres/forêts sur des données agricoles car :</p>
          <ul>
            <li>Elle suppose une relation <b>linéaire</b> entre les features et la log-probabilité → non réaliste pour des interactions complexes (ex: l'effet de la pluie dépend de l'altitude).</li>
            <li>Les seuils climatiques non-linéaires (ex: >800mm de pluie = saturation, <300mm = stress hydrique) sont naturellement capturés par des splits d'arbres, pas par une droite logistique.</li>
            <li>Avantage de la RL : <b>interprétabilité directe</b> des coefficients — utile pour les décideurs qui veulent comprendre chaque levier.</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

        # Rapport de classification complet
        rpt = pd.DataFrame(res['report']).T.round(3)
        for col in rpt.columns:
            rpt[col] = rpt[col].astype(str)
        st.dataframe(rpt, width='stretch')

# ═══════════════════════════════════════════════════════════════════
# PAGE : COMPARAISON ROC (Ex. 5 suite)
# ═══════════════════════════════════════════════════════════════════
elif page == "📉  Comparaison ROC":
    st.markdown('<div class="section-title">📉 Exercice 5 — Comparaison ROC des 3 Modèles</div>', unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    for name, label in MODEL_LABELS.items():
        fpr, tpr, _ = results[name]['roc']
        auc_val = results[name]['auc']
        ax.plot(fpr, tpr, lw=2.5, color=COLORS[name], label=f"{label}  (AUC = {auc_val:.3f})")
    ax.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Aléatoire (AUC = 0.500)')
    ax.fill_between([0, 1], [0, 1], alpha=0.04, color='gray')
    ax.set_xlabel("Taux de faux positifs (FPR)", fontsize=12)
    ax.set_ylabel("Taux de vrais positifs (Rappel / TPR)", fontsize=12)
    ax.set_title("Courbes ROC — Comparaison des 3 modèles", fontsize=14, pad=12)
    ax.legend(fontsize=11); ax.grid(alpha=0.25)
    fig_style(fig, [ax]); st.pyplot(fig); plt.close()

    # Tableau récapitulatif
    rows = []
    for name, label in MODEL_LABELS.items():
        r = results[name]
        rows.append({
            'Modèle':            label,
            'Accuracy':          f"{r['accuracy']:.4f}",
            'AUC':               f"{r['auc']:.4f}",
            'F1 (bonne récolte)': f"{r['report'].get('1',{}).get('f1-score',0):.4f}",
            'Recall (mauvaise)': f"{r['report'].get('0',{}).get('recall',0):.4f}",
        })
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

    st.markdown("""
    <div class="card">
      <div class="card-title">Q25–Q26 — Ce que mesure l'AUC et pourquoi c'est mieux que l'accuracy</div>
      <ul>
        <li><b>AUC (Area Under Curve)</b> mesure la capacité du modèle à <i>discriminer</i> les deux classes
        sur <i>tous les seuils de décision possibles</i>. AUC=1.0 = parfait, AUC=0.5 = aléatoire.</li>
        <li>Un modèle avec 75% d'accuracy peut avoir AUC ≈ 0.5 : si 75% des données sont "bonne récolte",
        un modèle qui prédit toujours "bonne" atteint 75% accuracy mais ne discrimine rien (AUC=0.5).</li>
        <li>En agriculture, <b>l'AUC est préférable</b> : l'agriculteur veut que le modèle l'alerte
        <i>fiablement</i> d'une mauvaise récolte (bon recall sur la classe 0), pas juste qu'il soit
        souvent correct en moyenne.</li>
      </ul>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# PAGE : PRÉDICTION SCÉNARIOS (Ex. 6)
# ═══════════════════════════════════════════════════════════════════
elif page == "🔮  Prédiction Scénarios":
    st.markdown('<div class="section-title">🔮 Exercice 6 — Prédiction sur 4 Scénarios Réels du Burundi</div>', unsafe_allow_html=True)

    med = df.median(numeric_only=True)

    SCENARIOS = [
        {'label': 'Kayanza – Maïs',        'province': 'Kayanza',  'culture': 'Maïs',
         'altitude_m': 1980, 'pluviometrie_mm': 920, 'temperature_moy_C': 17.8,
         'utilisation_engrais': 1, 'acces_irrigation': 0},
        {'label': 'Bubanza – Manioc',       'province': 'Bubanza',  'culture': 'Manioc',
         'altitude_m': 790,  'pluviometrie_mm': 550, 'temperature_moy_C': 25.4,
         'utilisation_engrais': 0, 'acces_irrigation': 1},
        {'label': 'Gitega – Haricot',       'province': 'Gitega',   'culture': 'Haricot',
         'altitude_m': 1720, 'pluviometrie_mm': 430, 'temperature_moy_C': 18.2,
         'utilisation_engrais': 0, 'acces_irrigation': 0},
        {'label': 'Cibitoke – Patate douce','province': 'Cibitoke', 'culture': 'Patate douce',
         'altitude_m': 810,  'pluviometrie_mm': 810, 'temperature_moy_C': 24.1,
         'utilisation_engrais': 1, 'acces_irrigation': 1},
    ]

    table_rows = []
    for sc in SCENARIOS:
        row = {'Scénario': sc['label']}
        sc_input = {
            'annee': 2023, 'saison': 'A',
            'superficie_ha': float(med.get('superficie_ha', 2.0)),
            'nb_menages':    int(med.get('nb_menages', 50)),
            **{k: v for k, v in sc.items() if k != 'label'}
        }
        X_sc = pd.DataFrame([sc_input])
        for name, label in MODEL_LABELS.items():
            pred = models[name].predict(X_sc)[0]
            prob = models[name].predict_proba(X_sc)[0][1] * 100
            emoji = "✅" if pred == 1 else "❌"
            row[label] = f"{emoji} {'Bonne' if pred==1 else 'Mauvaise'} ({prob:.0f}%)"
        table_rows.append(row)

    st.dataframe(pd.DataFrame(table_rows), width='stretch', hide_index=True)

    # Graphique probabilités
    fig, axes = plt.subplots(1, 4, figsize=(14, 5), sharey=True)
    for i, sc in enumerate(SCENARIOS):
        sc_input = {
            'annee': 2023, 'saison': 'A',
            'superficie_ha': float(med.get('superficie_ha', 2.0)),
            'nb_menages':    int(med.get('nb_menages', 50)),
            **{k: v for k, v in sc.items() if k != 'label'}
        }
        X_sc = pd.DataFrame([sc_input])
        probs = {label: models[name].predict_proba(X_sc)[0][1]*100
                 for name, label in MODEL_LABELS.items()}
        ax = axes[i]
        bar_colors = ['#2d6a4f' if p >= 50 else '#b91c1c' for p in probs.values()]
        bars = ax.barh(list(probs.keys()), list(probs.values()),
                       color=bar_colors, edgecolor='white')
        ax.axvline(50, color='gray', lw=1.2, ls='--', alpha=0.7)
        ax.set_title(sc['label'], fontsize=9, fontweight='bold')
        ax.set_xlim(0, 100)
        ax.set_xlabel("P(bonne récolte) %", fontsize=8)
        ax.bar_label(bars, fmt='%.0f%%', padding=3, fontsize=8)
        ax.set_facecolor(FIG_BG)
    fig_style(fig)
    st.pyplot(fig); plt.close()

    # Réponses aux questions
    st.markdown("""
    <div class="card">
      <div class="card-title">Q28 — Désaccords entre modèles</div>
      <p>Quand les modèles ne s'accordent pas : utiliser le <b>vote majoritaire</b> (2 sur 3).
      Pour les cas d'incertitude (probabilités proches de 50%), l'agronome doit compléter
      l'analyse par une inspection terrain. Le modèle avec la meilleure AUC est prioritaire
      en cas d'égalité.</p>
    </div>
    <div class="card">
      <div class="card-title">Q29 — Gitega – Haricot (430 mm) : analyse</div>
      <p>430 mm est nettement insuffisant pour le Haricot au Burundi (besoin minimal ~600 mm/saison).
      Une mauvaise récolte est attendue — prédiction cohérente avec la réalité agronomique.</p>
      <p><b>Recommandations pour l'agriculteur de Gitega :</b></p>
      <ul>
        <li>Installer un système de collecte/stockage des eaux de pluie (citernes, bassins)</li>
        <li>Utiliser des variétés de haricot tolérantes à la sécheresse (ex. RWR 2245)</li>
        <li>Appliquer du paillage organique pour limiter l'évaporation du sol</li>
        <li>Reporter la plantation à la saison B si les prévisions météo sont meilleures</li>
        <li>Diversifier avec une culture moins exigeante en eau (ex. Sorgho)</li>
      </ul>
    </div>
    <div class="card">
      <div class="card-title">Q30 — Recommandation au Ministère de l'Agriculture</div>
      <p><b>Modèle recommandé en production : Forêt Aléatoire</b></p>
      <ul>
        <li>Meilleure AUC et accuracy des 3 modèles</li>
        <li>Robuste aux valeurs aberrantes et aux données manquantes</li>
        <li>Performance stable confirmée par validation croisée (faible écart-type)</li>
        <li>Pas d'hypothèse de linéarité — adapté aux interactions climatiques complexes</li>
      </ul>
      <p><b>Données supplémentaires souhaitées :</b> indices NDVI (satellites), qualité des sols (pH, N/P/K),
      prix des intrants, prévisions météo à 3 mois, historique des maladies des cultures.</p>
      <p><b>Limites :</b> dataset simulé, absence de données satellitaires, biais potentiel pour les provinces
      sous-représentées, non-prise en compte des chocs économiques (prix des engrais, conflits).</p>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# PAGE : PRÉDICTION PERSONNALISÉE (Ex. 7)
# ═══════════════════════════════════════════════════════════════════
elif page == "🎯  Prédiction Personnalisée":
    st.markdown('<div class="section-title">🎯 Exercice 7 — Prédiction Personnalisée</div>', unsafe_allow_html=True)
    st.markdown("Saisissez les caractéristiques d'une parcelle agricole pour obtenir une prédiction en temps réel.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**📍 Localisation & Culture**")
        province = st.selectbox("Province",  PROVINCES)
        culture  = st.selectbox("Culture",   CULTURES)
        saison   = st.selectbox("Saison",    ["A", "B"])
        annee    = st.slider("Année", 2015, 2030, 2024)

    with col2:
        st.markdown("**🌦️ Conditions climatiques**")
        altitude = st.slider("Altitude (m)",           400,  2700, 1200, step=10)
        pluie    = st.slider("Pluviométrie (mm)",       100,  1800,  800, step=10)
        temp     = st.slider("Température moyenne (°C)", 14.0, 30.0, 20.0, step=0.1)

    col3, col4, col5 = st.columns(3)
    with col3:
        superficie  = st.slider("Superficie (ha)", 0.1, 20.0, 2.0, step=0.1)
    with col4:
        nb_menages  = st.number_input("Nb ménages", 1, 500, 50)
    with col5:
        engrais   = st.checkbox("🌿 Utilisation d'engrais",   value=False)
        irrigation = st.checkbox("💧 Accès à l'irrigation",   value=False)

    st.markdown("---")
    model_choice = st.selectbox(
        "🤖 Modèle à utiliser",
        list(MODEL_LABELS.values()),
        index=1  # Forêt Aléatoire par défaut
    )
    model_key = {v: k for k, v in MODEL_LABELS.items()}[model_choice]

    if st.button("🌱 Lancer la prédiction"):
        input_data = {
            'annee':              annee,
            'saison':             saison,
            'province':           province,
            'culture':            culture,
            'altitude_m':         altitude,
            'pluviometrie_mm':    pluie,
            'temperature_moy_C':  temp,
            'superficie_ha':      superficie,
            'utilisation_engrais': int(engrais),
            'acces_irrigation':    int(irrigation),
            'nb_menages':         nb_menages,
        }
        X_pred = pd.DataFrame([input_data])
        pred  = models[model_key].predict(X_pred)[0]
        probs = models[model_key].predict_proba(X_pred)[0]

        if pred == 1:
            st.markdown(f"""
            <div class="pred-good">
              ✅ BONNE RÉCOLTE PRÉDITE
              <span class="pred-sub">Probabilité de bonne récolte : {probs[1]*100:.1f}%  ·  Modèle : {model_choice}</span>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="pred-bad">
              ❌ MAUVAISE RÉCOLTE PRÉDITE
              <span class="pred-sub">Probabilité de mauvaise récolte : {probs[0]*100:.1f}%  ·  Modèle : {model_choice}</span>
            </div>""", unsafe_allow_html=True)

        # Métriques du modèle
        st.markdown("---")
        st.markdown("**Métriques globales du modèle sélectionné :**")
        r = results[model_key]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy",  f"{r['accuracy']:.4f}")
        c2.metric("AUC",       f"{r['auc']:.4f}")
        c3.metric("F1 (bonne)", f"{r['report'].get('1',{}).get('f1-score',0):.4f}")
        cv_m = np.mean(r['cv'])
        c4.metric("CV Moyenne", f"{cv_m:.4f}")

        # Importance des variables pour ce modèle
        clf_ = models[model_key].named_steps['clf']
        prep_ = models[model_key].named_steps['prep']
        if hasattr(clf_, 'feature_importances_'):
            st.markdown("**Importance des variables du modèle :**")
            try:
                fn_ = prep_.get_feature_names_out()
            except Exception:
                fn_ = [f'f{i}' for i in range(clf_.n_features_in_)]
            imp_ = pd.Series(clf_.feature_importances_, index=fn_).sort_values(ascending=True).tail(12)
            fig, ax = plt.subplots(figsize=(9, 4))
            imp_.plot(kind='barh', ax=ax, color='#52b788')
            ax.set_title(f"Importance des variables — {model_choice}", fontsize=11)
            ax.set_xlabel("Importance")
            fig_style(fig, [ax]); st.pyplot(fig); plt.close()
        elif hasattr(clf_, 'coef_'):
            st.markdown("**Coefficients du modèle (régression logistique) :**")
            try:
                fn_ = prep_.get_feature_names_out()
            except Exception:
                fn_ = [f'f{i}' for i in range(clf_.n_features_in_)]
            coef_ = pd.Series(clf_.coef_[0], index=fn_).abs().sort_values(ascending=True).tail(12)
            fig, ax = plt.subplots(figsize=(9, 4))
            coef_.plot(kind='barh', ax=ax, color='#f4a261')
            ax.set_title(f"Coefficients absolus — {model_choice}", fontsize=11)
            ax.set_xlabel("|Coefficient|")
            fig_style(fig, [ax]); st.pyplot(fig); plt.close()