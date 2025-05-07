import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.integrate import solve_ivp
import time

# === CSS personnalisé ===

st.markdown("""
    <style>
        .stApp {
            background-color: #000000;  /* Fond gris clair */
        }
        .stMarkdown, .stText, .stSubheader, .stTitle {
            color: #8e44ad;  /* Texte violet */
        }
        .stButton>button, .stSlider>div>div>div>div {
            background-color: #87CEEB !important;  /* Bleu ciel clair pour boutons/sliders */
            color: #000000 !important;  /* Texte noir pour lisibilité */
        }
    </style>
""", unsafe_allow_html=True)

# === Modèle ===
def competition_model(t, z, r1, r2, K1, K2, a, b):
    x, y = z
    dxdt = r1 * x * (1 - (x + a * y) / K1)
    dydt = r2 * y * (1 - (y + b * x) / K2)
    return [dxdt, dydt]

def run_simulation(r1, r2, K1, K2, a, b, x0, y0, t_max, points):
    t_span = (0, t_max)
    t_eval = np.linspace(*t_span, points)
    sol = solve_ivp(competition_model, t_span, [x0, y0], args=(r1, r2, K1, K2, a, b), t_eval=t_eval)
    return sol.t, sol.y[0], sol.y[1]

# === Titre ===
st.markdown('<h2 style="color: #4CAF50;"> Modèle de Compétition entre Espèces</h2>', unsafe_allow_html=True)
#st.markdown('<div class="title">🌿 Modèle de Compétition entre Espèces</div>', unsafe_allow_html=True)

# === Équations ===
st.markdown('<p style="color: #8e44ad;">Modèle mathématique</p>', unsafe_allow_html=True)
with st.expander("📖 Modèle utilisé (équations)"):
    st.latex(r"""
    \begin{cases}
    \frac{dx}{dt} = r_1 x \left(1 - \frac{x + a y}{K_1} \right) \\[6pt]
    \frac{dy}{dt} = r_2 y \left(1 - \frac{y + b x}{K_2} \right)
    \end{cases}
    """)

# === Layout principal ===
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown('<div class="info-box"><b>🔧 Paramètres biologiques</b></div>', unsafe_allow_html=True)
    st.markdown('<p style="color: #8e44ad;">Taux de croissance espèce 1 (r₁)</p>', unsafe_allow_html=True)
    r1 = st.slider("", 0.01, 1.0, 0.13, 0.01)
    st.caption("Plus r₁ est grand, plus l'espèce 1 croît rapidement.")
    st.markdown('<p style="color: #8e44ad;">Taux de croissance espèce 2 (r₂)</p>', unsafe_allow_html=True)
    r2 = st.slider("", 0.01, 1.0, 0.98, 0.01)
    st.caption("Plus r₂ est grand, plus l'espèce 2 croît rapidement.")
    
    st.markdown('<p style="color: #8e44ad;">Capacité de charge de l\'espèce 1 (K₁)</p>', unsafe_allow_html=True)
    K1 = st.number_input("", min_value=10, value=600)
    st.caption("Nombre maximal d'individus que le milieu peut supporter pour l'espèce 1.")
    st.markdown('<p style="color: #8e44ad;">Capacité de charge de l\'espèce 2 (K₂)</p>', unsafe_allow_html=True)
    K2 = st.number_input("", min_value=10, value=500)
    st.caption("Nombre maximal d'individus que le milieu peut supporter pour l'espèce 2.")

    st.markdown('<p style="color: #8e44ad;">Impact de l\'espèce 2 sur l\'espèce 1 (a)</p>', unsafe_allow_html=True)
    a = st.slider("", 0.0, 1.0, 0.1, 0.01)
    st.caption("Si a > 0, l'espèce 2 réduit la croissance de l'espèce 1 (compétition).")
    st.markdown('<p style="color: #8e44ad;">Impact de l\'espèce 1 sur l\'espèce 2 (b)</p>', unsafe_allow_html=True)
    b = st.slider("", 0.0, 1.0, 0.4, 0.01)
    st.caption("Si b > 0, l'espèce 1 réduit la croissance de l'espèce 2.")

    st.markdown('<div class="info-box"><b>⚙️ Conditions initiales</b></div>', unsafe_allow_html=True)
    st.markdown('<p style="color: #8e44ad;">Population initiale espèce 1 (N₁)</p>', unsafe_allow_html=True)
    x0 = st.number_input("", min_value=1, value=100)
    st.markdown('<p style="color: #8e44ad;">Population initiale espèce 2 (N₂)</p>', unsafe_allow_html=True)
    y0 = st.number_input("", min_value=1, value=70)
    st.caption("Nombre d'individus au début de la simulation.")

    st.markdown('<p style="color: #8e44ad;">⏳ Durée de la simulation</p>', unsafe_allow_html=True)
    t_max = st.slider("", 1, 1000, 100)
    simulate = st.button("🚀 Lancer la simulation")

with col2:
    if simulate:
        with st.spinner("Simulation en cours..."):
            t, x, y = run_simulation(r1, r2, K1, K2, a, b, x0, y0, t_max, t_max + 1)

            # === Graphique temporel ===
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(t, x, label="🟦 Espèce 1", color="royalblue", linewidth=2)
            ax.plot(t, y, label="🟥 Espèce 2", color="crimson", linewidth=2)
            ax.set_xlabel("Temps", fontsize=12)
            ax.set_ylabel("Population", fontsize=12)
            ax.set_title("Évolution temporelle des populations", fontsize=14)
            ax.grid(alpha=0.3)
            ax.legend()
            st.pyplot(fig)

            # === Résumé pédagogique ===
            st.markdown('<div class="info-box"><b>🔍 Résumé de la simulation</b></div>', unsafe_allow_html=True)
            final_x = x[-1]
            final_y = y[-1]

            if final_x < 1 and final_y < 1:
                st.error("⚠️ Les deux espèces s'éteignent à long terme.")
            elif final_x < 1:
                st.warning("⚠️ L'espèce 1 s'éteint, l'espèce 2 survit.")
            elif final_y < 1:
                st.warning("⚠️ L'espèce 2 s'éteint, l'espèce 1 survit.")
            else:
                st.success("✅ Les deux espèces coexistent à long terme.")

            # === Animation ===
            st.markdown('<div class="info-box"><b>🎞️ Animation des populations</b></div>', unsafe_allow_html=True)
            spot = st.empty()
            for i in range(0, len(t), max(1, len(t) // 200)):
                fig_anim, ax_anim = plt.subplots(figsize=(5, 5))
                ax_anim.set_xlim(0, 10)
                ax_anim.set_ylim(0, 10)
                ax_anim.set_xticks([])
                ax_anim.set_yticks([])
                ax_anim.set_facecolor("#e0f7fa")

                n_esp1 = min(500, max(0, round(x[i])))
                n_esp2 = min(500, max(0, round(y[i])))

                ax_anim.scatter(np.random.rand(n_esp1) * 8 + 1,
                                np.random.rand(n_esp1) * 8 + 1,
                                color="royalblue", alpha=0.6, label=f"Espèce 1: {round(x[i])}")
                ax_anim.scatter(np.random.rand(n_esp2) * 8 + 1,
                                np.random.rand(n_esp2) * 8 + 1,
                                color="crimson", alpha=0.6, label=f"Espèce 2: {round(y[i])}")
                ax_anim.set_title(f"t = {t[i]:.1f}", fontsize=12)
                ax_anim.legend(loc="upper right")
                spot.pyplot(fig_anim)
                plt.close(fig_anim)
                time.sleep(0.2)
