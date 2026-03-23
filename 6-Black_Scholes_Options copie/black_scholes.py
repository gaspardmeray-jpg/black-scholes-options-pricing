# =============================================================================
# PROJET PYTHON — BLACK-SCHOLES & VOLATILITÉ IMPLICITE
#
# CONCEPTS CLÉS :
#   - Black-Scholes : modèle de pricing d'options (1973, Prix Nobel 1997)
#   - Option call   : droit d'ACHETER un actif à prix K à l'échéance T
#   - Option put    : droit de VENDRE  un actif à prix K à l'échéance T
#   - Vol. implicite: la volatilité σ qui rend le prix BS = prix du marché
#   - Vol. smile    : la vol. implicite varie selon le strike (anomalie du modèle)
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import optimize
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

TICKER = "AAPL"   # action sous-jacente — change à ta guise


# =============================================================================
# EXERCICE 1 — PRIX BLACK-SCHOLES D'UN CALL OU PUT
# =============================================================================

def black_scholes(S, K, T, r, sigma, q=0.0, option_type='call'):
    """
    Calcule le prix théorique d'une option européenne par la formule de Black-Scholes.

    ── Paramètres ────────────────────────────────────────────────────────────
      S           : prix spot de l'actif sous-jacent (ex: 190$)
      K           : prix d'exercice (strike) de l'option (ex: 195$)
      T           : temps jusqu'à l'échéance en ANNÉES (ex: 30 jours = 30/365)
      r           : taux sans risque annuel (ex: 0.05 = 5%)
      sigma       : volatilité annuelle du sous-jacent (ex: 0.25 = 25%)
      q           : taux de dividendes continu annuel (0 si pas de dividendes)
      option_type : 'call' ou 'put'

    ── La formule Black-Scholes ───────────────────────────────────────────────
    Prix call = S·e^(-qT)·N(d1) - K·e^(-rT)·N(d2)
    Prix put  = K·e^(-rT)·N(-d2) - S·e^(-qT)·N(-d1)

    avec :
      d1 = [ ln(S/K) + (r - q + σ²/2)·T ] / (σ·√T)
      d2 = d1 - σ·√T

    N(x) = fonction de répartition de la loi normale standard (scipy.stats.norm.cdf)
         = probabilité qu'une variable N(0,1) soit inférieure à x

    ── Intuition financière ────────────────────────────────────────────────────
    N(d2) ≈ probabilité risque-neutre que l'option finisse dans la monnaie (ITM)
    N(d1) ≈ facteur de couverture (delta) : sensibilité du prix au cours du sous-jacent
    """

    if T <= 0:
        # Option expirée : vaut sa valeur intrinsèque
        if option_type == 'call':
            return max(S - K, 0)
        else:
            return max(K - S, 0)

    # Calcul de d1 et d2
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        prix = (S * np.exp(-q * T) * norm.cdf(d1)
                - K * np.exp(-r * T) * norm.cdf(d2))
    elif option_type == 'put':
        prix = (K * np.exp(-r * T) * norm.cdf(-d2)
                - S * np.exp(-q * T) * norm.cdf(-d1))
    else:
        raise ValueError("option_type doit être 'call' ou 'put'")

    return prix


def vega(S, K, T, r, sigma, q=0.0):
    """
    Calcule le vega : dérivée du prix BS par rapport à σ.

    Vega = S·e^(-qT)·N'(d1)·√T

    où N'(d1) est la densité de la loi normale = norm.pdf(d1)

    Le vega est utilisé dans la méthode de Newton pour trouver la vol. implicite :
    on a besoin de la "pente" de la fonction prix(σ) pour converger rapidement.
    Vega > 0 toujours : un hausse de la vol augmente toujours le prix d'une option.
    """
    if T <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


# =============================================================================
# EXERCICE 2 — VOLATILITÉ IMPLICITE (Newton-Raphson + méthodes scipy)
# =============================================================================

def vol_implicite_newton(prix_marche, S, K, T, r, q=0.0,
                         option_type='call', sigma0=0.3,
                         tol=1e-6, max_iter=100):
    """
    Calcule la volatilité implicite par la méthode de Newton-Raphson.

    ── Qu'est-ce que la volatilité implicite ? ───────────────────────────────
    C'est la valeur de σ telle que :
        black_scholes(S, K, T, r, σ) = prix_marché

    On ne peut pas inverser analytiquement la formule BS pour isoler σ.
    On utilise donc une méthode numérique itérative.

    ── Méthode de Newton-Raphson ─────────────────────────────────────────────
    On cherche le zéro de f(σ) = BS(σ) - prix_marché.
    Newton part d'une estimation initiale σ₀ et itère :

        σ_{n+1} = σ_n - f(σ_n) / f'(σ_n)
                = σ_n - (BS(σ_n) - prix_marché) / vega(σ_n)

    Convergence quadratique : le nombre de décimales correctes double à chaque itération.
    En pratique, 5 à 10 itérations suffisent pour une précision de 1e-6.

    ── Paramètres ────────────────────────────────────────────────────────────
      prix_marche : prix observé de l'option sur le marché
      sigma0      : estimation initiale (30% est un bon point de départ)
      tol         : tolérance (on s'arrête quand |f(σ)| < tol)
      max_iter    : nombre maximum d'itérations
    """
    if T <= 0 or prix_marche <= 0:
        return np.nan

    sigma = sigma0

    for i in range(max_iter):
        prix_bs = black_scholes(S, K, T, r, sigma, q, option_type)
        v       = vega(S, K, T, r, sigma, q)

        diff = prix_bs - prix_marche

        # Convergence atteinte
        if abs(diff) < tol:
            return sigma

        # Éviter la division par un vega quasi-nul (options très OTM/ITM)
        if abs(v) < 1e-10:
            return np.nan

        # Mise à jour de Newton
        sigma = sigma - diff / v

        # σ doit rester positif et raisonnable
        sigma = max(1e-6, min(sigma, 10.0))

    return sigma   # retourne la meilleure estimation même sans convergence parfaite


def vol_implicite_scipy(prix_marche, S, K, T, r, q=0.0,
                        option_type='call', methode='brentq'):
    """
    Calcule la volatilité implicite via scipy.optimize.

    scipy propose plusieurs méthodes de recherche de zéro.
    Ici on utilise brentq (méthode de Brent) par défaut :
      → très robuste, garantit la convergence si f change de signe
      → plus lent que Newton mais plus fiable sur des options extrêmes

    Autres méthodes disponibles via scipy.optimize.root_scalar :
      - bisect    : bissection (lente mais garantie)
      - newton    : Newton-Raphson (rapide mais peut diverger)
      - secant    : méthode de la sécante (pas besoin de la dérivée)
      - brenth    : variante de Brent
    """
    if T <= 0 or prix_marche <= 0:
        return np.nan

    # Fonction dont on cherche le zéro : f(σ) = BS(σ) - prix_marché
    f = lambda sigma: black_scholes(S, K, T, r, sigma, q, option_type) - prix_marche

    try:
        if methode == 'brentq':
            # brentq nécessite un intervalle [a, b] où f(a) et f(b) ont des signes opposés
            result = optimize.brentq(f, 1e-6, 10.0, xtol=1e-6, maxiter=200)
        else:
            result = optimize.root_scalar(f, method=methode,
                                          bracket=[1e-6, 10.0],
                                          xtol=1e-6).root
        return result
    except (ValueError, RuntimeError):
        return np.nan


def comparer_methodes_vi(prix_marche, S, K, T, r, q=0.0, option_type='call'):
    """Compare toutes les méthodes de calcul de la vol. implicite."""
    methodes = {
        'Newton (maison)' : vol_implicite_newton(prix_marche, S, K, T, r, q, option_type),
        'Brent (scipy)'   : vol_implicite_scipy(prix_marche, S, K, T, r, q, option_type, 'brentq'),
        'Secante (scipy)' : vol_implicite_scipy(prix_marche, S, K, T, r, q, option_type, 'secant'),
        'Bisection (scipy)': vol_implicite_scipy(prix_marche, S, K, T, r, q, option_type, 'bisect'),
    }
    print("\n  Comparaison des méthodes de vol. implicite :")
    for nom, vi in methodes.items():
        if vi is not None and not np.isnan(vi):
            prix_verif = black_scholes(S, K, T, r, vi, q, option_type)
            print(f"    {nom:20s} : σ = {vi*100:.4f}%  | BS(σ) = {prix_verif:.4f}  | cible = {prix_marche:.4f}")
        else:
            print(f"    {nom:20s} : échec")


# =============================================================================
# EXERCICE 3 — RÉCUPÉRATION DES DONNÉES D'OPTIONS VIA YFINANCE
# =============================================================================

def recuperer_option_chain(ticker):
    """
    Récupère la chaîne d'options depuis Yahoo Finance.

    ── Comment ça marche ? ───────────────────────────────────────────────────
    yf.Ticker(ticker) crée un objet Ticker.
    .options          retourne la liste des maturités disponibles (dates d'expiration)
    .option_chain(exp) retourne un objet avec deux attributs :
        .calls : DataFrame des calls pour cette maturité
        .puts  : DataFrame des puts pour cette maturité

    Colonnes importantes dans ces DataFrames :
      contractSymbol  : identifiant unique de l'option
      strike          : prix d'exercice
      lastPrice       : dernier prix de transaction
      bid / ask       : fourchette acheteur/vendeur
      impliedVolatility : vol. implicite calculée par Yahoo
      volume          : nombre de contrats échangés aujourd'hui
      openInterest    : nombre de contrats ouverts (proxy de liquidité)
      inTheMoney      : booléen, True si l'option est dans la monnaie
    """
    print(f"\nRécupération des options pour {ticker}...")
    tkr = yf.Ticker(ticker)

    # Prix spot actuel
    info  = tkr.fast_info
    S     = info.last_price
    print(f"  Prix spot : {S:.2f}$")

    # Liste des maturités disponibles
    expirations = tkr.options
    print(f"  Maturités disponibles : {len(expirations)}")
    for i, exp in enumerate(expirations[:6]):
        print(f"    [{i}] {exp}")
    if len(expirations) > 6:
        print(f"    ... et {len(expirations)-6} autres")

    return tkr, S, expirations


# =============================================================================
# EXERCICE 4 — NETTOYAGE DES DONNÉES (filtrage sur la liquidité)
# =============================================================================

def nettoyer_options(df, S, option_type='call', volume_min=10,
                     moneyness_min=0.85, moneyness_max=1.15):
    """
    Filtre les options pour ne garder que les plus liquides et pertinentes.

    ── Critères de liquidité ─────────────────────────────────────────────────
    1. Volume minimal (volume > volume_min)
       → volume = nombre de contrats échangés aujourd'hui
       → volume faible = option peu tradée = spread bid/ask large = prix peu fiable

    2. Moneyness proche de ATM (At-The-Money)
       → moneyness = K / S  (ratio strike / prix spot)
       → proche de 1 = option proche de la monnaie = plus liquide
       → très OTM (K >> S pour call) ou très ITM (K << S) : illiquides

    3. Suppression des valeurs manquantes sur les colonnes clés
    4. Vol. implicite positive et raisonnable (0 < IV < 5 = 500%)

    ── Moneyness ─────────────────────────────────────────────────────────────
    Pour un call :
      ITM (In The Money)  : K < S  → l'option a une valeur intrinsèque
      ATM (At The Money)  : K ≈ S  → point d'équilibre
      OTM (Out The Money) : K > S  → l'option n'a que de la valeur temps
    """
    n_avant = len(df)

    # Filtre 1 : volume minimal
    df = df[df['volume'] >= volume_min]

    # Filtre 2 : moneyness raisonnable (pas trop loin de ATM)
    df = df[(df['strike'] / S >= moneyness_min) &
            (df['strike'] / S <= moneyness_max)]

    # Filtre 3 : valeurs manquantes
    df = df.dropna(subset=['lastPrice', 'impliedVolatility', 'strike'])

    # Filtre 4 : vol. implicite positive
    df = df[(df['impliedVolatility'] > 0) & (df['impliedVolatility'] < 5)]

    # Filtre 5 : prix de l'option positif
    df = df[df['lastPrice'] > 0]

    print(f"  Nettoyage : {n_avant} → {len(df)} options ({option_type})")
    return df.copy().reset_index(drop=True)


# =============================================================================
# EXERCICES 5 — VISUALISATION DU SMILE ET DE LA TERM STRUCTURE
# =============================================================================

def visualiser_smile(calls_propres, puts_propres, S, expiration):
    """
    Affiche le "smile de volatilité" : vol. implicite en fonction du strike.

    ── Qu'est-ce que le smile de volatilité ? ────────────────────────────────
    Dans le modèle BS pur, σ est supposée constante → la vol. implicite devrait
    être la même pour tous les strikes. En pratique, ce n'est pas le cas :

    - Les options OTM put (K << S) ont une vol. implicite plus élevée
      → "crash fear" : les investisseurs paient plus cher pour se protéger
        contre les baisses extrêmes (queue gauche de la distribution)

    - La courbe ressemble à un sourire (smile) ou un smirk (asymétrique)
    → Cela prouve que les rendements réels ne suivent pas une loi normale
      (queue gauche plus épaisse = leptokurtose, skewness négative)
    """
    fig, ax = plt.subplots(figsize=(11, 5))

    if len(calls_propres) > 0:
        ax.scatter(calls_propres['strike'], calls_propres['impliedVolatility'] * 100,
                   label='Calls', color='royalblue', alpha=0.8, s=40)
        # Ligne de tendance
        if len(calls_propres) > 2:
            z = np.polyfit(calls_propres['strike'], calls_propres['impliedVolatility'] * 100, 2)
            p = np.poly1d(z)
            x_line = np.linspace(calls_propres['strike'].min(), calls_propres['strike'].max(), 100)
            ax.plot(x_line, p(x_line), color='royalblue', linewidth=1.5, linestyle='--', alpha=0.7)

    if len(puts_propres) > 0:
        ax.scatter(puts_propres['strike'], puts_propres['impliedVolatility'] * 100,
                   label='Puts', color='tomato', alpha=0.8, s=40)
        if len(puts_propres) > 2:
            z = np.polyfit(puts_propres['strike'], puts_propres['impliedVolatility'] * 100, 2)
            p = np.poly1d(z)
            x_line = np.linspace(puts_propres['strike'].min(), puts_propres['strike'].max(), 100)
            ax.plot(x_line, p(x_line), color='tomato', linewidth=1.5, linestyle='--', alpha=0.7)

    ax.axvline(S, color='green', linewidth=1.5, linestyle=':', label=f'Spot ({S:.1f}$)')
    ax.set_title(f"Smile de volatilité — {TICKER} | Expiration : {expiration}", fontsize=13)
    ax.set_xlabel("Strike ($)")
    ax.set_ylabel("Volatilité implicite (%)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"smile_{TICKER}_{expiration}.png", dpi=150)
    plt.close()
    print(f"  Smile sauvegardé : smile_{TICKER}_{expiration}.png")


def visualiser_term_structure(tkr, S, r, expirations, strike_cible=None):
    """
    Affiche la term structure de la volatilité :
    vol. implicite en fonction de la maturité pour un strike donné.

    ── Term structure ─────────────────────────────────────────────────────────
    La vol. implicite varie aussi avec la maturité :
    - En général, la vol. court terme > vol. long terme (mean-reversion)
    - Autour d'un événement (résultats, BCE...) : pic de vol court terme
    - La forme de cette courbe donne des infos sur les anticipations du marché
    """
    if strike_cible is None:
        # On prend l'ATM approximatif comme strike de référence
        strike_cible = round(S / 5) * 5   # arrondi au multiple de 5 le plus proche

    vols   = []
    mats   = []

    for exp in expirations[:8]:   # on limite à 8 maturités pour la lisibilité
        try:
            chain = tkr.option_chain(exp)
            calls = chain.calls
            calls = calls[(calls['volume'] > 5) & (calls['impliedVolatility'] > 0)]

            # Strike le plus proche du cible
            if calls.empty:
                continue
            idx = (calls['strike'] - strike_cible).abs().idxmin()
            row = calls.loc[idx]

            T = (pd.Timestamp(exp) - pd.Timestamp.today()).days / 365.0
            if T > 0:
                vols.append(row['impliedVolatility'] * 100)
                mats.append(T * 365)   # en jours pour l'axe
        except Exception:
            continue

    if len(vols) < 2:
        print("  Pas assez de données pour la term structure.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(mats, vols, 'o-', color='darkorange', linewidth=2, markersize=7)
    ax.set_title(f"Term structure de la volatilité — {TICKER} | Strike ≈ {strike_cible}$", fontsize=13)
    ax.set_xlabel("Jours jusqu'à l'expiration")
    ax.set_ylabel("Volatilité implicite (%)")
    plt.tight_layout()
    plt.savefig(f"term_structure_{TICKER}.png", dpi=150)
    plt.close()
    print(f"  Term structure sauvegardée : term_structure_{TICKER}.png")


# =============================================================================
# EXERCICES 6, 7, 8 — CALCUL VI + COMPARAISON + PRIX THÉORIQUE VIA df.apply()
# =============================================================================

def calculer_vi_sur_dataframe(df, S, r, option_type='call'):
    """
    Calcule la vol. implicite pour chaque option du DataFrame
    en utilisant df.apply() — sans boucle for explicite.

    ── df.apply() ────────────────────────────────────────────────────────────
    apply() applique une fonction à chaque ligne (axis=1) ou colonne (axis=0).
    C'est vectorisé et beaucoup plus rapide qu'une boucle for en Python.

    Syntaxe :
        df['nouvelle_colonne'] = df.apply(ma_fonction, axis=1)

    La fonction reçoit une Series (= une ligne) et retourne une valeur.
    On accède aux colonnes avec row['nom_colonne'].

    ── Calcul de T (temps jusqu'à expiration) ────────────────────────────────
    On extrait la date d'expiration depuis le contractSymbol.
    Format Yahoo Finance : AAPL231117C00150000
                              ^^^^^^  date = 231117 → 17 nov 2023
    """
    today = pd.Timestamp.today().normalize()

    def _calculer_vi_ligne(row):
        """Fonction appliquée à chaque ligne du DataFrame."""
        try:
            # Extraction de la date d'expiration depuis le symbole du contrat
            # Format : TICKER + YYMMDD + C/P + strike×1000
            symbol = row['contractSymbol']
            date_str = symbol[len(TICKER):][:6]   # 6 chiffres après le ticker
            expiration = pd.Timestamp('20' + date_str)
            T = (expiration - today).days / 365.0

            if T <= 0:
                return np.nan

            prix  = row['lastPrice']
            K     = row['strike']

            return vol_implicite_newton(prix, S, K, T, r, q=0.0,
                                        option_type=option_type)
        except Exception:
            return np.nan

    # apply() sur toutes les lignes (axis=1)
    df['IV_calculee'] = df.apply(_calculer_vi_ligne, axis=1)
    return df


def calculer_prix_bs_sur_dataframe(df, S, r, option_type='call'):
    """
    Calcule le prix Black-Scholes théorique pour chaque option.
    Utilise la vol. implicite calculée (IV_calculee) comme σ.

    ── Pourquoi utiliser IV_calculee et non sigma arbitraire ? ───────────────
    On veut comparer le prix BS calculé avec le prix du marché.
    Si on utilise IV_calculee (= la vol qui donne exactement le prix du marché),
    les deux prix devraient être identiques.
    C'est une vérification de cohérence : BS(IV_calculée) ≈ lastPrice.
    """
    today = pd.Timestamp.today().normalize()

    def _calculer_prix_ligne(row):
        try:
            symbol = row['contractSymbol']
            date_str = symbol[len(TICKER):][:6]
            expiration = pd.Timestamp('20' + date_str)
            T = (expiration - today).days / 365.0

            if T <= 0 or pd.isna(row.get('IV_calculee', np.nan)):
                return np.nan

            return black_scholes(S, row['strike'], T, r,
                                 row['IV_calculee'], q=0.0,
                                 option_type=option_type)
        except Exception:
            return np.nan

    df['Prix_BS'] = df.apply(_calculer_prix_ligne, axis=1)
    return df


def ajouter_comparaison(df):
    """
    Exercice 7 : crée une colonne 'comparaison' = écart entre les deux vol. implicites.

    IV_yahoo   : vol. implicite fournie par Yahoo Finance
    IV_calculee: vol. implicite que l'on a calculée

    Écart possible pour plusieurs raisons :
    1. Prix utilisé différent : Yahoo utilise le mid bid/ask, on utilise lastPrice
       → lastPrice peut être vieux (dernière transaction, pas forcément d'aujourd'hui)
    2. Méthode de calcul différente : Yahoo peut utiliser une autre méthode
    3. Paramètres différents : taux sans risque, dividendes
    4. Précision numérique
    """
    df['IV_yahoo']       = df['impliedVolatility']
    df['IV_calculee_pct'] = df['IV_calculee'] * 100
    df['IV_yahoo_pct']    = df['IV_yahoo'] * 100
    df['comparaison']     = df['IV_calculee_pct'] - df['IV_yahoo_pct']
    df['ecart_prix']      = df['Prix_BS'] - df['lastPrice']
    return df


def afficher_comparaison(df, titre):
    """Résumé statistique de la comparaison."""
    df_clean = df.dropna(subset=['IV_calculee', 'IV_yahoo', 'Prix_BS'])

    if df_clean.empty:
        print("  Pas assez de données pour la comparaison.")
        return

    print(f"\n── {titre} ──")
    print(f"  Options analysées   : {len(df_clean)}")
    print(f"  Écart IV moyen      : {df_clean['comparaison'].mean():+.2f}%")
    print(f"  Écart IV max (abs)  : {df_clean['comparaison'].abs().max():.2f}%")
    print(f"  Écart prix moyen    : {df_clean['ecart_prix'].mean():+.4f}$")
    print()
    print("  Aperçu (5 premières lignes) :")
    cols = ['strike', 'lastPrice', 'Prix_BS', 'IV_yahoo_pct', 'IV_calculee_pct', 'comparaison']
    cols_dispo = [c for c in cols if c in df_clean.columns]
    print(df_clean[cols_dispo].head().to_string(index=False))


# =============================================================================
# PROGRAMME PRINCIPAL
# =============================================================================

if __name__ == "__main__":

    np.random.seed(42)
    R = 0.05   # taux sans risque (environ le taux Fed Funds)

    # ── Exercice 1 : démonstration Black-Scholes ──────────────────────────────
    print("=" * 60)
    print("EX 1 — Prix Black-Scholes")
    print("=" * 60)
    S, K, T, r, sigma = 190.0, 195.0, 30/365, R, 0.25
    print(f"  Paramètres : S={S}, K={K}, T={T:.4f}an, r={r}, σ={sigma}")
    print(f"  Prix Call  : {black_scholes(S, K, T, r, sigma, option_type='call'):.4f}$")
    print(f"  Prix Put   : {black_scholes(S, K, T, r, sigma, option_type='put'):.4f}$")

    # Vérification parité call-put : C - P = S·e^(-qT) - K·e^(-rT)
    C = black_scholes(S, K, T, r, sigma, option_type='call')
    P = black_scholes(S, K, T, r, sigma, option_type='put')
    parite_gauche = C - P
    parite_droite = S - K * np.exp(-r * T)
    print(f"  Parité call-put : C-P={parite_gauche:.4f}  |  S-Ke^(-rT)={parite_droite:.4f}  ✓")

    # ── Exercice 2 : volatilité implicite ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("EX 2 — Volatilité implicite")
    print("=" * 60)
    prix_marche = 5.50
    vi = vol_implicite_newton(prix_marche, S, K, T, r, option_type='call')
    print(f"  Prix marché = {prix_marche}$  →  σ implicite = {vi*100:.4f}%")
    comparer_methodes_vi(prix_marche, S, K, T, r, option_type='call')

    # ── Exercice 3 : récupération des données ─────────────────────────────────
    print("\n" + "=" * 60)
    print("EX 3 — Données options Yahoo Finance")
    print("=" * 60)
    tkr, S_spot, expirations = recuperer_option_chain(TICKER)

    if not expirations:
        print("Impossible de récupérer les options. Vérifiez votre connexion internet.")
        exit(0)

    # On prend la première maturité avec au moins 7 jours restants
    today = pd.Timestamp.today().normalize()
    exp_choisie = next(
        (e for e in expirations
         if (pd.Timestamp(e) - today).days >= 7),
        expirations[1]
    )
    print(f"\n  Maturité choisie : {exp_choisie}")
    chain  = tkr.option_chain(exp_choisie)
    calls  = chain.calls
    puts   = chain.puts
    print(f"  Calls bruts : {len(calls)} | Puts bruts : {len(puts)}")
    print(f"\n  Colonnes disponibles : {list(calls.columns)}")

    # ── Exercice 4 : nettoyage ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EX 4 — Nettoyage des données")
    print("=" * 60)
    calls_propres = nettoyer_options(calls, S_spot, 'call')
    puts_propres  = nettoyer_options(puts,  S_spot, 'put')

    # ── Exercice 5 : smile de volatilité + term structure ─────────────────────
    print("\n" + "=" * 60)
    print("EX 5 — Smile de volatilité & term structure")
    print("=" * 60)
    visualiser_smile(calls_propres, puts_propres, S_spot, exp_choisie)
    visualiser_term_structure(tkr, S_spot, R, expirations)

    # ── Exercices 6, 7, 8 ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EX 6, 7, 8 — VI calculée, comparaison, prix théorique")
    print("=" * 60)

    for df_opt, opt_type, label in [
        (calls_propres, 'call', 'CALLS'),
        (puts_propres,  'put',  'PUTS'),
    ]:
        if df_opt.empty:
            continue

        # Ex 6 : calcul de la vol. implicite via notre Newton
        df_opt = calculer_vi_sur_dataframe(df_opt, S_spot, R, opt_type)

        # Ex 8 : calcul du prix BS théorique
        df_opt = calculer_prix_bs_sur_dataframe(df_opt, S_spot, R, opt_type)

        # Ex 7 : colonne comparaison
        df_opt = ajouter_comparaison(df_opt)

        afficher_comparaison(df_opt, label)

    print("\nTous les graphiques ont été sauvegardés.")
