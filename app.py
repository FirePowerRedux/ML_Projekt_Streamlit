import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Ustawienia strony
st.set_page_config(page_title="Steam Risk Predictor")


# 1) ŁADOWANIE MODELU za pomocą cache, żeby nie wczytywać pliku za każdym rerun

@st.cache_resource
def load_artifact(path: str = "model.joblib"):
    """
    Wczytuje zapisany plik z dysku, 
    cache_resource sprawia, że Streamlit nie wczytuje tego pliku przy każdej zmianie w UI
    """
    return joblib.load(path)

artifact = load_artifact()
model = artifact["model"]          # np. Pipeline( preprocessing i następnie classifier )

# branie dokładnie tych kolumn, na których model był fitowany 
FEATURES = list(getattr(model, "feature_names_in_", artifact.get("features", [])))


# Teksty na stronie
st.title(" Predykcja ryzyka słabego odbioru gry (low_reception)")
st.write("Wpisz dane gry i kliknij **Predykcja**. Dostaniesz klasę i prawdopodobieństwo.")


# 2) FUNKCJE POMOCNICZE, w postaci małych czytelnych klocków

def log1p0(x: float) -> float:
    """
    Zwraca log(1+x) czyli log1p.
    Takie przekształcenie często zmniejsza wpływ skrajnie dużych wartości ( jak w przypadku np. Peak CCU)
    Zabezpieczenie max(0, x) chroni przed przypadkiem, gdy ktoś poda wartość ujemną
    """
    return float(np.log1p(max(0.0, float(x))))

def build_features_row(inp: dict) -> pd.DataFrame:
    """
    Buduje jeden wiersz danych (DataFrame 1xN) w formacie oczekiwanym przez model
    Kluczowe jest, aby końcowy DF posiadał dokładnie kolumny FEATURES, czyli te same nazwy i kolejność
    """
    if "Score rank" in FEATURES and "Score rank" not in row:
    row["Score rank"] = np.nan
    #  Bazowe cechy bezpośrednio z inputów
    row = {
        "Price": inp["price"],
        "Required age": inp["required_age"],
        "Achievements": inp["achievements"],
        "DiscountDLC count": inp["discount_dlc"],
        # checkboxy -> int (True/False zamieniamy na 1/0)
        "Windows": int(inp["windows"]),
        "Mac": int(inp["mac"]),
        "Linux": int(inp["linux"]),
    }

    # Cecha pochodna "ile platform wspiera gra"
    row["platform_count"] = row["Windows"] + row["Mac"] + row["Linux"]

    # Metacritic ze specjalną logiką
    # W UI 0 oznacza brak danych
    # W cechach trzymane są (1) flagę has_metacritic i (2) Metacritic score jako NaN jeśli brak
    m = inp["metacritic"]
    row["has_metacritic"] = 1 if m > 0 else 0
    row["Metacritic score"] = np.nan if m == 0 else float(m)

    # Logarytmowane cechy liczbowe
    # Zwykle log1p jest używany, gdy zmienne mają duże rozrzuty, tak jak np. CCU, rekomendacje, playtime
    row["log_peak_ccu"] = log1p0(inp["peak_ccu"])
    row["log_recommendations"] = log1p0(inp["recommendations"])
    row["log_average_playtime_forever"] = log1p0(inp["avg_playtime_forever"])

    # Uzupełnianie brakujących kolumn
    # Ustawiane są brakujące wartości na 0.0.
    for col in FEATURES:
        if col not in row:
            row[col] = 0.0

    # Zwracanie DF z kolumnami w dokładnie tej samej kolejności, jakiej oczekuje model
    return pd.DataFrame([row], columns=FEATURES)

# 3) UI FORMULARZ, form sprawia, że predykcja włącza się dopiero po kliknięciu
with st.form("game_inputs"):
    # Liczbowe parametry gry
    price = st.number_input("Cena (Price)", min_value=0.0, value=49.99, step=1.0)
    required_age = st.number_input("Wymagany wiek (Required age)", min_value=0, value=0, step=1)

    peak_ccu = st.number_input("Peak CCU", min_value=0.0, value=1000.0, step=100.0)
    recommendations = st.number_input("Recommendations", min_value=0.0, value=200.0, step=10.0)
    avg_playtime_forever = st.number_input("Average playtime forever (minuty)", min_value=0.0, value=300.0, step=10.0)

    # Metacritic (0 oznacza brak danych)
    metacritic = st.number_input("Metacritic score (0 = brak danych)", min_value=0, max_value=100, value=0, step=1)

    # Checkboxy (platformy)
    c1, c2, c3 = st.columns(3)
    with c1:
        windows = st.checkbox("Windows", value=True)
    with c2:
        mac = st.checkbox("Mac", value=False)
    with c3:
        linux = st.checkbox("Linux", value=False)

    # Pozostałe cechy
    achievements = st.number_input("Achievements", min_value=0, value=0, step=1)
    discount_dlc = st.number_input("DiscountDLC count", min_value=0, value=0, step=1)

    # Przycisk w formularzu, jedyny moment, gdy włączana jest predykcja
    submitted = st.form_submit_button("Predykcja")

# 4) PREDYKCJA (dopiero po kliknięciu)
if submitted:
    # Zbieranie inputów do jednego słownika, żeby łatwo przekazać je do build_features_row
    inputs = dict(
        price=price,
        required_age=required_age,
        peak_ccu=peak_ccu,
        recommendations=recommendations,
        avg_playtime_forever=avg_playtime_forever,
        metacritic=metacritic,
        windows=windows,
        mac=mac,
        linux=linux,
        achievements=achievements,
        discount_dlc=discount_dlc,
    )

    # Budowa DataFrame w formacie oczekiwanym przez model
    X = build_features_row(inputs)

    # Predykcja klasy (0/1)
    pred_class = int(model.predict(X)[0])

    # Predykcja prawdopodobieństwa
    if hasattr(model, "predict_proba"):
        proba_1 = float(model.predict_proba(X)[0][1])
        st.write(f"**Prawdopodobieństwo low_reception=1:** {proba_1:.3f}")
    else:
        st.write("Model nie obsługuje predict_proba().")

    # Prezentacja wyniku w UI
    if pred_class == 1:
        st.error("Wynik: **1 (ryzyko słabego odbioru)**")
    else:
        st.success("Wynik: **0 (brak sygnału ryzyka)**")

    st.caption("Uwaga: model pomocniczy — traktuj jako filtr do ręcznej weryfikacji.")
