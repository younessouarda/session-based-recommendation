import streamlit as st
import torch
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from model_gru import GRUItemEventModel
import os
import sklearn.preprocessing
import torch.serialization
import numpy

# === CONFIGURATION ===
MODEL_PATH = "models/Modele_GRU_Complet.pth"
DATA_PATH = "data/cleaned_sessions.csv"
MAX_SEQ_LEN = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Charger modèle et encoders ===

@st.cache_resource
def load_model_and_data():
    import numpy as np
    import sklearn.preprocessing
    import types

    with torch.serialization.safe_globals([
        np.ndarray,
        np.core.multiarray._reconstruct,
        sklearn.preprocessing._label.LabelEncoder,
        types.BuiltinFunctionType,
    ]):
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

    df = pd.read_csv(DATA_PATH)
    df = df[df["event"].isin(["view", "addtocart", "transaction"])]
    df = df.dropna(subset=["itemid", "visitorid", "event"])
    df = df.sort_values(["visitorid", "timestamp"])

    if 'item_encoder' not in checkpoint or 'event_encoder' not in checkpoint:
        raise ValueError("Le fichier .pth ne contient pas les encoders nécessaires (item_encoder, event_encoder).")

    item_encoder = checkpoint["item_encoder"]
    event_encoder = checkpoint["event_encoder"]

    if "itemid_enc" not in df.columns:
        df["itemid_enc"] = item_encoder.transform(df["itemid"])
    if "event_type" not in df.columns:
        df["event_type"] = df["event"].map(event_encoder)


    model = GRUItemEventModel(len(item_encoder.classes_), 3, 200, 200)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE).eval()

    return model, df, item_encoder, event_encoder

# === Prédiction d'un utilisateur
def predict_next(model, session, item_encoder, event_encoder):
    items = [x[0] for x in session][-MAX_SEQ_LEN:]
    events = [x[1] for x in session][-MAX_SEQ_LEN:]
    item_tensor = torch.LongTensor(items).unsqueeze(0).to(DEVICE)
    event_tensor = torch.LongTensor(events).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        item_logits, event_logits = model(item_tensor, event_tensor)
        probs_items = torch.softmax(item_logits[0, -1], dim=0)
        probs_event = torch.softmax(event_logits[0, -1], dim=0)
        top_items = torch.topk(probs_items, 5).indices.tolist()
        predicted_event = torch.argmax(probs_event).item()
    return top_items, predicted_event

# === Interface Streamlit
st.title("Démonstration de Recommandation basée sur les sessions")
try:
    model, df, item_encoder, event_encoder = load_model_and_data()
except Exception as e:
    st.error(str(e))
    st.stop()

idx2item = {i: item for item, i in zip(item_encoder.classes_, range(len(item_encoder.classes_)))}
inv_event_encoder = {v: k for k, v in event_encoder.items()}

visitor_ids = df["visitorid"].dropna().unique().tolist()
visitor = st.selectbox("Sélectionnez un visitor ID", visitor_ids[:1000])

user_df = df[df["visitorid"] == visitor].sort_values("timestamp")
if len(user_df) < 2:
    st.warning("Pas assez d'interactions pour cette session.")
    st.stop()

session = list(zip(user_df["itemid_enc"], user_df["event_type"]))
recent = user_df[["timestamp", "itemid", "event"]].tail(10)

st.subheader("Historique récent")
st.dataframe(recent)

top5_ids, event_type_pred = predict_next(model, session, item_encoder, event_encoder)
top5_items = [idx2item[i] for i in top5_ids]
pred_action = inv_event_encoder[event_type_pred]

st.subheader("Recommandations")
for i, item in enumerate(top5_items):
    st.write(f"**#{i+1}** → Item ID: `{item}`")

st.success(f"Action probable : **{pred_action.upper()}**")
